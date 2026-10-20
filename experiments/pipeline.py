"""
The experiment itself, as plain functions. run.py only adds the CLI, workers and
bookkeeping around run_experiment().

Per run:
    1. load the full dataset and the (cached) variant indices
    2. load or train the round-0 critic (cached: identical for every targeting rule)
    3. for each round: pick starts (T), generate chains, label with h (L), filter (F),
       merge, fine-tune, evaluate
    4. optionally: train the FISOR actor and reward critic on the real data with the
       frozen feasibility critic, then roll the policy out in the environment
"""

import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from data.offline_dataset import OfflineDataset
from data.utils import shape_h
from data.variant import (
    coverage_stats,
    coverage_variant_indices,
    size_matched_variant_indices,
)
from dynamics.ensemble import DynamicsEnsemble, train_dynamics_ensemble
from FISOR.ensemble_feasibility_critic import EnsembleFeasibilityCritic
from generation.context import GenContext, LearnedStep, TrueStep
from generation.generator import generate
from generation.rollout import concat_raw

from . import paths
from .config import ExperimentConfig
from .envs import ENVS, EnvSpec

Log = Callable[[str], None]


# --------------------------------------------------------------------- small utils
def atomic_torch_save(obj, path: str) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path: str) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


class FileLock:
    """
    Cross-process lock via O_EXCL file creation (works on GPFS / Lustre).

    Used so that two workers that need the same cached round-0 critic or dynamics model
    do not both train it: the second one waits for the first one's checkpoint. A lock
    older than `stale_after` seconds (holder crashed or was killed) is taken over.
    """

    def __init__(self, path: str, stale_after: float = 6 * 3600, poll: float = 15.0):
        self.path, self.stale_after, self.poll = path + ".lock", stale_after, poll
        self.fd = None

    def acquire(self, done: Callable[[], bool]) -> bool:
        """True if acquired; False if `done()` became true while waiting."""
        while True:
            if done():
                return False
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"{os.getpid()}@{os.uname().nodename}".encode())
                return True
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > self.stale_after:
                        os.remove(self.path)
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(self.poll)

    def release(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------- data
def h_for(spec: EnvSpec, cfg: ExperimentConfig):
    if cfg.h_pos_scale == 1.0 and cfg.h_neg_scale == 1.0:
        return spec.h_fn
    return shape_h(spec.h_fn, pos_scale=cfg.h_pos_scale, neg_scale=cfg.h_neg_scale)


def load_full_dataset(spec: EnvSpec, cfg: ExperimentConfig, device: str) -> OfflineDataset:
    # Truncations (timeouts) keep mask = 1 so the critic bootstraps through them.
    return OfflineDataset(
        spec.load_raw(),
        device=device,
        n_members=cfg.K,
        seed=cfg.seed,
        timeout_is_done=False,
        h_fn=h_for(spec, cfg),
    )


def variant_indices(spec: EnvSpec, full: OfflineDataset, cfg: ExperimentConfig) -> np.ndarray:
    path = paths.variant_path(cfg.variant_key)
    if os.path.exists(path):
        return np.load(path)

    cov_idx = coverage_variant_indices(full, cfg.coverage, spec.boundary, seed=cfg.seed)
    if cfg.variant == "coverage":
        idx = cov_idx
    elif cfg.variant == "size_matched":
        # Same number of transitions as the coverage variant, but trajectories are
        # removed at random: separates "less boundary data" from "less data".
        idx = size_matched_variant_indices(full, len(cov_idx), seed=cfg.seed)
    else:
        raise ValueError(f"unknown variant {cfg.variant!r}")

    tmp = f"{path}.tmp{os.getpid()}.npy"
    np.save(tmp, idx)
    os.replace(tmp, path)
    return idx


# --------------------------------------------------------------------- critic
def make_critic(spec_dims: Tuple[int, int], cfg: ExperimentConfig, device: str) -> EnsembleFeasibilityCritic:
    obs_dim, act_dim = spec_dims
    return EnsembleFeasibilityCritic(
        obs_dim,
        act_dim,
        K=cfg.K,
        hidden_dims=tuple(cfg.hidden_dims),
        discount=cfg.discount,
        expectile=cfg.expectile,
        critic_lr=cfg.critic_lr,
        value_lr=cfg.critic_lr,
        critic_batch_size=cfg.batch_size,
        device=device,
    )


def train_critic(critic, dataset: OfflineDataset, steps: int, batch_size: int, log: Log, log_every: int, label: str) -> None:
    t0 = time.time()
    for step in range(1, steps + 1):
        info = critic.update(dataset.sample(batch_size))
        if step % log_every == 0 or step == steps:
            log(
                f"  [{label}] step {step}/{steps} vc={info['vc']:.3f} "
                f"qc_loss={info['safe_critic_loss']:.4f} ({time.time() - t0:.0f}s)"
            )


def evaluate_critic(spec: EnvSpec, critic, full_real: OfflineDataset) -> List[dict]:
    from evaluation.critic import evaluate_controlled, evaluate_dsrl_proxy

    if spec.has_ground_truth:
        return evaluate_controlled(critic, full_real)
    return evaluate_dsrl_proxy(critic, full_real, full_real.h_fn)


def round0_critic(spec, cfg, full, data, dims, device, log) -> Tuple[EnsembleFeasibilityCritic, List[dict]]:
    """Load the shared round-0 critic, or train and cache it."""
    path = paths.round0_path(cfg.round0_key)
    critic = make_critic(dims, cfg, device)
    lock = FileLock(path)
    if lock.acquire(done=lambda: os.path.exists(path)):
        try:
            log(f"training round-0 critic {cfg.round0_key} ({cfg.round0_steps} steps)")
            set_seed(cfg.seed)
            train_critic(critic, data, cfg.round0_steps, cfg.batch_size, log, cfg.log_every, "round0")
            records = evaluate_critic(spec, critic, full)
            atomic_torch_save({"critic": critic.state_dict(), "records": records}, path)
            return critic, records
        finally:
            lock.release()

    ckpt = torch.load(path, map_location=device, weights_only=False)
    critic.load_state_dict(ckpt["critic"])
    log(f"loaded round-0 critic {cfg.round0_key}")
    return critic, ckpt["records"]


# --------------------------------------------------------------------- dynamics
def get_step_model(spec, cfg, data, device, log):
    if cfg.dynamics == "true":
        if spec.true_step is None:
            raise ValueError(f"{spec.name} has no true dynamics; use dynamics=learned")
        return TrueStep(spec.true_step)

    path = paths.dynamics_path(cfg.dynamics_key)
    lock = FileLock(path)
    if lock.acquire(done=lambda: os.path.exists(path)):
        try:
            log(f"training dynamics ensemble {cfg.dynamics_key}")
            model, info = train_dynamics_ensemble(
                data, K=cfg.dyn_K, hidden_dims=tuple(cfg.dyn_hidden_dims), seed=cfg.seed,
                max_epochs=cfg.dyn_max_epochs, steps_per_epoch=cfg.dyn_steps_per_epoch, log_fn=log,
            )
            log(f"  dynamics: {info}")
            model.save(path)
        finally:
            lock.release()
    else:
        model = DynamicsEnsemble.load(path, device)
        log(f"loaded dynamics ensemble {cfg.dynamics_key}")
    return LearnedStep(model, threshold_scale=cfg.disagreement_scale)


def make_context(spec, cfg, data: OfflineDataset, step_model, device) -> GenContext:
    real_obs = data.data["observations"]
    real_act = data.data["actions"]
    g = torch.Generator(device=device)
    g.manual_seed(cfg.seed)
    return GenContext(
        h_fn=data.h_fn,
        valid_fn=spec.make_valid_fn(real_obs),
        terminal_fn=spec.terminal_fn,
        step_model=step_model,
        act_low=real_act.min(dim=0).values,
        act_high=real_act.max(dim=0).values,
        data_obs=real_obs,
        data_act=real_act,
        device=torch.device(device),
        generator=g,
    )


def synthetic_quality(spec, raw) -> List[dict]:
    """Controlled env only: where did the synthetic data actually land?"""
    if not spec.has_ground_truth or len(raw["observations"]) == 0:
        return []
    from controlled.env_api import _ground_truth, check_feasible

    s = raw["observations"]
    gt = _ground_truth()
    m = {
        "in_band_frac": float(gt.in_band(s, eps=0.1).mean()),
        "truly_feasible_frac": float(check_feasible(s).mean()),
    }
    return [{"eval_set": "synthetic", "metric": k, "value": v} for k, v in m.items()]


# --------------------------------------------------------------------- policy
def policy_stage(spec, cfg, critic, data: OfflineDataset, out_dir: str, device: str, log: Log) -> List[dict]:
    from evaluation.policy import evaluate_policy
    from FISOR.agent import FISORAgent, FISORConfig

    if spec.make_eval_env is None:
        return []
    real = data.real_only()
    agent_cfg = FISORConfig(
        critic_hidden_dims=tuple(cfg.hidden_dims),
        batch_size=cfg.policy_batch_size,
        critic_batch_size=cfg.batch_size,
        decay_steps=cfg.policy_steps,
    )
    obs_dim, act_dim = real.data["observations"].shape[1], real.data["actions"].shape[1]
    agent = FISORAgent(
        obs_dim, act_dim, agent_cfg, device=device,
        feasibility_critic=critic, train_feasibility=False,
    )
    ckpt = os.path.join(out_dir, "agent.pt")
    if os.path.exists(ckpt):
        agent.load(ckpt)
        log(f"resuming policy training at step {agent.step}")

    t0 = time.time()
    save_every = max(cfg.log_every * 4, 1)
    while agent.step < cfg.policy_steps:
        info = agent.update(real.sample(cfg.policy_batch_size))
        if agent.step % cfg.log_every == 0:
            log(f"  [policy] step {agent.step}/{cfg.policy_steps} ({time.time() - t0:.0f}s)")
        if agent.step % save_every == 0:
            agent.save(ckpt)
    agent.save(ckpt)

    log(f"evaluating policy over {cfg.eval_episodes} episodes")
    return evaluate_policy(agent, spec.make_eval_env, cfg.eval_episodes, spec.target_cost, seed=10_000 + cfg.seed)


# --------------------------------------------------------------------- the run
def run_experiment(cfg: ExperimentConfig, out_dir: str, device: str, log: Log) -> List[dict]:
    spec = ENVS[cfg.env]
    set_seed(cfg.seed)

    full = load_full_dataset(spec, cfg, device)
    data = full.subset(variant_indices(spec, full, cfg))
    dims = (data.data["observations"].shape[1], data.data["actions"].shape[1])
    log(f"variant {cfg.variant} {cfg.coverage}%: {len(data)}/{len(full)} transitions")

    records: List[dict] = []

    def add(round_: int, recs: List[dict]):
        records.extend(dict(r, round=round_) for r in recs)

    cov = coverage_stats(data, spec.boundary, reference=full)
    add(0, [{"eval_set": "dataset", "metric": k, "value": float(v)} for k, v in cov.items()])

    # ---- resume from the last completed round if this run was interrupted
    state_path = os.path.join(out_dir, "state.pt")
    critic = make_critic(dims, cfg, device)
    synth_parts: List[Dict[str, np.ndarray]] = []
    start_round = 1
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        critic.load_state_dict(st["critic"])
        records, synth_parts, start_round = st["records"], st["synth"], st["round"] + 1
        log(f"resuming after round {st['round']}")
    else:
        critic, r0 = round0_critic(spec, cfg, full, data, dims, device, log)
        add(0, r0)

    total_budget = int(round(cfg.budget_frac * len(data)))
    per_round = total_budget // max(cfg.rounds, 1)
    step_model = None
    if cfg.rule != "none" and per_round > 0:
        step_model = get_step_model(spec, cfg, data, device, log)
        ctx = make_context(spec, cfg, data, step_model, device)

    for r in range(start_round, cfg.rounds + 1):
        log(f"round {r}/{cfg.rounds}")
        if step_model is not None:
            set_seed(cfg.seed * 1000 + r)
            raw, stats = generate(
                cfg.rule, data.data["observations"], per_round, critic, ctx,
                seed=cfg.seed * 1000 + r, branching=cfg.branching, chain_len=cfg.chain_len,
                n_candidates=cfg.n_candidates, horizon=cfg.lookahead, nn_noise=cfg.nn_noise,
                policy_noise=cfg.policy_noise, log_fn=log,
                uncertainty=cfg.uncertainty, candidate_multiplier=cfg.candidate_multiplier,
                beta=cfg.beta, uniform_frac=spec.uniform_frac, noise_scale=cfg.noise_scale,
            )
            synth_parts.append(raw)
            add(r, [{"eval_set": "generation", "metric": k, "value": float(v)} for k, v in stats.items()])
            add(r, synthetic_quality(spec, raw))

        train_data = data
        if synth_parts and sum(len(p["observations"]) for p in synth_parts):
            train_data = data.with_synthetic(concat_raw(synth_parts), label_noise=cfg.label_noise)
        log(f"  training on {len(train_data)} transitions ({train_data.n_synthetic} synthetic)")

        if cfg.reset_critic_each_round:
            critic = make_critic(dims, cfg, device)
            train_critic(critic, train_data, cfg.round0_steps, cfg.batch_size, log, cfg.log_every, f"round{r}")
        else:
            train_critic(critic, train_data, cfg.steps_per_round, cfg.batch_size, log, cfg.log_every, f"round{r}")
        add(r, evaluate_critic(spec, critic, full))

        atomic_torch_save({"critic": critic.state_dict(), "records": records, "synth": synth_parts, "round": r}, state_path)

    if cfg.policy:
        add(cfg.rounds, policy_stage(spec, cfg, critic, data, out_dir, device, log))

    torch.save(critic.state_dict(), os.path.join(out_dir, "critic_final.pt"))
    return records
