"""
Stage C: one array task runs `per_task` consecutive manifest entries.

Every run is idempotent and resumable:
  results/<run_id>/config.json     written first
  results/<run_id>/rounds.jsonl    one line per round (round 0 = baseline metrics)
  results/<run_id>/round_<r>.pt    critic + synthetic buffer after round r
  results/<run_id>/final.json      final metrics (+ policy metrics if train_agent)
  results/<run_id>/map.npz         feasibility maps (controlled)
  results/<run_id>/DONE            written last
  results/<run_id>/FAILED          traceback, if it crashed (removed on retry)

Usage:
  python -m experiments.run --manifest manifests/runs_cpu.jsonl --index $SLURM_ARRAY_TASK_ID --per-task 8
  python -m experiments.run --manifest manifests/runs_cpu.jsonl --run-id <id>     # debug one run
"""

import argparse
import json
import sys
import traceback
from typing import Any, Dict, List, Optional

from .config import ExperimentConfig
from .grid import read_manifest
from .paths import baseline_done_marker, done_marker, run_dir


def _write_json(path, obj) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=float)
    import os

    os.replace(tmp, path)


def _append_jsonl(path, obj) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True, default=float) + "\n")


def last_completed_round(cfg: ExperimentConfig) -> int:
    """Highest r with round_<r>.pt present (-1 if none): where to resume."""
    rounds = [int(p.stem.split("_")[1]) for p in run_dir(cfg).glob("round_*.pt")]
    return max(rounds) if rounds else -1


def run_experiment(cfg: ExperimentConfig, device: str) -> Dict[str, Any]:
    # Imports here so `--help` and manifest tooling work without torch.
    import torch

    from data.mixed import MixedSampler, SyntheticBuffer
    from envs.registry import get_env_spec, make_training_h, make_valid_fn
    from eval.evaluate import (evaluate_feasibility, evaluate_policy,
                               evaluation_states, feasibility_map,
                               uncertainty_calibration)
    from FISOR.ensemble_feasibility_critic import EnsembleFeasibilityCritic
    from synthetic.augment import (generate_round, make_candidate_fn,
                                   synthetic_stats)
    from synthetic.rollout import make_action_proposer, make_transition_model
    from train.loops import (build_agent, train_feasibility_critic,
                             train_fisor_agent)

    from .paths import baseline_path, dynamics_path
    from .prepare import load_variant, resolve_device

    device = resolve_device(device)

    d = run_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    _write_json(d / "config.json", cfg.to_dict())

    spec = get_env_spec(cfg.env)
    training_h = make_training_h(spec, cfg.dense_h, cfg.h_pos_scale, cfg.h_neg_scale)
    dataset = load_variant(cfg, device)
    eval_states = evaluation_states(spec)  # fixed sets cached by stage A
    valid_fn = make_valid_fn(spec, dataset.data["observations"].cpu().numpy())
    obs_dim = dataset.data["observations"].shape[1]
    act_dim = dataset.data["actions"].shape[1]

    # --- round 0: shared baseline, or resume -------------------------------
    critic = EnsembleFeasibilityCritic(obs_dim, act_dim, K=cfg.ensemble_size,
                                       discount=cfg.discount, expectile=cfg.expectile,
                                       device=device)
    buffer = SyntheticBuffer(obs_dim, act_dim, cfg.ensemble_size, critic.device)
    start = last_completed_round(cfg) + 1
    if start == 0:
        critic.load_state_dict(torch.load(baseline_path(cfg), map_location=device, weights_only=False))
        m0 = evaluate_feasibility(critic, spec, eval_states, training_h)
        _append_jsonl(d / "rounds.jsonl", {"round": 0, **m0})
        torch.save({"critic": critic.state_dict(), "buffer": buffer.state_dict()},
                   d / "round_0.pt")
        start = 1
    else:
        ck = torch.load(d / f"round_{start - 1}.pt", map_location=device, weights_only=False)
        critic.load_state_dict(ck["critic"])
        buffer.load_state_dict(ck["buffer"])

    model = make_transition_model(cfg.transition_model, spec, str(dynamics_path(cfg)), device)
    proposer = make_action_proposer(cfg.action_proposer, spec, model, dataset, training_h)
    candidate_fn = make_candidate_fn(spec, dataset, model, valid_fn)

    # --- rounds 1..n --------------------------------------------------------
    for r in range(start, cfg.n_rounds + 1):
        stats: Dict[str, float] = {}
        if cfg.rule != "none":
            out = generate_round(cfg, r - 1, spec, dataset, critic, model, proposer,
                                 training_h, valid_fn, candidate_fn)
            if out.transitions is not None:
                buffer.add_transitions(out.transitions)
            if out.anchor_states is not None:
                buffer.add_anchors(out.anchor_states, out.anchor_h)
            stats = synthetic_stats(out, spec, dataset)

        if cfg.retrain_each_round:
            critic = EnsembleFeasibilityCritic(obs_dim, act_dim, K=cfg.ensemble_size,
                                               discount=cfg.discount, expectile=cfg.expectile,
                                               device=device)
        steps = cfg.base_steps if cfg.retrain_each_round else cfg.round_steps
        sampler = MixedSampler(dataset, buffer, cfg.synthetic_ratio, seed=cfg.seed + r)
        critic = train_feasibility_critic(critic, sampler, steps,
                                          anchor_weight=cfg.anchor_weight if cfg.use_anchors else 0.0,
                                          checkpoint_path=str(d / f"round_{r}.partial"),
                                          checkpoint_every=10_000)

        m = evaluate_feasibility(critic, spec, eval_states, training_h)
        _append_jsonl(d / "rounds.jsonl", {"round": r, **m, **{"syn/" + k: v for k, v in stats.items()}})
        torch.save({"critic": critic.state_dict(), "buffer": buffer.state_dict()}, d / f"round_{r}.pt")

    # --- final --------------------------------------------------------------
    final: Dict[str, Any] = {"feasibility": evaluate_feasibility(critic, spec, eval_states, training_h)}
    if spec.has_ground_truth:
        from controlled.env_api import check_feasible

        s = eval_states["random"]
        final["calibration"] = uncertainty_calibration(critic, s, check_feasible(s))
        import numpy as np

        np.savez_compressed(d / "map.npz",
                            v_h=feasibility_map(critic, quantity="v_h"),
                            entropy=feasibility_map(critic, quantity="entropy"))

    if cfg.train_agent:
        agent = build_agent(obs_dim, act_dim, critic, cfg.discount, device)
        sampler = MixedSampler(dataset, buffer, cfg.synthetic_ratio, seed=cfg.seed)
        agent = train_fisor_agent(agent, sampler, cfg.agent_steps,
                                  anchor_weight=cfg.anchor_weight if cfg.use_anchors else 0.0,
                                  use_synthetic_actions=cfg.agent_uses_synthetic_actions,
                                  checkpoint_path=str(d / "agent.pt"))
        final["policy"] = evaluate_policy(agent, spec, cfg.eval_episodes, seed=10_000 + cfg.seed)

    _write_json(d / "final.json", final)
    for p in d.glob("round_*.pt"):  # keep disk usage bounded; keep the last one
        if p.name != f"round_{cfg.n_rounds}.pt":
            p.unlink()
    done_marker(cfg).touch()
    return final


def select_tasks(configs: List[ExperimentConfig], index: int, per_task: int) -> List[ExperimentConfig]:
    return configs[index * per_task : (index + 1) * per_task]


def run_many(configs: List[ExperimentConfig], device: str, skip_done: bool = True) -> int:
    """Returns the number of failed runs; never stops at the first failure."""
    import random

    n_failed = 0
    for cfg in configs:
        if skip_done and done_marker(cfg).exists():
            print(f"[skip] {cfg.run_id}")
            continue
        if not baseline_done_marker(cfg).exists():
            print(f"[missing baseline] {cfg.run_id} needs {cfg.baseline_key}", file=sys.stderr)
            n_failed += 1
            continue
        failed = run_dir(cfg) / "FAILED"
        try:
            random.seed(cfg.seed)
            from data.utils import set_seed

            set_seed(cfg.seed)
            print(f"[run] {cfg.run_id}")
            run_experiment(cfg, device)
            if failed.exists():
                failed.unlink()
        except Exception:
            n_failed += 1
            run_dir(cfg).mkdir(parents=True, exist_ok=True)
            failed.write_text(traceback.format_exc())
            print(f"[FAILED] {cfg.run_id}\n{traceback.format_exc()}", file=sys.stderr)
    return n_failed


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--per-task", type=int, default=1)
    p.add_argument("--run-id", help="run only this run_id from the manifest")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-skip", action="store_true")
    args = p.parse_args(argv)

    configs = read_manifest(args.manifest)
    if args.run_id:
        configs = [c for c in configs if c.run_id == args.run_id]
    else:
        configs = select_tasks(configs, args.index, args.per_task)

    sys.exit(1 if run_many(configs, args.device, skip_done=not args.no_skip) else 0)


if __name__ == "__main__":
    main()
