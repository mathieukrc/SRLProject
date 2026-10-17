"""
One augmentation round: score -> pick chain starts -> roll out -> label -> filter,
repeated until the round's budget of ACCEPTED transitions is met, plus anchor states.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from data import synthetic as pickers
from data.offline_dataset import OfflineDataset
from data.utils import HFn
from envs.registry import EnvSpec
from experiments.config import ExperimentConfig

from .rollout import (ActionProposer, SyntheticTransitions, filter_transitions,
                      label_transitions, random_actions, rollout_chains, select,
                      truncate_chains)

CandidateFn = Callable[[int], torch.Tensor]


# ----------------------------------------------------------------------------
# Scores used by the pickers
# ----------------------------------------------------------------------------
def make_uncertainty_fn(critic, kind: str = "entropy"):
    if kind == "entropy":
        return lambda s: critic.feasibility_entropy(s).float()
    if kind == "std":
        return lambda s: critic.v_h_members(s).std(dim=0).float()
    raise ValueError(kind)


def make_violation_fn(critic, training_h: HFn):
    def violation(s: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.relu(training_h(s).reshape(-1) - critic.v_h(s).reshape(-1)).float()

    return violation


def make_candidate_fn(
    spec: EnvSpec,
    dataset: OfflineDataset,
    model,
    valid_fn,
    noise_scale: float = 0.05,
    generator: Optional[torch.Generator] = None,
) -> Optional[CandidateFn]:
    """'box' envs -> None (pickers keep their uniform + noise pool).
    'dataset' envs -> half real states, half 1-3 random-action model steps from them."""
    if spec.candidate_source == "box":
        return None
    states = dataset.data["observations"]
    act_dim = dataset.data["actions"].shape[-1]

    def candidates(n: int) -> torch.Tensor:
        idx = torch.randint(len(states), (n,), device=states.device, generator=generator)
        s = states[idx].clone()
        moved = torch.arange(n, device=s.device) >= n // 2
        k = int(torch.randint(1, 4, (1,), generator=None).item())
        sm = s[moved]
        for _ in range(k):
            sm, _ = model.step(sm, random_actions(sm, act_dim, generator), generator)
        s[moved] = sm
        return s

    return candidates


class _States:
    """Minimal stand-in for OfflineDataset (the pickers only read observations)."""

    def __init__(self, observations: torch.Tensor) -> None:
        self.data = {"observations": observations}


def _subsample(dataset: OfflineDataset, seed: int, max_n: int = 50000):
    """GMM fitting on millions of DSRL states each attempt is slow; 50k is plenty."""
    obs = dataset.data["observations"]
    if len(obs) <= max_n:
        return dataset
    g = torch.Generator(device=obs.device)
    g.manual_seed(seed)
    idx = torch.randperm(len(obs), device=obs.device, generator=g)[:max_n]
    return _States(obs[idx])


def pick_chain_starts(
    rule: str,
    dataset: OfflineDataset,
    n: int,
    *,
    critic,
    training_h: HFn,
    valid_fn,
    candidate_fn: Optional[CandidateFn],
    candidate_multiplier: int,
    beta: Optional[float],
    seed: int,
) -> torch.Tensor:
    common = dict(seed=seed, valid_fn=valid_fn)
    if n == 0:
        return dataset.data["observations"][:0].clone()
    if rule == "random":
        if candidate_fn is not None:  # uniform in a lidar box is meaningless
            return pickers._fill_valid(candidate_fn, n, valid_fn)
        return pickers.random_chain_start_picker(dataset, n, **common)
    if rule == "density_matched":
        return pickers.density_matched_chain_start_picker(_subsample(dataset, seed), n, **common)
    if rule == "noise_augmentation":
        return pickers.noise_augmentation_chain_start_picker(dataset, n, **common)

    adaptive = dict(common, candidate_multiplier=candidate_multiplier, beta=beta,
                    candidate_fn=candidate_fn)
    if rule == "uncertainty_guided":
        return pickers.uncertainty_guided_chain_start_picker(
            dataset, n, make_uncertainty_fn(critic), **adaptive)
    if rule == "violation_guided":
        return pickers.violation_guided_chain_start_picker(
            dataset, n, make_violation_fn(critic, training_h), **adaptive)
    if rule == "combined":
        return pickers.combined_chain_start_picker(
            dataset, n, make_uncertainty_fn(critic), make_violation_fn(critic, training_h),
            **adaptive)
    raise ValueError(f"No picker for rule {rule!r}")


# ----------------------------------------------------------------------------
# A round
# ----------------------------------------------------------------------------
@dataclass
class RoundOutput:
    transitions: Optional[SyntheticTransitions]
    anchor_states: Optional[torch.Tensor]
    anchor_h: Optional[torch.Tensor]
    stats: Dict[str, float] = field(default_factory=dict)


def _round_seed(cfg: ExperimentConfig, round_idx: int, attempt: int, salt: int = 0) -> int:
    return (cfg.seed * 1_000_003 + round_idx * 1009 + attempt * 31 + salt) % (2 ** 31 - 1)


def generate_round(
    cfg: ExperimentConfig,
    round_idx: int,
    spec: EnvSpec,
    dataset: OfflineDataset,
    critic,
    model,
    proposer: ActionProposer,
    training_h: HFn,
    valid_fn,
    candidate_fn: Optional[CandidateFn],
    max_attempts: int = 10,
) -> RoundOutput:
    device = dataset.device
    target = cfg.per_round(cfg.total_transitions, round_idx)
    threshold = cfg.max_disagreement * float(getattr(model, "disagreement_scale", 1.0))
    pick = dict(critic=critic, training_h=training_h, valid_fn=valid_fn,
                candidate_fn=candidate_fn, candidate_multiplier=cfg.candidate_multiplier,
                beta=cfg.beta)

    accepted, n_proposed, n_accepted, chain_offset = [], 0, 0, 0
    for attempt in range(max_attempts if target > 0 else 0):
        remaining = target - n_accepted
        if remaining <= 0:
            break
        n_starts = max(16, math.ceil(remaining / cfg.horizon * 1.5))
        seed = _round_seed(cfg, round_idx, attempt)
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        starts = pick_chain_starts(cfg.rule, dataset, n_starts, seed=seed, **pick)
        raw = rollout_chains(starts, model, proposer, cfg.horizon, g)
        trans = label_transitions(raw, training_h, spec.is_terminal_fn)
        keep = filter_transitions(trans, valid_fn, threshold)
        keep = truncate_chains(keep, trans["chain_id"], trans["step"])
        n_proposed += len(trans["costs"])
        trans = select(trans, keep)
        trans["chain_id"] = trans["chain_id"] + chain_offset
        chain_offset += n_starts
        accepted.append(trans)
        n_accepted += len(trans["costs"])

    transitions = None
    if accepted:
        transitions = {k: torch.cat([t[k] for t in accepted]) for k in accepted[0]}
        transitions = {k: v[:target] for k, v in transitions.items()}
        if len(transitions["costs"]) < target:
            print(f"  [warn] round {round_idx}: only {len(transitions['costs'])}/{target} "
                  f"transitions accepted after {max_attempts} attempts")

    anchor_states = anchor_h = None
    n_anchor = cfg.per_round(cfg.total_anchors, round_idx)
    if n_anchor > 0:
        anchor_states = pick_chain_starts(cfg.rule, dataset, n_anchor,
                                          seed=_round_seed(cfg, round_idx, 0, salt=7), **pick)
        with torch.no_grad():
            anchor_h = training_h(anchor_states).reshape(-1)

    stats = {
        "target_transitions": float(target),
        "accepted_transitions": float(0 if transitions is None else len(transitions["costs"])),
        "acceptance_rate": float(n_accepted / n_proposed) if n_proposed else float("nan"),
        "n_anchors": float(0 if anchor_states is None else len(anchor_states)),
        "disagreement_threshold": threshold,
    }
    return RoundOutput(transitions, anchor_states, anchor_h, stats)


@torch.no_grad()
def synthetic_stats(out: RoundOutput, spec: EnvSpec, real: OfflineDataset) -> Dict[str, float]:
    stats = dict(out.stats)
    ref = real.data["observations"]
    ref = ref[torch.randperm(len(ref), device=ref.device)[:5000]]
    mean, std = ref.mean(0), ref.std(0).clamp_min(1e-6)

    def describe(prefix: str, states: Optional[torch.Tensor]) -> None:
        if states is None or len(states) == 0:
            return
        s = states[:5000]
        stats[prefix + "unsafe_frac"] = float((spec.h_fn(s).reshape(-1) > 0).float().mean())
        d = torch.cdist((s - mean) / std, (ref - mean) / std).min(dim=1).values
        stats[prefix + "nn_dist_to_real"] = float(d.mean())
        if spec.has_ground_truth:
            from controlled.env_api import _ground_truth

            gt = _ground_truth()
            arr = s.double().cpu().numpy()
            stats[prefix + "in_true_band"] = float(gt.in_band(arr).mean())
            stats[prefix + "true_feasible_frac"] = float(gt.feasible(arr).mean())

    if out.transitions is not None:
        stats["mean_disagreement"] = float(out.transitions["disagreement"].mean())
        describe("trans_", out.transitions["observations"])
    describe("anchor_", out.anchor_states)
    return stats
