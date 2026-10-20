"""
generate(): turn a targeting rule and a transition budget into synthetic transitions.

    T  targeting rule       data/synthetic.py chain-start pickers (unchanged interface)
    L  labeller             exact h from the known cost function (done by OfflineDataset)
    F  filter               validity + dynamics disagreement (generation/rollout.py)

The budget is counted in transitions *after* filtering, and generation repeats until the
budget is met. Without this, rules whose starts get filtered more often would silently
receive less data and the comparison between rules would be unfair.
"""

import math
import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from data import synthetic as T

from .actions import propose_first_actions
from .context import GenContext
from .rollout import concat_raw, empty_raw, rollout_chains, take_raw

RULES = ("none",) + tuple(sorted(T.START_TARGETING_RULES))


class _StatesView:
    """Minimal stand-in for OfflineDataset, as the pickers only read observations."""

    def __init__(self, states: torch.Tensor):
        self.data = {"observations": states}


def scoring_functions(critic, h_fn, uncertainty: str = "entropy"):
    """uncertainty_fn and violation_fn for the guided pickers.

    violation = relu(h(s) - V_h(s)). The bound V_h(s) >= h(s) holds for the true value,
    so a positive score is a verifiable error. Anchors are not used as a loss; this is
    only a targeting signal.
    """

    if uncertainty == "entropy":
        unc = critic.feasibility_entropy
    elif uncertainty == "std":
        unc = lambda s: critic.v_h_members(s).std(dim=0)  # noqa: E731
    else:
        raise ValueError(f"unknown uncertainty measure {uncertainty!r}")

    def viol(s):
        return torch.relu(h_fn(s) - critic.v_h(s))

    return unc, viol


def pick_starts(
    rule: str,
    real_states: torch.Tensor,
    n: int,
    critic,
    ctx: GenContext,
    seed: int,
    uncertainty: str = "entropy",
    candidate_multiplier: int = 10,
    beta: Optional[float] = 1.0,
    uniform_frac: float = 0.5,
    noise_scale: float = 0.05,
    max_fit_points: int = 50_000,
) -> torch.Tensor:
    if rule not in RULES or rule == "none":
        raise ValueError(f"cannot pick starts for rule {rule!r}")

    # GMM fitting and bounds on millions of DSRL states are slow; a subsample suffices.
    if len(real_states) > max_fit_points:
        g = torch.Generator(device=real_states.device)
        g.manual_seed(seed)
        idx = torch.randperm(len(real_states), device=real_states.device, generator=g)
        real_states = real_states[idx[:max_fit_points]]
    view = _StatesView(real_states)
    valid = lambda s: ctx.valid_fn(s).reshape(-1).bool()  # noqa: E731

    if rule == "random":
        return T.random_chain_start_picker(view, n, seed=seed, valid_fn=valid)
    if rule == "density_matched":
        return T.density_matched_chain_start_picker(view, n, seed=seed, valid_fn=valid)
    if rule == "noise_augmentation":
        return T.noise_augmentation_chain_start_picker(
            view, n, noise_scale=noise_scale, seed=seed, valid_fn=valid
        )

    unc, viol = scoring_functions(critic, ctx.h_fn, uncertainty)
    common = dict(
        candidate_multiplier=candidate_multiplier, seed=seed, valid_fn=valid,
        beta=beta, uniform_frac=uniform_frac,
    )
    if rule == "uncertainty_guided":
        return T.uncertainty_guided_chain_start_picker(view, n, unc, **common)
    if rule == "violation_guided":
        return T.violation_guided_chain_start_picker(view, n, viol, **common)
    if rule == "combined":
        return T.combined_chain_start_picker(view, n, unc, viol, **common)
    raise AssertionError(rule)


def generate(
    rule: str,
    real_states: torch.Tensor,
    budget: int,
    critic,
    ctx: GenContext,
    seed: int,
    branching: int = 4,
    chain_len: int = 5,
    n_candidates: int = 32,
    horizon: int = 5,
    nn_noise: float = 0.1,
    policy_noise: float = 0.1,
    max_iters: int = 20,
    log_fn: Optional[Callable[[str], None]] = None,
    **pick_kwargs,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Returns (raw transitions with exactly `budget` rows when possible, stats)."""
    obs_dim, act_dim = real_states.shape[1], ctx.act_low.shape[0]
    if rule == "none" or budget <= 0:
        return empty_raw(obs_dim, act_dim), {"n_generated": 0, "n_starts": 0}

    parts, total, n_starts, it = [], 0, 0, 0
    per_start = branching * chain_len
    while total < budget and it < max_iters:
        remaining = budget - total
        # assume ~70% of a chain survives the filter; over-generating is trimmed below
        n = max(16, int(math.ceil(remaining / (per_start * 0.7))))
        starts = pick_starts(rule, real_states, n, critic, ctx, seed=seed * 1000 + it, **pick_kwargs)
        first = propose_first_actions(starts, ctx, branching, n_candidates, horizon, nn_noise)
        raw = rollout_chains(starts, first, ctx, chain_len, policy_noise, n_candidates, horizon)
        parts.append(raw)
        total += len(raw["observations"])
        n_starts += n
        it += 1

    if total < budget:
        warnings.warn(
            f"generate({rule}): only {total}/{budget} transitions survived the filter "
            f"after {max_iters} iterations"
        )
    raw = take_raw(concat_raw(parts), budget) if total else empty_raw(obs_dim, act_dim)

    with torch.no_grad():
        h_next = ctx.h_fn(torch.as_tensor(raw["next_observations"], device=ctx.device))
    stats = {
        "n_generated": int(len(raw["observations"])),
        "n_starts": int(n_starts),
        "iterations": int(it),
        "keep_rate": float(total / max(n_starts * per_start, 1)),
        "frac_terminal": float(raw["terminals"].mean()) if len(raw["terminals"]) else 0.0,
        "frac_next_unsafe": float((h_next > 0).float().mean()) if len(h_next) else 0.0,
    }
    if log_fn is not None:
        log_fn(f"  generated {stats['n_generated']} ({rule}), keep_rate={stats['keep_rate']:.2f}")
    return raw, stats
