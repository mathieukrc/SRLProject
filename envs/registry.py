"""
Everything environment-specific lives behind EnvSpec, so the generator, the
training code and the runner never branch on the environment name.
Observation / action sizes are taken from the loaded dataset, not from here.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

from data.synthetic import ValidFunction
from data.utils import HFn, shape_h
from data.variant import BoundaryCriterion

# (obs, next_obs) -> bool (B,): the SAME termination rule as the environment that
# produced the data.
TerminalFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class EnvSpec:
    name: str
    h_fn: HFn  # raw signed constraint from cost_area.py, h > 0 unsafe
    is_terminal_fn: TerminalFn
    load_raw_dataset: Callable[[], Dict[str, np.ndarray]]
    boundary_criterion: BoundaryCriterion
    timeout_is_done: bool
    has_ground_truth: bool
    has_simulator: bool
    candidate_source: str  # "box" | "dataset"
    dsrl_id: Optional[str] = None


def _numpy_h(h_fn: HFn) -> Callable[[np.ndarray], np.ndarray]:
    def h(states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(states), dtype=torch.float32)
            return h_fn(s).numpy()

    return h


# ----------------------------------------------------------------------------
# termination rules
# ----------------------------------------------------------------------------
def controlled_terminal_fn(obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    from controlled.controlled_env import CFG

    pos = next_obs[..., :2]
    out = pos.abs().max(dim=-1).values > CFG["arena"]
    goal = torch.as_tensor(CFG["goal"], dtype=pos.dtype, device=pos.device)
    reached = torch.linalg.norm(pos - goal, dim=-1) < CFG["goal_radius"]
    return out | reached


def dsrl_terminal_fn(obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """DSRL safety-gymnasium tasks only end on timeout."""
    return torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)


# ----------------------------------------------------------------------------
# loaders
# ----------------------------------------------------------------------------
def _load_controlled() -> Dict[str, np.ndarray]:
    from controlled.env_api import get_offline_observations

    return get_offline_observations()


def _load_dsrl(dsrl_id: str) -> Callable[[], Dict[str, np.ndarray]]:
    def load() -> Dict[str, np.ndarray]:
        from data.utils import load_dsrl_dict

        d = load_dsrl_dict(dsrl_id)
        keys = ("observations", "next_observations", "actions", "rewards",
                "costs", "terminals", "timeouts")
        return {k: np.asarray(d[k], dtype=np.float32) for k in keys}

    return load


def get_env_spec(name: str) -> EnvSpec:
    from cost_area import h_car_circle, h_controlled_env, h_point_goal
    from data.dsrl_ids import OFFLINE_CAR_CIRCLE, OFFLINE_GOAL_POINT

    if name == "controlled":
        return EnvSpec(
            name=name,
            h_fn=h_controlled_env,
            is_terminal_fn=controlled_terminal_fn,
            load_raw_dataset=_load_controlled,
            boundary_criterion=BoundaryCriterion(
                h=_numpy_h(h_controlled_env), precursor_steps=5, radius=0.1, dims=(0, 1)
            ),
            timeout_is_done=False,
            has_ground_truth=True,
            has_simulator=True,
            candidate_source="box",
        )
    if name == "car_circle":
        return EnvSpec(
            name=name,
            h_fn=h_car_circle,
            is_terminal_fn=dsrl_terminal_fn,
            load_raw_dataset=_load_dsrl(OFFLINE_CAR_CIRCLE),
            boundary_criterion=BoundaryCriterion(h=_numpy_h(h_car_circle), precursor_steps=10),
            timeout_is_done=True,
            has_ground_truth=False,
            has_simulator=False,
            candidate_source="dataset",
            dsrl_id=OFFLINE_CAR_CIRCLE,
        )
    if name == "point_goal":
        return EnvSpec(
            name=name,
            h_fn=h_point_goal,
            is_terminal_fn=dsrl_terminal_fn,
            load_raw_dataset=_load_dsrl(OFFLINE_GOAL_POINT),
            boundary_criterion=BoundaryCriterion(h=_numpy_h(h_point_goal), precursor_steps=10),
            timeout_is_done=True,
            has_ground_truth=False,
            has_simulator=False,
            candidate_source="dataset",
            dsrl_id=OFFLINE_GOAL_POINT,
        )
    raise ValueError(f"Unknown env {name!r}")


# ----------------------------------------------------------------------------
# h used for training, and validity of synthetic states
# ----------------------------------------------------------------------------
def make_training_h(spec: EnvSpec, dense: bool, pos_scale: float, neg_scale: float) -> HFn:
    """The ONE h used to label real transitions, synthetic transitions and anchors."""
    if dense:
        return shape_h(spec.h_fn, pos_scale=pos_scale, neg_scale=neg_scale)

    def binary(s: torch.Tensor) -> torch.Tensor:
        h = spec.h_fn(s)
        return torch.where(h > 0, torch.full_like(h, pos_scale), torch.full_like(h, -1.0))

    return binary


def make_valid_fn(spec: EnvSpec, reference_states: np.ndarray) -> ValidFunction:
    """Physical plausibility of a synthetic state."""
    if spec.name == "controlled":
        from controlled.controlled_env import CFG

        arena, vmax = CFG["arena"], CFG["v_max"]

        def valid_controlled(s: torch.Tensor) -> torch.Tensor:
            return (
                (s[..., :2].abs() <= arena).all(dim=-1)
                & (s[..., 2:].abs() <= vmax + 1e-6).all(dim=-1)
                & torch.isfinite(s).all(dim=-1)
            )

        return valid_controlled

    ref = np.asarray(reference_states, dtype=np.float64)
    lo, hi = np.quantile(ref, 0.001, axis=0), np.quantile(ref, 0.999, axis=0)
    margin = 0.05 * (hi - lo)
    lo_t = torch.as_tensor(lo - margin, dtype=torch.float32)
    hi_t = torch.as_tensor(hi + margin, dtype=torch.float32)

    def valid_box(s: torch.Tensor) -> torch.Tensor:
        lo_d, hi_d = lo_t.to(s.device), hi_t.to(s.device)
        return ((s >= lo_d) & (s <= hi_d)).all(dim=-1) & torch.isfinite(s).all(dim=-1)

    return valid_box
