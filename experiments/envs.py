"""
Environment registry. Everything task-specific lives here, so run.py contains no
`if env == ...` branches. To add a task, add one EnvSpec to ENVS.

DSRL imports are done lazily inside the functions, so controlled-env runs do not need
dsrl / gymnasium / pybullet / mujoco to be importable.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

from cost_area import h_car_circle, h_controlled_env, h_point_goal
from data.variant import BoundaryCriterion

TensorFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class EnvSpec:
    name: str
    load_raw: Callable[[], Dict[str, np.ndarray]]
    h_fn: TensorFn
    make_valid_fn: Callable[[torch.Tensor], TensorFn]  # real observations -> valid_fn
    terminal_fn: TensorFn
    boundary: BoundaryCriterion
    true_step: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    has_ground_truth: bool = False
    make_eval_env: Optional[Callable[[int], object]] = None
    target_cost: Optional[float] = None
    uniform_frac: float = 0.5  # share of uniform candidates in the guided pickers


def _numpy_h(h_fn: TensorFn):
    def h(s):
        return h_fn(torch.as_tensor(np.asarray(s), dtype=torch.float32)).numpy()

    return h


def _never_terminal(s: torch.Tensor) -> torch.Tensor:
    return torch.zeros(s.shape[0], dtype=torch.bool, device=s.device)


def _bounds_valid_fn(real_obs: torch.Tensor, margin: float = 0.1, extra: Optional[TensorFn] = None) -> TensorFn:
    lo, hi = real_obs.min(dim=0).values, real_obs.max(dim=0).values
    pad = margin * (hi - lo)
    lo, hi = lo - pad, hi + pad

    def valid(s):
        ok = ((s >= lo) & (s <= hi)).all(dim=-1)
        return ok & extra(s) if extra is not None else ok

    return valid


# --------------------------------------------------------------------- controlled env
def _controlled_raw():
    from controlled.env_api import get_offline_observations

    return get_offline_observations()


def _controlled_step(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """torch port of controlled_env.dynamics (same equations)."""
    from controlled.controlled_env import CFG

    acc = a.clamp(-1.0, 1.0) * CFG["a_max"]
    p = s[:, :2] + s[:, 2:] * CFG["dt"]
    v = (s[:, 2:] + acc * CFG["dt"]).clamp(-CFG["v_max"], CFG["v_max"])
    return torch.cat((p, v), dim=-1)


def _controlled_terminal(s: torch.Tensor) -> torch.Tensor:
    from controlled.controlled_env import CFG

    return s[:, :2].abs().max(dim=-1).values > CFG["arena"]


def _controlled_valid(_real_obs: torch.Tensor) -> TensorFn:
    from controlled.controlled_env import CFG

    def valid(s):
        inside = s[:, :2].abs().max(dim=-1).values <= CFG["arena"]
        slow = s[:, 2:].abs().max(dim=-1).values <= CFG["v_max"] + 1e-6
        return inside & slow

    return valid


class _ControlledGymAdapter:
    """Gymnasium-style wrapper around controlled_env.ControlledEnv."""

    def __init__(self, seed: int):
        from controlled.controlled_env import ControlledEnv

        self.env = ControlledEnv(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        return self.env.reset().astype(np.float32), {}

    def step(self, a):
        obs, r, c, done, info = self.env.step(a)
        info = dict(info, cost=c)
        return obs.astype(np.float32), r, bool(done and not info["timeout"]), bool(info["timeout"]), info


# --------------------------------------------------------------------- DSRL
def _dsrl_raw(name: str):
    def load():
        from data.utils import load_dsrl_dict

        return load_dsrl_dict(name)

    return load


def _dsrl_env(name: str):
    def make(seed: int):
        import dsrl  # noqa: F401  (registers the Offline* envs)
        import gymnasium as gym

        env = gym.make(name)
        try:
            env.reset(seed=seed)
        except TypeError:
            pass
        return env

    return make


def _point_goal_valid(real_obs: torch.Tensor) -> TensorFn:
    # PointGoal1 observation: 12 sensor dims, then goal / hazard / vase lidar (16 each),
    # which must lie in [0, 1].
    def lidar_ok(s):
        lid = s[:, 12:60]
        return ((lid >= 0) & (lid <= 1)).all(dim=-1)

    return _bounds_valid_fn(real_obs, extra=lidar_ok)


ENVS: Dict[str, EnvSpec] = {
    "controlled": EnvSpec(
        name="controlled",
        load_raw=_controlled_raw,
        h_fn=h_controlled_env,
        make_valid_fn=_controlled_valid,
        terminal_fn=_controlled_terminal,
        boundary=BoundaryCriterion(
            h=_numpy_h(h_controlled_env), precursor_steps=10, radius=0.1, dims=(0, 1)
        ),
        true_step=_controlled_step,
        has_ground_truth=True,
        make_eval_env=lambda seed: _ControlledGymAdapter(seed),
        target_cost=None,
    ),
    "car_circle": EnvSpec(
        name="car_circle",
        load_raw=_dsrl_raw("OfflineCarCircle-v0"),
        h_fn=h_car_circle,
        make_valid_fn=lambda obs: _bounds_valid_fn(obs),
        terminal_fn=_never_terminal,
        boundary=BoundaryCriterion(h=_numpy_h(h_car_circle), precursor_steps=10),
        make_eval_env=_dsrl_env("OfflineCarCircle-v0"),
        target_cost=10.0,
        uniform_frac=0.0,
    ),
    "point_goal": EnvSpec(
        name="point_goal",
        load_raw=_dsrl_raw("OfflinePointGoal1-v0"),
        h_fn=h_point_goal,
        make_valid_fn=_point_goal_valid,
        terminal_fn=_never_terminal,
        boundary=BoundaryCriterion(h=_numpy_h(h_point_goal), precursor_steps=10),
        make_eval_env=_dsrl_env("OfflinePointGoal1-v0"),
        target_cost=10.0,
        uniform_frac=0.0,
    ),
}
