from typing import Optional

import controlled.controlled_env as ce
import numpy as np
import torch

from .base import DynamicsBase


class TrueControlledDynamics(DynamicsBase):
    num_members = 1
    disagreement_threshold = float("inf")

    def __init__(self, cfg: dict = ce.CFG, device: Optional[str] = None) -> None:
        self.cfg = cfg
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @torch.no_grad()
    def predict_members(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        acc = act.clamp(-1.0, 1.0) * c["a_max"]
        p = obs[..., :2] + obs[..., 2:] * c["dt"]
        v = (obs[..., 2:] + acc * c["dt"]).clamp(-c["v_max"], c["v_max"])
        return torch.cat([p, v], dim=-1).unsqueeze(0)

    @torch.no_grad()
    def disagreement(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)


def controlled_terminal_fn(cfg: dict = ce.CFG):
    goal = torch.tensor(cfg["goal"])

    def terminal(next_obs: torch.Tensor) -> torch.Tensor:
        pos = next_obs[..., :2]
        out = pos.abs().max(dim=-1).values > cfg["arena"]
        reached = (pos - goal.to(pos)).norm(dim=-1) < cfg["goal_radius"]
        return out | reached

    return terminal


def check_against_numpy(n: int = 10000, seed: int = 0, cfg: dict = ce.CFG) -> float:
    rng = np.random.default_rng(seed)
    s = np.concatenate(
        [rng.uniform(-1.3, 1.3, (n, 2)), rng.uniform(-1.2, 1.2, (n, 2))], axis=1
    )
    s[: n // 10, 2:] = np.sign(s[: n // 10, 2:]) * cfg["v_max"]  # at the clip
    a = rng.uniform(-1.5, 1.5, (n, 2))  # includes out-of-range actions
    ref = ce.dynamics(s, a, cfg)
    model = TrueControlledDynamics(cfg, device="cpu")
    out = model.predict_mean(
        torch.as_tensor(s, dtype=torch.float32), torch.as_tensor(a, dtype=torch.float32)
    ).numpy()
    return float(np.abs(out - ref).max())
