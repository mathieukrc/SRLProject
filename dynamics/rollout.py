from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from .base import DynamicsBase

ActionFn = Callable[[torch.Tensor, int], torch.Tensor]  # (obs (B, d), t) -> act (B, a)
TerminalFn = Callable[[torch.Tensor], torch.Tensor]  # next_obs (B, d) -> bool (B,)


@dataclass
class RolloutBatch:
    observations: torch.Tensor  # (M, obs_dim)
    actions: torch.Tensor  # (M, act_dim)
    next_observations: torch.Tensor  # (M, obs_dim)
    masks: torch.Tensor  # (M,) 0 = terminal
    disagreement: torch.Tensor  # (M,)
    start_index: torch.Tensor  # (M,) which start state the step came from
    step: torch.Tensor  # (M,) step index within its rollout

    def __len__(self) -> int:
        return self.observations.shape[0]

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return dict(self.__dict__)


@torch.no_grad()
def rollout(
    dynamics: DynamicsBase,
    start_obs: torch.Tensor,
    action_fn: ActionFn,
    horizon: int,
    threshold: Optional[float] = None,
    terminal_fn: Optional[TerminalFn] = None,
    mode: str = "random_member",
    generator: Optional[torch.Generator] = None,
    action_clip: float = 1.0 - 1e-5,  # same preprocessing as OfflineDataset
    sample_noise: bool = False,
) -> RolloutBatch:
    """
    threshold: defaults to dynamics.disagreement_threshold (inf for true dynamics).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    threshold = dynamics.disagreement_threshold if threshold is None else threshold

    B = start_obs.shape[0]
    device = start_obs.device
    obs = start_obs
    alive = torch.ones(B, dtype=torch.bool, device=device)
    starts = torch.arange(B, device=device)
    rows = {k: [] for k in ("obs", "act", "next", "mask", "dis", "start", "step")}

    kwargs = {"sample_noise": sample_noise} if sample_noise else {}

    for t in range(horizon):
        act = action_fn(obs, t).clamp(-action_clip, action_clip)
        nxt, dis = dynamics.transition(
            obs, act, mode=mode, generator=generator, **kwargs
        )

        trusted = alive & (dis <= threshold)
        terminal = (
            terminal_fn(nxt)
            if terminal_fn is not None
            else torch.zeros(B, dtype=torch.bool, device=device)
        )

        keep = trusted
        rows["obs"].append(obs[keep])
        rows["act"].append(act[keep])
        rows["next"].append(nxt[keep])
        rows["mask"].append((~terminal[keep]).float())
        rows["dis"].append(dis[keep])
        rows["start"].append(starts[keep])
        rows["step"].append(torch.full((int(keep.sum()),), t, device=device))

        alive = trusted & ~terminal
        if not bool(alive.any()):
            break
        obs = nxt

    cat = {k: torch.cat(v) for k, v in rows.items()}
    return RolloutBatch(
        observations=cat["obs"],
        actions=cat["act"],
        next_observations=cat["next"],
        masks=cat["mask"],
        disagreement=cat["dis"],
        start_index=cat["start"],
        step=cat["step"].long(),
    )
