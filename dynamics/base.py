from typing import Optional, Tuple

import torch

MODES = ("random_member", "mean")


class DynamicsBase:
    num_members: int = 1
    disagreement_threshold: float = float("inf")

    def predict_members(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def disagreement(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def predict_mean(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.predict_members(obs, act).mean(dim=0)

    @torch.no_grad()
    def transition(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        mode: str = "random_member",
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        members = self.predict_members(obs, act)
        dis = self.disagreement(obs, act)
        return pick_members(members, mode, generator), dis


def pick_members(
    members: torch.Tensor, mode: str, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    if mode == "mean":
        return members.mean(dim=0)
    K, B = members.shape[:2]
    idx = torch.randint(K, (B,), device=members.device, generator=generator)
    return members[idx, torch.arange(B, device=members.device)]
