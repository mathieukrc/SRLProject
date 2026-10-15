import torch
import torch.nn as nn


def safe_expectile_loss(diff: torch.Tensor, expectile: float = 0.8) -> torch.Tensor:
    weight = torch.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff**2)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005) -> None:
    for t, s in zip(target.parameters(), source.parameters()):
        t.lerp_(s, tau)
