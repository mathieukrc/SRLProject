import torch
import torch.nn as nn


def safe_expectile_loss(diff: torch.Tensor, expectile: float = 0.8) -> torch.Tensor:
    weight = torch.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff**2)


def expectile_loss(diff: torch.Tensor, expectile: float = 0.8) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * diff**2


def weighted_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    w = w.view(w.shape[0], *([1] * (x.dim() - 2)), w.shape[-1])
    return (x * w).sum(-1) / w.sum(-1).clamp_min(1e-8)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005) -> None:
    for t, s in zip(target.parameters(), source.parameters()):
        t.lerp_(s, tau)
