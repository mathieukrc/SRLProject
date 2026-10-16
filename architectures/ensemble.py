from typing import Callable

import torch
import torch.nn as nn


class Ensemble(nn.Module):
    members: nn.ModuleList

    def __init__(self, make_model: Callable[[], nn.Module], num: int = 2) -> None:
        super().__init__()

        self.members = nn.ModuleList(make_model() for _ in range(num))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.members], dim=0)
