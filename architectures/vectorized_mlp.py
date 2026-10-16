import math
from typing import Callable, Sequence

import torch
import torch.nn as nn


class VectorizedLinear(nn.Module):
    weight: nn.Parameter
    bias: nn.Parameter

    def __init__(self, in_features: int, out_features: int, ensemble_size: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(
            torch.empty(ensemble_size, in_features, out_features)
        )
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = math.sqrt(6.0 / (self.in_features + self.out_features))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.ensemble_size, -1, -1)

        return torch.baddbmm(self.bias, x, self.weight)


class VectorizedMLP(nn.Module):
    layers: nn.ModuleList
    head: nn.Module
    activation: nn.Module

    def __init__(
        self,
        in_dim: int,
        ensemble_size: int,
        out_dim: int = 1,
        hidden_dims: Sequence[int] = (256, 256),
        activation: Callable[[], nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        dims = [in_dim, *hidden_dims]
        self.layers = nn.ModuleList(
            VectorizedLinear(a, b, ensemble_size) for a, b in zip(dims[:-1], dims[1:])
        )
        self.head = VectorizedLinear(dims[-1], out_dim, ensemble_size)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        for layer in self.layers:
            x = self.activation(layer(x))
        return self.head(x).squeeze(-1)
