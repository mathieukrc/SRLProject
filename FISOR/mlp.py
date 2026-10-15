from typing import Callable, Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    layers: nn.ModuleList
    head: nn.Module
    activation: nn.Module

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: Callable[[], nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims]
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])
        )
        self.head = nn.Linear(dims[-1], 1)
        self.activation = activation()

        for layer in [*self.layers, self.head]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.head(x).squeeze(-1)
