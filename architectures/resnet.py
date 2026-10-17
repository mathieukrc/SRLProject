from typing import Callable, Optional

import torch
import torch.nn as nn


def lecun_normal_(linear: nn.Linear) -> None:
    fan_in = linear.weight.shape[1]
    std = (1.0 / fan_in) ** 0.5 / 0.87962566103423978  # Flax's truncation correction
    nn.init.trunc_normal_(linear.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
    nn.init.zeros_(linear.bias)


class MLPResNetBlock(nn.Module):
    layers: nn.ModuleList

    def __init__(
        self,
        features: int,
        activation: Callable[[], nn.Module] = nn.Mish,
        dropout_rate: Optional[float] = None,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        if dropout_rate is not None and dropout_rate > 0:
            self.layers.append(nn.Dropout(dropout_rate))

        if use_layer_norm:
            self.layers.append(nn.LayerNorm((features), eps=1e-6))

        self.layers.append(nn.Linear(features, 4 * features))
        lecun_normal_(self.layers[-1])
        self.layers.append(activation())
        self.layers.append(nn.Linear(4 * features, features))
        lecun_normal_(self.layers[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for layer in self.layers:
            x = layer(x)

        return residual + x


class MLPResNet(nn.Module):
    layers: nn.ModuleList

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_blocks: int = 3,
        hidden_dims: int = 256,
        dropout_rate: Optional[float] = 0.1,
        use_layer_norm: bool = True,
        activation: Callable[[], nn.Module] = nn.Mish,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dims))
        nn.init.xavier_uniform_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

        for _ in range(num_blocks):
            self.layers.append(
                MLPResNetBlock(hidden_dims, activation, dropout_rate, use_layer_norm)
            )

        self.layers.append(activation())
        self.layers.append(nn.Linear(hidden_dims, out_dim))
        nn.init.xavier_uniform_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
