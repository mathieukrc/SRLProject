import math
from typing import Optional

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    in_dim: int
    output_size: int
    learnable: bool
    kernel: Optional[torch.Tensor]

    def __init__(
        self, in_dim: int = 1, output_size: int = 64, learnable: bool = True
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.output_size = output_size
        self.learnable = learnable

        if self.learnable:
            self.kernel = nn.Parameter(
                torch.empty((self.output_size // 2, self.in_dim), dtype=torch.float32)
            )
            nn.init.normal_(self.kernel, mean=0.0, std=0.2)
        else:
            self.register_parameter("kernel", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.learnable:
            f = 2 * torch.pi * x @ self.kernel.T
        else:
            half_dim = self.output_size // 2
            scale = math.log(10000) / (half_dim - 1)
            freqs = torch.exp(
                torch.arange(half_dim, device=x.device, dtype=x.dtype) * -scale
            )
            f = x * freqs
        return torch.cat((torch.cos(f), torch.sin(f)), dim=-1)
