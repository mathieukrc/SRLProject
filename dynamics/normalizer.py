import torch
import torch.nn as nn


class Normalizer(nn.Module):
    def __init__(self, x: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", x.mean(dim=0))
        self.register_buffer("std", x.std(dim=0).clamp_min(eps))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean
