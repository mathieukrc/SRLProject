import torch
import torch.nn as nn

from architectures.mlp import MLP

from .normalizer import Normalizer


class DynamicsModel(nn.Module):
    net: MLP
    in_norm: Normalizer
    out_norm: Normalizer

    def __init__(self, net: MLP, in_norm: Normalizer, out_norm: Normalizer):
        super().__init__()
        self.net = net
        self.in_norm = in_norm
        self.out_norm = out_norm

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = self.in_norm.normalize(torch.cat((obs, act), dim=-1))
        delta = self.out_norm.denormalize(self.net(x))
        return obs + delta
