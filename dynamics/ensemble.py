import os
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.vectorized_mlp import VectorizedMLP

from .base import MODES, DynamicsBase, pick_members
from .normalizer import Normalizer

FORMAT_VERSION = 1


class EnsembleDynamics(nn.Module, DynamicsBase):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_members: int = 5,
        hidden_dims: Sequence[int] = (200, 200, 200),
        probabilistic: bool = False,
        in_norm: Optional[Normalizer] = None,
        out_norm: Optional[Normalizer] = None,
        obs_low: Optional[torch.Tensor] = None,
        obs_high: Optional[torch.Tensor] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.num_members = num_members
        self.hidden_dims = tuple(hidden_dims)
        self.probabilistic = probabilistic

        out_dim = obs_dim * (2 if probabilistic else 1)
        self.net = VectorizedMLP(
            obs_dim + act_dim,
            ensemble_size=num_members,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            activation=nn.SiLU,
        )
        if probabilistic:
            self.max_logvar = nn.Parameter(torch.full((num_members, 1, obs_dim), 0.5))
            self.min_logvar = nn.Parameter(torch.full((num_members, 1, obs_dim), -10.0))

        self.in_norm = in_norm or Normalizer(torch.zeros(2, obs_dim + act_dim))
        self.out_norm = out_norm or Normalizer(torch.zeros(2, obs_dim))

        inf = torch.full((obs_dim,), float("inf"))
        self.register_buffer("obs_low", -inf if obs_low is None else obs_low.float())
        self.register_buffer("obs_high", inf if obs_high is None else obs_high.float())
        self.register_buffer("_threshold", torch.tensor(float("inf")))

        self.metadata: Dict[str, Any] = {}

    @property
    def disagreement_threshold(self) -> float:
        return float(self._threshold)

    @disagreement_threshold.setter
    def disagreement_threshold(self, value: float) -> None:
        self._threshold.fill_(float(value))

    def forward_normalized(
        self, x_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = self.net(x_norm)
        if not self.probabilistic:
            return out, None
        mean, logvar = out.split(self.obs_dim, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def _forward(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.in_norm.normalize(torch.cat((obs, act), dim=-1))
        return self.forward_normalized(x)

    def _to_state(self, obs: torch.Tensor, delta_norm: torch.Tensor) -> torch.Tensor:
        nxt = obs.unsqueeze(0) + self.out_norm.denormalize(delta_norm)
        return torch.maximum(torch.minimum(nxt, self.obs_high), self.obs_low)

    @staticmethod
    def _spread(mean_norm: torch.Tensor) -> torch.Tensor:
        """Std across members per dim (normalised units), averaged over dims."""
        var = ((mean_norm - mean_norm.mean(dim=0, keepdim=True)) ** 2).mean(dim=0)
        return var.sqrt().mean(dim=-1)

    @torch.no_grad()
    def predict_members(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        mean, _ = self._forward(obs, act)
        return self._to_state(obs, mean)

    @torch.no_grad()
    def disagreement(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        mean, _ = self._forward(obs, act)
        return self._spread(mean)

    @torch.no_grad()
    def transition(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        mode: str = "random_member",
        generator: Optional[torch.Generator] = None,
        sample_noise: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        mean, logvar = self._forward(obs, act)
        dis = self._spread(mean)
        delta = mean
        if sample_noise and self.probabilistic:
            eps = torch.randn(mean.shape, device=mean.device, generator=generator)
            delta = mean + (0.5 * logvar).exp() * eps
        return pick_members(self._to_state(obs, delta), mode, generator), dis

    def constructor_kwargs(self) -> Dict[str, Any]:
        return dict(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            num_members=self.num_members,
            hidden_dims=list(self.hidden_dims),
            probabilistic=self.probabilistic,
        )

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        torch.save(
            {
                "format_version": FORMAT_VERSION,
                "kwargs": self.constructor_kwargs(),
                "state_dict": self.state_dict(),
                "metadata": self.metadata,
            },
            tmp,
        )
        os.replace(tmp, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
        expect: Optional[Dict[str, Any]] = None,
    ) -> "EnsembleDynamics":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported format {ckpt.get('format_version')}")
        meta = ckpt.get("metadata", {})
        for k, v in (expect or {}).items():
            if meta.get(k) != v:
                raise ValueError(
                    f"{path}: metadata {k}={meta.get(k)!r}, expected {v!r}"
                )
        model = cls(**ckpt["kwargs"])
        model.load_state_dict(ckpt["state_dict"])
        model.metadata = meta
        return model.to(device).eval()
