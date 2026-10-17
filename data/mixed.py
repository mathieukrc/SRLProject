"""
Real + synthetic sampling with a fixed synthetic ratio, plus anchor batches.
MixedSampler is used everywhere (round 0 = empty buffer), so rule "none" runs the
same code path with no synthetic data.
"""

from typing import Dict, Optional, Tuple

import torch

from data.offline_dataset import OfflineDataset

TRANSITION_KEYS = ("observations", "actions", "next_observations", "costs", "rewards", "masks")


def _poisson(n_members: int, n: int, device, generator=None) -> torch.Tensor:
    if n_members == 1:
        return torch.ones((1, n), device=device)
    return torch.poisson(torch.ones((n_members, n), device=device), generator=generator)


class SyntheticBuffer:
    """Grows across rounds; every added row gets Poisson bootstrap weights."""

    def __init__(self, obs_dim: int, act_dim: int, n_members: int, device) -> None:
        self.obs_dim, self.act_dim, self.n_members = obs_dim, act_dim, n_members
        self.device = torch.device(device)
        self.trans: Dict[str, torch.Tensor] = {
            "observations": torch.empty((0, obs_dim), device=self.device),
            "actions": torch.empty((0, act_dim), device=self.device),
            "next_observations": torch.empty((0, obs_dim), device=self.device),
            "costs": torch.empty((0,), device=self.device),
            "rewards": torch.empty((0,), device=self.device),
            "masks": torch.empty((0,), device=self.device),
            "bootstrap_weights": torch.empty((n_members, 0), device=self.device),
        }
        self.anchors: Dict[str, torch.Tensor] = {
            "observations": torch.empty((0, obs_dim), device=self.device),
            "h": torch.empty((0,), device=self.device),
            "bootstrap_weights": torch.empty((n_members, 0), device=self.device),
        }

    def add_transitions(self, trans: Dict[str, torch.Tensor], generator=None) -> None:
        n = len(trans["costs"])
        if n == 0:
            return
        for k in TRANSITION_KEYS:
            self.trans[k] = torch.cat([self.trans[k], trans[k].to(self.device).float()])
        self.trans["bootstrap_weights"] = torch.cat(
            [self.trans["bootstrap_weights"], _poisson(self.n_members, n, self.device, generator)],
            dim=1,
        )

    def add_anchors(self, states: torch.Tensor, h: torch.Tensor, generator=None) -> None:
        n = len(states)
        if n == 0:
            return
        self.anchors["observations"] = torch.cat([self.anchors["observations"], states.to(self.device).float()])
        self.anchors["h"] = torch.cat([self.anchors["h"], h.to(self.device).float().reshape(-1)])
        self.anchors["bootstrap_weights"] = torch.cat(
            [self.anchors["bootstrap_weights"], _poisson(self.n_members, n, self.device, generator)],
            dim=1,
        )

    @property
    def n_transitions(self) -> int:
        return int(self.trans["costs"].shape[0])

    @property
    def n_anchors(self) -> int:
        return int(self.anchors["h"].shape[0])

    def sample_transitions(self, n: int, generator=None) -> Dict[str, torch.Tensor]:
        idx = torch.randint(self.n_transitions, (n,), device=self.device, generator=generator)
        return {k: (v[:, idx] if k == "bootstrap_weights" else v[idx]) for k, v in self.trans.items()}

    def sample_anchors(self, n: int, generator=None) -> Dict[str, torch.Tensor]:
        idx = torch.randint(self.n_anchors, (n,), device=self.device, generator=generator)
        return {k: (v[:, idx] if k == "bootstrap_weights" else v[idx]) for k, v in self.anchors.items()}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {**{"t/" + k: v.cpu() for k, v in self.trans.items()},
                **{"a/" + k: v.cpu() for k, v in self.anchors.items()}}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            group, key = k.split("/", 1)
            (self.trans if group == "t" else self.anchors)[key] = v.to(self.device)


class MixedSampler:
    def __init__(
        self,
        real: OfflineDataset,
        synthetic: SyntheticBuffer,
        synthetic_ratio: float,
        anchor_batch_size: int = 256,
        seed: Optional[int] = None,
    ) -> None:
        self.real, self.synthetic = real, synthetic
        self.ratio, self.anchor_batch_size = synthetic_ratio, anchor_batch_size
        self.generator = torch.Generator(device=real.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        n_syn = int(round(batch_size * self.ratio)) if self.synthetic.n_transitions > 0 else 0
        real = dict(self.real.sample(batch_size - n_syn))
        real["bootstrap_weights"] = real["bootstrap_weights"].expand(
            self.synthetic.n_members, -1
        ) if real["bootstrap_weights"].shape[0] == 1 else real["bootstrap_weights"]
        keys = TRANSITION_KEYS + ("bootstrap_weights",)
        if n_syn > 0:
            syn = self.synthetic.sample_transitions(n_syn, self.generator)
            batch = {k: torch.cat([real[k], syn[k]], dim=1 if k == "bootstrap_weights" else 0)
                     for k in keys}
            is_syn = torch.cat([torch.zeros(batch_size - n_syn, dtype=torch.bool, device=self.real.device),
                                torch.ones(n_syn, dtype=torch.bool, device=self.real.device)])
        else:
            batch = {k: real[k] for k in keys}
            is_syn = torch.zeros(batch_size, dtype=torch.bool, device=self.real.device)
        batch["is_synthetic"] = is_syn

        # Shuffle: the critics take the first critic_batch_size rows.
        perm = torch.randperm(batch_size, device=self.real.device, generator=self.generator)
        batch = {k: (v[:, perm] if k == "bootstrap_weights" else v[perm]) for k, v in batch.items()}

        anchors = None
        if self.synthetic.n_anchors > 0:
            anchors = self.synthetic.sample_anchors(self.anchor_batch_size, self.generator)
        return batch, anchors
