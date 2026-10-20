"""
Everything the generator needs to know about an environment, bundled in one object.

The generator never imports an environment directly. It only sees:
    h_fn        known cost structure, h(s) > 0 means violated (torch, batched)
    valid_fn    which states are physically plausible (torch -> bool)
    terminal_fn which next states end an episode (torch -> bool)
    step_model  true dynamics (controlled env) or a learned ensemble (DSRL)
This is what keeps the same generation code usable for every task.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import torch

from dynamics.ensemble import DynamicsEnsemble

TensorFn = Callable[[torch.Tensor], torch.Tensor]


class StepModel:
    """step(obs, act) -> (next_obs, disagreement or None). threshold: max disagreement."""

    threshold: float = float("inf")

    def step(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError


class TrueStep(StepModel):
    """Known dynamics, e.g. the controlled environment. No disagreement filter."""

    def __init__(self, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.fn = fn

    @torch.no_grad()
    def step(self, obs, act):
        return self.fn(obs, act), None


class LearnedStep(StepModel):
    """Mean prediction of a dynamics ensemble, with its calibrated threshold."""

    def __init__(self, model: DynamicsEnsemble, threshold_scale: float = 1.0):
        self.model = model
        self.threshold = float(model.threshold) * threshold_scale

    @torch.no_grad()
    def step(self, obs, act):
        return self.model.step(obs, act)


@dataclass
class GenContext:
    h_fn: TensorFn
    valid_fn: TensorFn
    terminal_fn: TensorFn
    step_model: StepModel
    act_low: torch.Tensor
    act_high: torch.Tensor
    data_obs: torch.Tensor  # real observations, for nearest-neighbour actions
    data_act: torch.Tensor
    device: torch.device
    generator: torch.Generator
    # lazily built nearest-neighbour index
    _tree: object = field(default=None, repr=False)
    _scale: Optional[np.ndarray] = field(default=None, repr=False)

    def uniform_actions(self, n: int) -> torch.Tensor:
        u = torch.rand(
            (n, self.act_low.shape[0]), device=self.device, generator=self.generator
        )
        return self.act_low + u * (self.act_high - self.act_low)

    def clip_actions(self, a: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(a, self.act_high), self.act_low)

    def nn_index(self, max_points: int = 200_000):
        if self._tree is None:
            from scipy.spatial import cKDTree

            obs = self.data_obs.detach().cpu().numpy()
            if len(obs) > max_points:  # keeps the index cheap on the DSRL datasets
                rng = np.random.default_rng(0)
                self._nn_rows = rng.choice(len(obs), max_points, replace=False)
                obs = obs[self._nn_rows]
            else:
                self._nn_rows = np.arange(len(obs))
            self._scale = obs.std(axis=0) + 1e-6
            self._tree = cKDTree(obs / self._scale)
        return self._tree, self._scale, self._nn_rows
