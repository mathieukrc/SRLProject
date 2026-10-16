from typing import TypedDict

import torch


class Batch(TypedDict):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    masks: torch.Tensor


class WeightedBatch(Batch):
    bootstrap_weights: torch.Tensor
