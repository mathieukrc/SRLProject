from typing import TypedDict

import torch


class Batch(TypedDict):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    costs: torch.Tensor
    masks: torch.Tensor
