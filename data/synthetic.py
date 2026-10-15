from typing import Callable

import torch

from .offline_dataset import OfflineDataset

UncertaintyFunction = Callable[
    [torch.Tensor], torch.Tensor
]  # Takes in tensor of states and returns their uncertainty
ViolationFunction = Callable[
    [torch.Tensor], torch.Tensor
]  # Takes in tensor of states and returns whether the critic disagrees with h(s) on those states


def random_chain_start_picker(obs: OfflineDataset, budget: int) -> torch.Tensor:
    """
    Args:
        obs: Offline dataset
        budget: amount of states to generate
    Returns:
        A tensor of randomly generated states from the range of obs
    """
    raise NotImplementedError()


START_TARGETING_RULES = {
    "random",
    "density_matched",
    "noise_augmentation",
    "uncertainty_guided",
    "violation_guided",
    "combined",
}
