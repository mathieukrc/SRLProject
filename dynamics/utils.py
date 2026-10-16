from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.offline_dataset import OfflineDataset


def make_dynamics_xy(dataset: OfflineDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    obs = dataset.data["observations"].float()
    act = dataset.data["actions"].float()
    next_obs = dataset.data["next_observations"].float()
    return torch.cat((obs, act), dim=-1), next_obs - obs


def mean_loss(net: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            total += criterion(net(x), y).item() * len(x)
            n += len(x)
    return total / n
