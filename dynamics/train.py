import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from architectures.mlp import MLP
from data.offline_dataset import OfflineDataset

from .model import DynamicsModel
from .normalizer import Normalizer
from .utils import make_dynamics_xy, mean_loss


def train_dynamics_model(
    train_dataset: OfflineDataset,
    validation_dataset: OfflineDataset,
    num_epochs: int,
    lr: float = 0.001,
    batch_size: int = 256,
    patience: Optional[int] = 20,
) -> DynamicsModel:
    device = train_dataset.device
    state_dim = train_dataset.data["observations"].size(-1)
    action_dim = train_dataset.data["actions"].size(-1)

    x_train, y_train = make_dynamics_xy(train_dataset)
    x_val, y_val = make_dynamics_xy(validation_dataset)

    in_norm = Normalizer(x_train).to(device)
    out_norm = Normalizer(y_train).to(device)

    train_loader = DataLoader(
        TensorDataset(in_norm.normalize(x_train), out_norm.normalize(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(in_norm.normalize(x_val), out_norm.normalize(y_val)),
        batch_size=4 * batch_size,
        shuffle=False,
    )

    net = MLP(state_dim + action_dim, out_dim=state_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        net.train()
        running, n = 0.0, 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()

            loss = criterion(net(inputs), targets)
            loss.backward()
            optimizer.step()

            running += loss.item() * len(inputs)
            n += len(inputs)
        train_loss = running / n

        net.eval()
        val_loss = mean_loss(net, val_loader, criterion)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(net.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Loss: {train_loss:.5f}, Validation loss: {val_loss:.5f} (best: {best_val:.5f})"
        )

        if patience is not None and epochs_without_improvement >= patience:
            break

    net.load_state_dict(best_state)
    return DynamicsModel(net, in_norm, out_norm).eval()
