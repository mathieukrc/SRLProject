import os
import random

import dsrl
import gymnasium as gym
import numpy as np
import torch

from .dsrl_ids import OFFLINE_CAR_CIRCLE, OFFLINE_GOAL_POINT


def set_seed(seed=42):
    # Python built-in random
    random.seed(seed)
    # OS environment for Python hash functions
    os.environ["PYTHONHASHSEED"] = str(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch GPU (single GPU)
    torch.cuda.manual_seed(seed)
    # PyTorch GPU (multi-GPU)
    torch.cuda.manual_seed_all(seed)
    # CuDNN deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_envs() -> None:
    print(sorted(k for k in gym.envs.registry if k.startswith("Offline")))


def load_dsrl_dict(name: str) -> None:
    env = gym.make(name)
    dataset = env.get_dataset()

    for k, v in dataset.items():
        print(k, v.shape)


def cost_to_h(costs: np.ndarray, cost_scale: float = 25.0) -> np.ndarray:
    return np.where(costs > 0, cost_scale, -1.0).astype(np.float32)


def compute_masks(terminals: np.ndarray, timeouts: np.ndarray) -> np.ndarray:
    dones = np.logical_or(terminals, timeouts).astype(np.float32)
    return 1.0 - dones


def load_all_dsrl() -> None:
    print("Loading datasets into memory (and downloading if missing)")
    list_envs()
    print("Loading " + OFFLINE_CAR_CIRCLE + "...")
    load_dsrl_dict(OFFLINE_CAR_CIRCLE)
    print("Loading " + OFFLINE_GOAL_POINT + "...")
    load_dsrl_dict(OFFLINE_GOAL_POINT)
