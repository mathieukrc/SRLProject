import os
import random
from typing import Callable, Dict, Optional

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
    import dsrl
    import gymnasium as gym

    print(sorted(k for k in gym.envs.registry if k.startswith("Offline")))


def load_dsrl_dict(name: str) -> Dict[str, np.ndarray]:
    import dsrl
    import gymnasium as gym

    env = gym.make(name)
    dataset = env.get_dataset()

    return dataset


def cost_to_h(costs: np.ndarray, cost_scale: float = 25.0) -> np.ndarray:
    return np.where(np.asarray(costs) > 0, cost_scale, -1.0).astype(np.float32)


HFn = Callable[[torch.Tensor], torch.Tensor]


def shape_h(
    h_fn: HFn,
    pos_scale: float = 25.0,
    neg_scale: float = 1.0,
    neg_clip: Optional[float] = None,
) -> HFn:
    def shaped(s: torch.Tensor) -> torch.Tensor:
        h = h_fn(s)
        if neg_clip is not None:
            h = torch.clamp(h, min=-neg_clip)
        return torch.where(h > 0, h * pos_scale, h * neg_scale)

    return shaped


def compute_masks(
    terminals: np.ndarray, timeouts: np.ndarray, timeout_is_done: bool = True
) -> np.ndarray:
    dones = (
        np.logical_or(terminals, timeouts).astype(np.float32)
        if timeout_is_done
        else terminals.astype(np.float32)
    )
    return 1.0 - dones


def normalise_rewards(
    rewards: np.ndarray,
    max_episode_reward: float,
    min_episode_reward: float,
    scaling: float,
) -> np.ndarray:
    reward_range = max_episode_reward - min_episode_reward
    if reward_range <= 0:
        raise ValueError("max_episode_reward must exceed min_episode reward")
    return rewards / reward_range * scaling


def load_all_dsrl() -> None:
    print("Loading datasets into memory (and downloading if missing)")
    list_envs()
    print("Loading " + OFFLINE_CAR_CIRCLE + "...")
    load_dsrl_dict(OFFLINE_CAR_CIRCLE)
    print("Loading " + OFFLINE_GOAL_POINT + "...")
    load_dsrl_dict(OFFLINE_GOAL_POINT)
