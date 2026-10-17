from typing import Callable

import numpy as np
import torch

from controlled.env_api import get_offline_observations
from cost_area import h_car_circle, h_controlled_env, h_point_goal
from data.dsrl_ids import OFFLINE_CAR_CIRCLE, OFFLINE_GOAL_POINT
from data.utils import load_dsrl_dict


def check_h_validity(
    obs: np.ndarray,
    cost: np.ndarray,
    h_func: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    cost_tensor = torch.tensor(cost, dtype=torch.float32)

    pred_h = h_func(obs_tensor)

    valid_gt = cost_tensor <= 0
    valid_pred = pred_h <= 0

    print(obs_tensor[torch.where(valid_gt != valid_pred)].size())

    assert torch.equal(valid_gt, valid_pred)


def check_all_h() -> None:
    data = get_offline_observations()
    check_h_validity(data["next_observations"], data["costs"], h_controlled_env)
    print("h check passed on controlled_env")

    data = load_dsrl_dict(OFFLINE_GOAL_POINT)
    check_h_validity(data["next_observations"], data["costs"], h_point_goal)
    print("h check passed on offline_goal_point")

    data = load_dsrl_dict(OFFLINE_CAR_CIRCLE)
    check_h_validity(data["next_observations"], data["costs"], h_car_circle)
    print("h check passed on offline_car_circle")
