import torch

from controlled.controlled_env import CFG


def cost_controlled_env(x: torch.Tensor) -> torch.Tensor:
    obst = torch.tensor(CFG["obstacles"]).to(x.device)

    pos = x[:, :1]
    obst_pos = obst[:, :1]

    distances = torch.cdist(pos, obst_pos, p=2)
    radius = obst[:, 2]

    distances = distances - radius

    return distances[torch.min(distances, 1)]


def h_car_circle(s: torch.Tensor) -> torch.Tensor:
    """
    Analytical safety constraint for CarCircle.

    Args:
        s: Physical agent state with world x-coordinate at index 0, shape (B, act_dim)

    Returns:
        h(s) <= 0 if safe, h(s) > 0 if unsafe.
    """
    x = s[:, 0]
    return torch.abs(x) - 6.0


def h_point_goal(s: torch.Tensor) -> torch.Tensor:
    """
    Approximate PointGoal1 safety constraint from the observation.

    PointGoal1 hazards have radius 0.2.
    The pseudo-lidar maximum detection distance is 3.0.

    Returns:
        h(s) > 0: estimated position is inside a hazard.
        h(s) <= 0: estimated position is outside hazards.
    """
    if s.size(-1) != 60:
        raise ValueError(f"Expected 60 observation features, got {s.size}")

    # Observation layout: 12 agent features, followed by
    # 16 goal-lidar, 16 hazard-lidar, and 16 vase-lidar values.
    hazard_lidar = s[:, 28:44]

    # Convert the strongest pseudo-lidar reading into an
    # estimated distance to the nearest detected hazard.
    lb = torch.zeros((hazard_lidar.shape[0])).to(hazard_lidar.device)
    ub = torch.ones((hazard_lidar.shape[0])).to(hazard_lidar.device)

    nearest_distance = 3.0 * (
        1.0 - torch.clamp(hazard_lidar[hazard_lidar.max(dim=-1)], lb, ub)
    )

    return 0.2 - nearest_distance
