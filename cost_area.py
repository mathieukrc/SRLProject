import torch

from controlled.controlled_env import CFG


def h_controlled_env(s: torch.Tensor) -> torch.Tensor:
    x, y = s[..., 0], s[..., 1]

    h = torch.maximum(x.abs(), y.abs()) - CFG["arena"]

    obst = torch.as_tensor(CFG["obstacles"], dtype=s.dtype, device=s.device)
    dx = x.unsqueeze(-1) - obst[:, 0]
    dy = y.unsqueeze(-1) - obst[:, 1]
    dist = torch.sqrt(dx**2 + dy**2 + 1e-12)
    h_obst = (obst[:, 2] - dist).max(dim=-1).values

    return torch.maximum(h, h_obst)


def h_car_circle(s: torch.Tensor) -> torch.Tensor:
    """
    Analytical safety constraint for CarCircle.

    Args:
        s: Physical agent state with world x-coordinate at index 0, shape (B, act_dim)

    Returns:
        h(s) <= 0 if safe, h(s) > 0 if unsafe.
    """
    if s.shape[-1] != 8:
        raise ValueError(f"Expected 8 observation features, got {s.shape[-1]}")
    x_world = s[..., 0] / 0.1
    return x_world.abs() - 6.0


def h_point_goal(s: torch.Tensor) -> torch.Tensor:
    """
    Approximate PointGoal1 safety constraint from the observation.

    PointGoal1 hazards have radius 0.2.
    The pseudo-lidar maximum detection distance is 3.0.

    Returns:
        h(s) > 0: estimated position is inside a hazard.
        h(s) <= 0: estimated position is outside hazards.
    """
    hazard_lidar = slice(28, 44)
    lidar_max_dist = 3.0
    hazard_size = 0.2

    if s.shape[-1] != 60:
        raise ValueError(f"Expected 60 observation features, got {s.shape[-1]}")

    lidar = s[..., hazard_lidar].clamp(0.0, 1.0)
    nearest = lidar_max_dist * (1.0 - lidar.max(dim=-1).values)

    return hazard_size - nearest
