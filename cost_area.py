from typing import Callable, Dict

import numpy as np
import scipy
from sklearn.cluster import DBSCAN


def cost_controlled_env(data: Dict[str, np.ndarray]) -> Callable[[float, float], int]:
    """
    Args:
        data: offline observations from the controlled environment
    Returns:
        A function that yields the cost in {0, 1} for coordinates (x, y)
    """
    ARENA = 1.0

    valid = []
    in_obstacle = []
    oob = []

    for o, c in zip(data["next_observations"], data["costs"]):
        if abs(o[0]) <= ARENA and abs(o[1]) <= ARENA:
            if c <= 0:
                valid.append(o)
            else:
                in_obstacle.append(o)
        else:
            oob.append(o)

    valid = np.array(valid)
    in_obstacle = np.array(in_obstacle)
    oob = np.array(oob)

    clustering = DBSCAN(eps=ARENA / 50, min_samples=1).fit(in_obstacle[:, :2])
    labels = np.array(clustering.labels_)
    clusters = [in_obstacle[labels == l] for l in np.unique(labels)]

    centers = [(np.mean(c[:, 0]), np.mean(c[:, 1])) for c in clusters]
    radius = [np.max(abs(c[:, :2] - cent)) for c, cent in zip(clusters, centers)]

    def loss(params, P_in, P_out, k):
        cx, cy, log_r = params
        r = np.exp(log_r)
        d_in = np.hypot(P_in[:, 0] - cx, P_in[:, 1] - cy)
        d_out = np.hypot(P_out[:, 0] - cx, P_out[:, 1] - cy)
        z = np.concatenate([k * (d_in - r), k * (r - d_out)])
        return np.logaddexp(0, z).mean()

    fitted = []
    for P_in, (cx, cy), r0 in zip(clusters, centers, radius):
        P_in = P_in[:, :2]
        r0 = max(r0, ARENA / 100)
        near = np.hypot(valid[:, 0] - cx, valid[:, 1] - cy) < 3 * r0
        P_out = valid[near, :2]
        x = np.array([cx, cy, np.log(r0)])
        for sharp in (5, 20, 100):  # start soft, end sharp
            x = scipy.optimize.minimize(loss, x, args=(P_in, P_out, sharp / r0)).x
        fitted.append((x[0], x[1], np.exp(x[2])))

    def cost(x: float, y: float) -> int:
        if abs(x) > ARENA or abs(y) > ARENA:
            return 1
        return int(any(np.hypot(x - cx, y - cy) <= r for cx, cy, r in fitted))

    return cost


def h_car_circle(s: np.ndarray) -> float:
    """
    Analytical safety constraint for CarCircle.

    Args:
        s: Physical agent state with world x-coordinate at index 0.

    Returns:
        h(s) <= 0 if safe, h(s) > 0 if unsafe.
    """
    x = float(np.asarray(s)[0])
    return abs(x) - 6.0






def h_point_goal(s: np.ndarray) -> float:
    """
    Approximate PointGoal1 safety constraint from the observation.

    PointGoal1 hazards have radius 0.2.
    The pseudo-lidar maximum detection distance is 3.0.

    Returns:
        h(s) > 0: estimated position is inside a hazard.
        h(s) <= 0: estimated position is outside hazards.
    """
    s = np.asarray(s, dtype=float).ravel()

    if s.size != 60:
        raise ValueError(f"Expected 60 observation features, got {s.size}")

    # Observation layout: 12 agent features, followed by
    # 16 goal-lidar, 16 hazard-lidar, and 16 vase-lidar values.
    hazard_lidar = s[28:44]

    # Convert the strongest pseudo-lidar reading into an
    # estimated distance to the nearest detected hazard.
    nearest_distance = 3.0 * (1.0 - np.clip(np.max(hazard_lidar), 0.0, 1.0))

    return float(0.2 - nearest_distance)
