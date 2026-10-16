"""
Interface to the controlled environment, following the team's problem formulation.

Functions:
    check_feasible(states)               -> bool array (N,)
    get_offline_observations()           -> dict with 'observations', 'next_observations',
                                            'actions', 'rewards', 'costs', 'terminals', 'timeouts'
    compute_feasibility_metrics(s, feas) -> dict with false-feasible / false-infeasible
                                            proportions and IoU

Shapes (N transitions, n_dims = 4 state dimensions (x, y, vx, vy), action_dims = 2):
    observations       (N, 4)
    next_observations  (N, 4)
    actions            (N, 2)
    rewards            (N,)
    costs              (N,)

Needs controlled_env.py in the same folder. The first call computes (or loads) the
ground truth and the dataset and caches them in ./env_outputs, so later calls are fast.
"""

import os
from typing import Dict

import controlled.controlled_env as ce
import numpy as np

_GT = None  # cached ground-truth query object


def _ground_truth() -> "ce.GroundTruth":
    """Load the ground-truth value function from disk, or compute and save it."""
    global _GT
    if _GT is None:
        path = os.path.join(ce.OUT_DIR, "ground_truth.npz")
        if os.path.exists(path):
            raw = np.load(path)
            gt = {k: raw[k] for k in ("V", "pos", "vel", "h")}
        else:
            os.makedirs(ce.OUT_DIR, exist_ok=True)
            gt = ce.compute_ground_truth(verbose=False)
            np.savez_compressed(path, **gt)
        _GT = ce.GroundTruth(gt)
    return _GT


def check_feasible(states: np.ndarray) -> np.ndarray:
    """
    Args:
        states: array of shape (N, n_dims) containing states (x, y, vx, vy)
    Returns:
        A boolean array of shape (N,) containing the feasibility of each state
        in the input (True = feasible, i.e. some policy can keep it safe forever).
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:  # allow a single state of shape (n_dims,)
        states = states[None, :]
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError(f"states must have shape (N, 4), got {states.shape}")
    return _ground_truth().feasible(states).astype(bool)


def get_offline_observations() -> Dict[str, np.ndarray]:
    """
    Returns:
        A dictionary with keys 'observations', 'next_observations', 'actions',
        'rewards', 'costs', 'terminals' and 'timeouts'. Loaded from the precomputed
        file if it exists, otherwise generated (and saved) first.
    """
    path = os.path.join(ce.OUT_DIR, "dataset.npz")
    if not os.path.exists(path):
        os.makedirs(ce.OUT_DIR, exist_ok=True)
        q = _ground_truth()
        gt = dict(V=q.V, pos=q.pos, vel=q.vel)
        data = ce.annotate_dataset(ce.collect(), gt)
        np.savez_compressed(path, **data)
    d = np.load(path)
    return {
        "observations": d["obs"].astype(np.float32),
        "next_observations": d["next_obs"].astype(np.float32),
        "actions": d["act"].astype(np.float32),
        "rewards": d["rew"].astype(np.float32),
        "costs": d["cost"].astype(np.float32),
        "terminals": d["terminal"].astype(np.float32),
        "timeouts": d["timeout"].astype(np.float32),
    }


def export_h5(path: str = "offline_dataset.h5") -> str:
    """Optional: store the dataset in an h5 file (needs `pip install h5py`)."""
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py is not installed: run  pip install h5py") from e
    data = get_offline_observations()
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v, compression="gzip")
    return path


def compute_feasibility_metrics(
    states: np.ndarray, feasible: np.ndarray
) -> Dict[str, float]:
    """
    Args:
        states: Array of shape (N, state_dim) containing a list of states
        feasible: Array of shape (N,) containing whether the corresponding states are deemed feasible
    Returns:
        Proportion of states that are false-feasible and false-infeasible, as well as the Intersection over Union (IoU)

        Keys:
            false_feasible:   fraction of all N states that are truly infeasible but deemed feasible
                              (the unsafe mistake)
            false_infeasible: fraction of all N states that are truly feasible but deemed infeasible
                              (overly conservative)
            iou:              |deemed feasible AND truly feasible| / |deemed feasible OR truly feasible|
                              (nan if neither set contains any state)
    """
    states = np.asarray(states)
    pred = np.asarray(feasible).astype(bool)
    if pred.ndim != 1 or len(pred) != len(states):
        raise ValueError(
            f"feasible must have shape (N,) matching states, got {pred.shape} for {len(states)} states"
        )

    true = check_feasible(states)

    n = len(true)
    tp = np.sum(pred & true)
    fp = np.sum(pred & ~true)  # infeasible, deemed feasible
    fn = np.sum(~pred & true)  # feasible, deemed infeasible
    union = tp + fp + fn

    return {
        "false_feasible": float(fp / n) if n > 0 else float("nan"),
        "false_infeasible": float(fn / n) if n > 0 else float("nan"),
        "iou": float(tp / union) if union > 0 else float("nan"),
    }


if __name__ == "__main__":
    # quick self-test
    d = get_offline_observations()
    N = len(d["observations"])
    shapes = {k: v.shape for k, v in d.items()}
    print(f"N={N}  shapes={shapes}")
    assert d["observations"].shape == (N, 4) and d["next_observations"].shape == (
        N,
        4,
    )
    assert (
        d["actions"].shape == (N, 2)
        and d["rewards"].shape == (N,)
        and d["costs"].shape == (N,)
    )
    feas = check_feasible(d["observations"])
    assert feas.shape == (N,) and feas.dtype == bool
    print(f"feasible fraction of dataset states: {feas.mean():.3f}")
    tests = np.array(
        [
            [-0.8, 0.8, 0.0, 0.0],  # far from obstacles, at rest   -> feasible
            [0.4, 0.10, 0.0, 0.0],  # near obstacle, at rest        -> feasible
            [0.4, 0.10, 0.0, -1.0],  # near obstacle, driving at it  -> infeasible
            [0.8, 0.0, 1.0, 0.0],
        ]
    )  # near wall, moving at it fast  -> infeasible
    print("check_feasible on 4 hand-made states:", check_feasible(tests))
    print("expected: [True, True, False, False]")

    # metrics: the ground truth scored against itself must be perfect
    m = compute_feasibility_metrics(d["observations"], feas)
    assert (
        m["false_feasible"] == 0.0 and m["false_infeasible"] == 0.0 and m["iou"] == 1.0
    )
    # ...and the complement must be maximally wrong
    m = compute_feasibility_metrics(d["observations"], ~feas)
    assert (
        m["iou"] == 0.0
        and abs(m["false_feasible"] + m["false_infeasible"] - 1.0) < 1e-9
    )
    print("self-test passed")
