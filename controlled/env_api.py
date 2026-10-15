"""
Interface to the controlled environment, following the team's problem formulation.

Two functions:
    check_feasible(states)          -> bool array (N,)
    get_offline_observations(...)   -> dict with 'observations', 'next_observations',
                                       'actions', 'rewards', 'costs'

Shapes (N transitions, n_dims = 4 state dimensions (x, y, vx, vy), action_dims = 2):
    observations       (N, 4)
    next_observations  (N, 4)
    actions            (N, 2)
    rewards            (N,)
    costs              (N,)

Needs controlled_env.py in the same folder. The first call computes (or loads) the
ground truth and the datasets and caches them in ./env_outputs, so later calls are fast.
"""

import os
from typing import Dict

import numpy as np

import controlled.controlled_env as ce

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


def get_offline_observations(variant: str = "full") -> Dict[str, np.ndarray]:
    """
    Args:
        variant: "full" (rich boundary coverage) or "thin" (boundary band removed)
    Returns:
        A dictionary with keys 'observations', 'next_observations', 'actions',
        'rewards' and 'costs'. Loaded from the precomputed file if it exists,
        otherwise generated (and saved) first.
    """
    if variant not in ("full", "thin"):
        raise ValueError("variant must be 'full' or 'thin'")
    path = os.path.join(ce.OUT_DIR, f"dataset_{variant}.npz")
    if not os.path.exists(path):
        os.makedirs(ce.OUT_DIR, exist_ok=True)
        q = _ground_truth()
        gt = dict(V=q.V, pos=q.pos, vel=q.vel)
        full, thin = ce.make_variants(ce.collect(), gt)
        np.savez_compressed(os.path.join(ce.OUT_DIR, "dataset_full.npz"), **full)
        np.savez_compressed(os.path.join(ce.OUT_DIR, "dataset_thin.npz"), **thin)
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


def export_h5(path: str = "offline_dataset.h5", variant: str = "full") -> str:
    """Optional: store a dataset variant in an h5 file (needs `pip install h5py`)."""
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py is not installed: run  pip install h5py") from e
    data = get_offline_observations(variant)
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v, compression="gzip")
    return path


if __name__ == "__main__":
    # quick self-test
    for variant in ("full", "thin"):
        d = get_offline_observations(variant)
        N = len(d["observations"])
        shapes = {k: v.shape for k, v in d.items()}
        print(f"[{variant}] N={N}  shapes={shapes}")
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
        print(f"[{variant}] feasible fraction of dataset states: {feas.mean():.3f}")
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
    print("self-test passed")
