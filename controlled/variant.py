import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from data.offline_dataset import OfflineDataset
from scipy.spatial import cKDTree

CostFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BoundaryCriterion:
    """
    Defines which transitions count as "near the boundary".

    h:               cost function on states.
    threshold:       a state is unsafe if h(s) > threshold.
    precursor_steps: also flag the k transitions before an unsafe one in the same
                     trajectory (approximates the infeasible region in front of a hazard).
    radius:          also flag states within this distance of any observed unsafe state
                     (catches near misses that never incur cost). None disables it.
    dims:            observation dimensions used for the radius distance, e.g. (0, 1) for
                     position only. None uses all dimensions. The radius is in the units
                     of these dimensions.
    """

    h: CostFn
    threshold: float = 0.0
    precursor_steps: int = 0
    radius: Optional[float] = None
    dims: Optional[Sequence[int]] = None


def split_trajectories(dataset: OfflineDataset) -> np.ndarray:
    """
    Recovers trajectory ids from the flat transition arrays. A trajectory ends where
    the episode was terminated (mask == 0) or where next_observations[i] does not
    continue into observations[i + 1] (e.g. episodes that ended by reaching the goal,
    which are not flagged as terminal/timeout).

    Returns:
        Integer array of shape (N,) with the trajectory id of every transition.
    """
    obs = dataset.data["observations"].cpu().numpy()
    next_obs = dataset.data["next_observations"].cpu().numpy()
    masks = dataset.data["masks"].cpu().numpy()

    ends = np.zeros(len(obs), dtype=bool)
    ends[-1] = True
    ends[:-1] = (masks[:-1] == 0) | np.any(next_obs[:-1] != obs[1:], axis=1)

    traj = np.concatenate([[0], np.cumsum(ends[:-1])])

    lengths = np.bincount(traj)
    if np.mean(lengths == 1) > 0.5:
        warnings.warn(
            f"split_trajectories: {np.mean(lengths == 1):.0%} of trajectories have length 1 "
            f"(mean length {lengths.mean():.1f}). Check that the dataset is unshuffled and "
            "that observations/next_observations are stored identically."
        )

    return traj


def _states(dataset: OfflineDataset) -> Tuple[np.ndarray, np.ndarray]:
    return (
        dataset.data["observations"].cpu().numpy(),
        dataset.data["next_observations"].cpu().numpy(),
    )


def _unsafe(criterion: BoundaryCriterion, states: np.ndarray) -> np.ndarray:
    return np.asarray(criterion.h(states)).reshape(-1) > criterion.threshold


def _unsafe_states(dataset: OfflineDataset, criterion: BoundaryCriterion) -> np.ndarray:
    """All observed unsafe states (from both observations and next_observations)."""
    obs, next_obs = _states(dataset)
    return np.concatenate(
        [obs[_unsafe(criterion, obs)], next_obs[_unsafe(criterion, next_obs)]]
    )


def near_boundary_transitions(
    dataset: OfflineDataset,
    criterion: BoundaryCriterion,
    reference: Optional[OfflineDataset] = None,
) -> np.ndarray:
    """
    Boolean array (N,): whether each transition is near the boundary under `criterion`.

    reference: dataset whose unsafe states define the proximity radius. Defaults to
        `dataset` itself. Pass the full original dataset when computing stats for a
        variant, so the definition doesn't shrink when unsafe trajectories are removed.
    """
    obs, next_obs = _states(dataset)
    traj = split_trajectories(dataset)

    unsafe = _unsafe(criterion, obs) | _unsafe(criterion, next_obs)
    near = unsafe.copy()

    # Precursor window: flag transition i if transition i + j (same trajectory) is unsafe.
    for j in range(1, criterion.precursor_steps + 1):
        if j >= len(unsafe):
            break
        ahead = np.zeros_like(unsafe)
        ahead[:-j] = unsafe[j:] & (traj[j:] == traj[:-j])
        near |= ahead

    # Proximity radius: flag states close to any observed unsafe state.
    if criterion.radius is not None:
        ref = _unsafe_states(reference if reference is not None else dataset, criterion)
        if len(ref) == 0:
            warnings.warn(
                "near_boundary_transitions: no unsafe states in the reference dataset; "
                "the proximity radius flags nothing."
            )
        else:
            cols = list(criterion.dims) if criterion.dims is not None else slice(None)
            tree = cKDTree(ref[:, cols])
            for states in (obs, next_obs):
                dist, _ = tree.query(
                    states[:, cols], k=1, distance_upper_bound=criterion.radius
                )
                near |= np.isfinite(dist)

    return near


def _approaches_boundary(
    dataset: OfflineDataset,
    traj: np.ndarray,
    n_traj: int,
    criterion: BoundaryCriterion,
) -> np.ndarray:
    """A trajectory approaches the boundary if any of its transitions is near it."""
    near = near_boundary_transitions(dataset, criterion)
    approaches = np.zeros(n_traj, dtype=bool)
    approaches[np.unique(traj[near])] = True
    return approaches


def get_coverage_variants(
    dataset: OfflineDataset,
    coverage_percent: float,
    criterion: BoundaryCriterion,
    seed: int = 0,
) -> OfflineDataset:
    """
    Args:
        dataset: Offline dataset
        coverage_percent: Proportion of trajectories that approach the boundary to be kept
            (in percent, 0-100). 100 keeps the dataset as is, 0 removes every trajectory
            that approaches the boundary.
        criterion: Defines which transitions count as near the boundary.
        seed: Seed for choosing which boundary trajectories are kept. With a fixed seed,
            variants are nested: a lower coverage level is a subset of a higher one.
    Returns:
        A new OfflineDataset object with only the kept trajectories

    A trajectory "approaches the boundary" if any of its transitions is unsafe
    (h > threshold at the state or next state), lies within `precursor_steps` before an
    unsafe transition, or lies within `radius` of an observed unsafe state. Trajectories
    that never come near the boundary are always kept.
    """
    if not 0.0 <= coverage_percent <= 100.0:
        raise ValueError(
            f"coverage_percent must be in [0, 100], got {coverage_percent}"
        )

    traj = split_trajectories(dataset)
    n_traj = int(traj[-1]) + 1

    approaches = _approaches_boundary(dataset, traj, n_traj, criterion)

    boundary_ids = np.flatnonzero(approaches)
    n_keep = int(round(len(boundary_ids) * coverage_percent / 100.0))

    order = np.random.default_rng(seed).permutation(boundary_ids)
    kept_boundary = order[:n_keep]

    keep_traj = ~approaches
    keep_traj[kept_boundary] = True

    return dataset.subset(np.flatnonzero(keep_traj[traj]))


def get_size_matched_variant(
    dataset: OfflineDataset, target_n_transitions: int, seed: int = 0
) -> OfflineDataset:
    """
    Removes random whole trajectories (regardless of boundary proximity) until at most
    target_n_transitions remain. Pair each coverage variant with one of these of the
    same size: if the coverage variant degrades feasibility estimation more, the effect
    is due to boundary coverage rather than dataset size. With a fixed seed, outputs are
    nested across target sizes.
    """
    traj = split_trajectories(dataset)
    n_traj = int(traj[-1]) + 1
    lengths = np.bincount(traj, minlength=n_traj)

    order = np.random.default_rng(seed).permutation(n_traj)
    remove = np.zeros(n_traj, dtype=bool)
    remaining = int(lengths.sum())
    for t in order:
        if remaining <= target_n_transitions:
            break
        remove[t] = True
        remaining -= int(lengths[t])

    return dataset.subset(np.flatnonzero(~remove[traj]))


def coverage_stats(
    dataset: OfflineDataset,
    criterion: BoundaryCriterion,
    reference: Optional[OfflineDataset] = None,
) -> dict:
    """
    reference: pass the full original dataset so the proximity radius is measured
        against the same unsafe states for every variant.
    """
    obs, next_obs = _states(dataset)
    traj = split_trajectories(dataset)
    unsafe = _unsafe(criterion, obs) | _unsafe(criterion, next_obs)
    near = near_boundary_transitions(dataset, criterion, reference=reference)
    return {
        "n_transitions": int(len(obs)),
        "n_trajectories": int(traj[-1]) + 1,
        "unsafe_fraction": float(np.mean(unsafe)),
        "near_fraction": float(np.mean(near)),
    }
