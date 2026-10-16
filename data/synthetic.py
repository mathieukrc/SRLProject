from typing import Callable

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from .offline_dataset import OfflineDataset


UncertaintyFunction = Callable[[torch.Tensor], torch.Tensor]
ViolationFunction = Callable[[torch.Tensor], torch.Tensor]


START_TARGETING_RULES = {
    "random",
    "density_matched",
    "noise_augmentation",
    "uncertainty_guided",
    "violation_guided",
    "combined",
}


def _get_states(obs: OfflineDataset) -> torch.Tensor:
    states = obs.data["observations"]

    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError("Observations must have shape (N, d), N > 0")

    if not torch.isfinite(states).all().item():
        raise ValueError("Observations must be finite")

    return states


def _check_budget(budget: int) -> None:
    if not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError("budget must be a non-negative integer")

def _make_generator(states: torch.Tensor, seed: int | None):
    if seed is None:
        return None

    generator = torch.Generator(device=states.device)
    generator.manual_seed(seed)

    return generator

def _bounds(states: torch.Tensor):
    return states.min(dim=0).values, states.max(dim=0).values


def _uniform(
    states: torch.Tensor,
    budget: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:

    lower, upper = _bounds(states)

    random_values = torch.rand(
        (budget, states.shape[1]),
        device=states.device,
        dtype=states.dtype,
        generator=generator,
    )

    return lower + random_values * (upper - lower)


def _noise(
    states: torch.Tensor,
    budget: int,
    noise_scale: float = 0.05,
    generator: torch.Generator | None = None,
) -> torch.Tensor:

    if not np.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("noise_scale must be finite and non-negative")

    lower, upper = _bounds(states)

    indices = torch.randint(
        len(states),
        (budget,),
        device=states.device,
        generator=generator,
    )

    base = states[indices]

    noise = torch.randn(
        base.shape,
        device=states.device,
        dtype=states.dtype,
        generator=generator,
    ) * (noise_scale * (upper - lower))

    return torch.clamp(
        base + noise,
        min=lower,
        max=upper,
    )


def _candidates(
    states: torch.Tensor,
    budget: int,
    candidate_multiplier: int = 10,
    noise_scale: float = 0.05,
    generator: torch.Generator | None = None,
) -> torch.Tensor:

    if (
        not isinstance(candidate_multiplier, (int, np.integer))
        or isinstance(candidate_multiplier, (bool, np.bool_))
        or candidate_multiplier < 1
    ):
        raise ValueError("candidate_multiplier must be a positive integer")

    pool_size = max(1000, budget * candidate_multiplier)

    n_uniform = pool_size // 2
    n_noise = pool_size - n_uniform

    return torch.cat(
        [
            _uniform(states, n_uniform, generator),
            _noise(states, n_noise, noise_scale, generator),
        ],
        dim=0,
    )

def _scores(
    fn: Callable,
    candidates: torch.Tensor,
    name: str,
) -> torch.Tensor:
    with torch.no_grad():
        values = fn(candidates)

    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must return a torch.Tensor")

    if values.device != candidates.device:
        raise ValueError(
            f"{name} must return scores on the same device as candidates"
        )

    values = values.reshape(-1)

    if values.shape[0] != candidates.shape[0]:
        raise ValueError(
            f"{name} must return one value per candidate"
        )

    if not torch.isfinite(values).all().item():
        raise ValueError(f"{name} returned non-finite values")

    return values


def _violations(
    fn: ViolationFunction,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Validate boolean violation predictions."""

    values = _scores(
        fn,
        candidates,
        "violation_fn",
    )

    if values.dtype != torch.bool:
        raise TypeError(
            "violation_fn must return a boolean tensor"
        )

    return values


# --------------------------------------------------
# 1. Random sampling
# --------------------------------------------------

def random_chain_start_picker(
    obs: OfflineDataset,
    budget: int,
    seed: int | None = None,
) -> torch.Tensor:

    _check_budget(budget)
    states = _get_states(obs)

    generator = _make_generator(states, seed)

    return _uniform(states, budget, generator)


# --------------------------------------------------
# 2. Density-matched sampling
# --------------------------------------------------

def density_matched_chain_start_picker(
    obs: OfflineDataset,
    budget: int,
    n_components: int = 5,
    seed: int = 42,
) -> torch.Tensor:
    """Sample states from a GMM fitted to offline observations."""

    _check_budget(budget)
    states = _get_states(obs)

    if budget == 0:
        return states[:0].clone()

    data = states.detach().cpu().numpy()

    components = min(n_components, len(data))

    if components < 1:
        raise ValueError("n_components must be positive")

    model = GaussianMixture(
        n_components=components,
        covariance_type="full",
        reg_covar=1e-5,
        random_state=seed,
    )

    model.fit(data)

    generated, _ = model.sample(budget)

    result = torch.as_tensor(
        generated,
        dtype=states.dtype,
        device=states.device,
    )

    lower, upper = _bounds(states)

    return torch.maximum(
        torch.minimum(result, upper),
        lower,
    )


# --------------------------------------------------
# 3. Gaussian noise augmentation
# --------------------------------------------------

def noise_augmentation_chain_start_picker(
    obs: OfflineDataset,
    budget: int,
    noise_scale: float = 0.05,
    seed: int | None = None,
) -> torch.Tensor:

    _check_budget(budget)
    states = _get_states(obs)

    generator = _make_generator(states, seed)

    return _noise(states, budget, noise_scale, generator)


# --------------------------------------------------
# 4. Uncertainty-guided sampling
# --------------------------------------------------

def uncertainty_guided_chain_start_picker(
    obs: OfflineDataset,
    budget: int,
    uncertainty_fn: UncertaintyFunction,
    candidate_multiplier: int = 10,
    seed: int | None = None,
) -> torch.Tensor:
    """Select candidate states with highest model uncertainty."""

    _check_budget(budget)
    states = _get_states(obs)

    if budget == 0:
        return states[:0].clone()

    generator = _make_generator(states, seed)

    candidates = _candidates(
        states,
        budget,
        candidate_multiplier,
        generator=generator,
    )

    scores = _scores(
        uncertainty_fn,
        candidates,
        "uncertainty_fn",
    )

    indices = torch.topk(
        scores, k=budget, largest=True
    ).indices

    return candidates[indices]


# --------------------------------------------------
# 5. Violation-guided sampling
# --------------------------------------------------

def violation_guided_chain_start_picker(
    obs: OfflineDataset,
    budget: int,
    violation_fn: ViolationFunction,
    candidate_multiplier: int = 10,
    seed: int | None = None,
) -> torch.Tensor:
    """Prioritize states where critic and safety model disagree."""

    _check_budget(budget)
    states = _get_states(obs)

    if budget == 0:
        return states[:0].clone()

    generator = _make_generator(states, seed)

    candidates = _candidates(
        states,
        budget,
        candidate_multiplier,
        generator=generator,
    )

    violations = _violations(
        violation_fn,
        candidates,
    )

    violating_indices = torch.where(violations)[0]
    other_indices = torch.where(~violations)[0]

    # If there are fewer than budget violations,
    # fill remaining positions with other candidates.
    indices = torch.cat(
        [violating_indices, other_indices]
    )[:budget]

    return candidates[indices]


# --------------------------------------------------
# 6. Combined violation + uncertainty sampling
# --------------------------------------------------

def combined_chain_start_picker(
    obs: OfflineDataset,
    budget: int,
    uncertainty_fn: UncertaintyFunction,
    violation_fn: ViolationFunction,
    candidate_multiplier: int = 10,
    seed: int | None = None,
) -> torch.Tensor:
    """
    First prioritize violating states.
    Then rank candidates by uncertainty.
    """

    _check_budget(budget)
    states = _get_states(obs)

    if budget == 0:
        return states[:0].clone()

    generator = _make_generator(states, seed)

    candidates = _candidates(
        states,
        budget,
        candidate_multiplier,
        generator=generator,
    )

    uncertainty = _scores(
        uncertainty_fn,
        candidates,
        "uncertainty_fn",
    )

    violations = _violations(
        violation_fn,
        candidates,
    )

    violating_indices = torch.where(violations)[0]
    other_indices = torch.where(~violations)[0]

    # Rank each group by uncertainty.
    violating_sorted = violating_indices[
        torch.argsort(
            uncertainty[violating_indices],
            descending=True,
        )
    ]

    other_sorted = other_indices[
        torch.argsort(
            uncertainty[other_indices],
            descending=True,
        )
    ]

    # Prefer violations, filling any shortage with
    # highest-uncertainty non-violating candidates.
    selected = torch.cat(
        [violating_sorted, other_sorted]
    )[:budget]

    return candidates[selected]
