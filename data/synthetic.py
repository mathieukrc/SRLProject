from typing import Callable, Optional

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from .offline_dataset import OfflineDataset

UncertaintyFunction = Callable[
    [torch.Tensor], torch.Tensor
]  # Uncertainty at a given state
ViolationFunction = Callable[[torch.Tensor], torch.Tensor]  # relu(h(s) - V_h(s))
ValidFunction = Callable[[torch.Tensor], torch.Tensor]  # usable chain start


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


def _make_generator(states: torch.Tensor, seed: Optional[int]):
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
    generator: Optional[torch.Generator] = None,
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
    generator: Optional[torch.Generator] = None,
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
    generator: Optional[torch.Generator] = None,
    valid_fn: Optional[ValidFunction] = None,
    uniform_frac: float = 0.5,
) -> torch.Tensor:

    if not 0.0 <= uniform_frac <= 1.0:
        raise ValueError("uniform_frac must be in [0, 1]")

    if (
        not isinstance(candidate_multiplier, (int, np.integer))
        or isinstance(candidate_multiplier, (bool, np.bool_))
        or candidate_multiplier < 1
    ):
        raise ValueError("candidate_multiplier must be a positive integer")

    pool_size = max(1000, budget * candidate_multiplier)

    n_uniform = int(round(pool_size * uniform_frac))
    n_noise = pool_size - n_uniform

    return torch.cat(
        [
            _fill_valid(lambda n: _uniform(states, n, generator), n_uniform, valid_fn),
            _fill_valid(
                lambda n: _noise(states, n, noise_scale, generator), n_noise, valid_fn
            ),
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
        raise ValueError(f"{name} must return scores on the same device as candidates")

    values = values.reshape(-1)

    if values.shape[0] != candidates.shape[0]:
        raise ValueError(f"{name} must return one value per candidate")

    if not torch.isfinite(values).all().item():
        raise ValueError(f"{name} returned non-finite values")

    return values


def _mask(fn: ValidFunction, candidates: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        m = fn(candidates).reshape(-1)
    if m.dtype != torch.bool or m.shape[0] != candidates.shape[0]:
        raise ValueError("valid_fn must return one bool per candidate")
    return m


def _fill_valid(
    sample_fn, n: int, valid_fn: Optional[ValidFunction], max_tries: int = 20
):
    if valid_fn is None or n == 0:
        return sample_fn(n)
    kept, total = [], 0
    for _ in range(max_tries):
        batch = sample_fn(max(2 * (n - total), 16))
        batch = batch[_mask(valid_fn, batch)]
        kept.append(batch)
        total += len(batch)
        if total >= n:
            return torch.cat(kept)[:n]
    raise ValueError(
        f"Only {total}/{n} valid samples after {max_tries} tries; check valid_fn"
    )


def _select(scores: torch.Tensor, k: int, generator=None, beta: Optional[float] = 1.0):
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    if k > scores.shape[0]:
        raise ValueError("k larger than number of candidates")
    if beta is None:
        return torch.topk(scores, k=k).indices
    w = scores.clamp_min(0)
    w = (w / (w.max() + 1e-12)) ** beta + 1e-8  # all-zero scores -> uniform
    return torch.multinomial(w, k, replacement=False, generator=generator)


def _two_tier(
    primary: torch.Tensor, rank_scores: torch.Tensor, k: int, generator, beta
):
    p_idx, o_idx = torch.where(primary)[0], torch.where(~primary)[0]
    n_p = min(k, len(p_idx))
    chosen_p = p_idx[_select(rank_scores[p_idx], n_p, generator, beta)]
    chosen_o = o_idx[_select(rank_scores[o_idx], k - n_p, generator, beta)]
    return torch.cat([chosen_p, chosen_o])


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

    if not values.is_floating_point():
        raise TypeError(
            "violation_fn must return float scores, e.g. relu(h(s) - V_h(s))"
        )
    if (values < 0).any().item():
        raise ValueError("violation_fn scores must be non-negative")

    return values


# --------------------------------------------------
# 1. Random sampling
# --------------------------------------------------


def random_chain_start_picker(
    obs: OfflineDataset,
    budget: int,
    seed: Optional[int] = None,
    valid_fn: Optional[ValidFunction] = None,
) -> torch.Tensor:

    _check_budget(budget)
    states = _get_states(obs)

    generator = _make_generator(states, seed)

    return _fill_valid(lambda n: _uniform(states, n, generator), budget, valid_fn)


# --------------------------------------------------
# 2. Density-matched sampling
# --------------------------------------------------


def density_matched_chain_start_picker(
    obs: OfflineDataset,
    budget: int,
    n_components: int = 5,
    seed: Optional[int] = None,
    valid_fn: Optional[ValidFunction] = None,
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
    model.random_state = np.random.RandomState(seed)

    lower, upper = _bounds(states)

    def sample_fn(n):
        generated, _ = model.sample(n)
        out = torch.as_tensor(generated, dtype=states.dtype, device=states.device)
        return torch.maximum(torch.minimum(out, upper), lower)

    return _fill_valid(sample_fn, budget, valid_fn)


# --------------------------------------------------
# 3. Gaussian noise augmentation
# --------------------------------------------------


def noise_augmentation_chain_start_picker(
    obs: OfflineDataset,
    budget: int,
    noise_scale: float = 0.05,
    seed: Optional[int] = None,
    valid_fn: Optional[ValidFunction] = None,
) -> torch.Tensor:

    _check_budget(budget)
    states = _get_states(obs)

    generator = _make_generator(states, seed)

    return _fill_valid(
        lambda n: _noise(states, n, noise_scale, generator), budget, valid_fn
    )


# --------------------------------------------------
# 4. Uncertainty-guided sampling
# --------------------------------------------------


def uncertainty_guided_chain_start_picker(
    obs: OfflineDataset,
    budget: int,
    uncertainty_fn: UncertaintyFunction,
    candidate_multiplier: int = 10,
    seed: Optional[int] = None,
    valid_fn: Optional[ValidFunction] = None,
    beta: Optional[float] = 1.0,
    uniform_frac: float = 0.5,
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
        valid_fn=valid_fn,
        uniform_frac=uniform_frac,
    )
    scores = _scores(uncertainty_fn, candidates, "uncertainty_fn")
    indices = _select(scores, budget, generator, beta)  # was: torch.topk(...)
    return candidates[indices]


# --------------------------------------------------
# 5. Violation-guided sampling
# --------------------------------------------------


def violation_guided_chain_start_picker(
    obs: OfflineDataset,
    budget: int,
    violation_fn: ViolationFunction,
    candidate_multiplier: int = 10,
    seed: Optional[int] = None,
    valid_fn: Optional[ValidFunction] = None,
    beta: Optional[float] = 1.0,
    margin: float = 1e-3,
    uniform_frac: float = 0.5,
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
        valid_fn=valid_fn,
        uniform_frac=uniform_frac,
    )
    viol = _violations(violation_fn, candidates)
    indices = _two_tier(viol > margin, viol, budget, generator, beta)
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
    seed: Optional[int] = None,
    valid_fn: Optional[ValidFunction] = None,
    beta: Optional[float] = 1.0,
    margin: float = 1e-3,
    uniform_frac: float = 0.5,
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
        valid_fn=valid_fn,
        uniform_frac=uniform_frac,
    )
    uncertainty = _scores(uncertainty_fn, candidates, "uncertainty_fn")
    viol = _violations(violation_fn, candidates)
    indices = _two_tier(viol > margin, uncertainty, budget, generator, beta)
    return candidates[indices]
