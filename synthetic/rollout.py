"""
Turning chain-start states into labelled synthetic transitions:

  starts --(action proposer)--> actions --(transition model)--> next states
         --(label_transitions: training h + termination rule)--> costs, masks
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from data.offline_dataset import OfflineDataset
from data.utils import HFn, label_costs
from envs.registry import EnvSpec, TerminalFn

ACT_LIM = 1 - 1e-5  # same clipping as OfflineDataset


# ----------------------------------------------------------------------------
# Transition models
# ----------------------------------------------------------------------------
class SimulatorTransitionModel:
    """controlled_env.dynamics; disagreement is always 0 (oracle condition)."""

    disagreement_scale = 1.0

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)

    @torch.no_grad()
    def step(self, obs, act, generator=None):
        from controlled.controlled_env import dynamics

        nxt = dynamics(obs.double().cpu().numpy(), act.double().cpu().numpy())
        nxt = torch.as_tensor(nxt, dtype=obs.dtype, device=obs.device)
        return nxt, torch.zeros(len(obs), device=obs.device)


class LearnedTransitionModel:
    """Wraps DynamicsEnsemble; samples a random member per transition (MBPO-style)."""

    def __init__(self, ensemble) -> None:
        self.ensemble = ensemble
        self.disagreement_scale = float(ensemble.disagreement_scale)

    @torch.no_grad()
    def step(self, obs, act, generator=None):
        member = torch.randint(
            self.ensemble.n_members, (len(obs),), device=obs.device, generator=generator
        )
        return self.ensemble.predict(obs, act, member)


def make_transition_model(kind: str, spec: EnvSpec, dynamics_path: str, device: str):
    if kind == "simulator":
        if not spec.has_simulator:
            raise ValueError(f"{spec.name} has no simulator")
        return SimulatorTransitionModel(torch.device(device))
    from dynamics.ensemble import load_dynamics

    return LearnedTransitionModel(load_dynamics(dynamics_path, device))


# ----------------------------------------------------------------------------
# Action proposers
# ----------------------------------------------------------------------------
ActionProposer = Callable[[torch.Tensor, Optional[torch.Generator]], torch.Tensor]


def random_actions(states: torch.Tensor, act_dim: int, generator=None) -> torch.Tensor:
    u = torch.rand((len(states), act_dim), device=states.device, generator=generator)
    return (2 * u - 1).clamp(-ACT_LIM, ACT_LIM)


class _BehaviourIndex:
    """Random subsample of the real data for nearest-neighbour action lookup."""

    def __init__(self, dataset: OfflineDataset, max_ref: int = 20000, seed: int = 0) -> None:
        n = len(dataset)
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(n, max_ref), replace=False)
        idx_t = torch.as_tensor(idx, device=dataset.device)
        obs = dataset.data["observations"][idx_t]
        self.mean, self.std = obs.mean(0), obs.std(0).clamp_min(1e-6)
        self.obs = (obs - self.mean) / self.std
        self.act = dataset.data["actions"][idx_t]


def behaviour_actions(
    states: torch.Tensor,
    dataset: OfflineDataset,
    k: int = 10,
    noise_std: float = 0.1,
    generator=None,
    _index: Optional[_BehaviourIndex] = None,
) -> torch.Tensor:
    """Action of a random one of the k nearest real states, plus noise."""
    index = _index or _BehaviourIndex(dataset)
    q = (states - index.mean) / index.std
    out = []
    for i in range(0, len(q), 4096):
        d = torch.cdist(q[i : i + 4096], index.obs)
        nn_idx = d.topk(min(k, d.shape[1]), largest=False).indices  # (b, k)
        pick = torch.randint(nn_idx.shape[1], (len(nn_idx),), device=q.device, generator=generator)
        out.append(index.act[nn_idx[torch.arange(len(nn_idx), device=q.device), pick]])
    a = torch.cat(out)
    noise = torch.randn(a.shape, device=a.device, generator=generator) * noise_std
    return (a + noise).clamp(-ACT_LIM, ACT_LIM)


@torch.no_grad()
def model_recovery_actions(
    states: torch.Tensor,
    model,
    h_fn: HFn,
    act_dim: int,
    n_candidates: int = 64,
    lookahead: int = 3,
    generator=None,
) -> torch.Tensor:
    """Sample n_candidates actions per state, hold each for `lookahead` model steps,
    return the one minimising max_t h(s_t)."""
    B = len(states)
    if B == 0:
        return torch.empty((0, act_dim), device=states.device)
    s = states.repeat_interleave(n_candidates, dim=0)
    a = random_actions(s, act_dim, generator)
    worst = torch.full((len(s),), -float("inf"), device=states.device)
    for _ in range(lookahead):
        s, _ = model.step(s, a, generator)
        worst = torch.maximum(worst, h_fn(s).reshape(-1))
    best = worst.view(B, n_candidates).argmin(dim=1)
    return a.view(B, n_candidates, act_dim)[torch.arange(B, device=states.device), best]


def make_action_proposer(
    name: str,
    spec: EnvSpec,
    model,
    dataset: OfflineDataset,
    h_fn: HFn,
    mix: Optional[Dict[str, float]] = None,
) -> ActionProposer:
    act_dim = dataset.data["actions"].shape[-1]
    index = _BehaviourIndex(dataset) if name in ("behaviour", "mixed") else None

    proposers: Dict[str, ActionProposer] = {
        "random": lambda s, g: random_actions(s, act_dim, g),
        "behaviour": lambda s, g: behaviour_actions(s, dataset, generator=g, _index=index),
        "model_recovery": lambda s, g: model_recovery_actions(s, model, h_fn, act_dim, generator=g),
    }
    if name != "mixed":
        return proposers[name]

    weights = mix or {"model_recovery": 0.6, "behaviour": 0.2, "random": 0.2}
    names = list(weights)
    probs = torch.tensor([weights[n] for n in names], dtype=torch.float32)

    def mixed(states: torch.Tensor, generator=None) -> torch.Tensor:
        choice = torch.multinomial(
            probs.to(states.device), len(states), replacement=True, generator=generator
        )
        out = torch.empty((len(states), act_dim), device=states.device)
        for i, n in enumerate(names):
            m = choice == i
            if m.any():
                out[m] = proposers[n](states[m], generator)
        return out

    return mixed


# ----------------------------------------------------------------------------
# Rollouts and labels
# ----------------------------------------------------------------------------
SyntheticTransitions = Dict[str, torch.Tensor]
RawChains = Dict[str, torch.Tensor]


@torch.no_grad()
def rollout_chains(
    starts: torch.Tensor,
    model,
    proposer: ActionProposer,
    horizon: int,
    generator: Optional[torch.Generator] = None,
) -> RawChains:
    n = len(starts)
    chain = torch.arange(n, device=starts.device)
    rows = {k: [] for k in ("observations", "actions", "next_observations",
                            "disagreement", "chain_id", "step")}
    s = starts
    for t in range(horizon):
        a = proposer(s, generator)
        s2, dis = model.step(s, a, generator)
        for k, v in zip(rows, (s, a, s2, dis, chain, torch.full_like(chain, t))):
            rows[k].append(v)
        s = s2
    return {k: torch.cat(v) for k, v in rows.items()}


def select(trans: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: v[mask] for k, v in trans.items()}


def _first_step_where(flag: torch.Tensor, chain_id: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """Per row: the first step of its chain at which `flag` is True (horizon if never)."""
    n_chains = int(chain_id.max().item()) + 1 if len(chain_id) else 0
    big = int(step.max().item()) + 1 if len(step) else 0
    first = torch.full((n_chains,), big, dtype=torch.long, device=step.device)
    if flag.any():
        first.scatter_reduce_(0, chain_id[flag], step[flag], reduce="amin")
    return first[chain_id]


@torch.no_grad()
def label_transitions(raw: RawChains, training_h: HFn, is_terminal_fn: TerminalFn) -> SyntheticTransitions:
    """costs exactly as for the real data; steps after a terminal step are dropped."""
    terminal = is_terminal_fn(raw["observations"], raw["next_observations"])
    masks = (~terminal).float()
    costs = label_costs(training_h, raw["observations"], raw["next_observations"], masks)
    out = dict(raw)
    out["costs"] = costs
    out["masks"] = masks
    out["rewards"] = torch.zeros_like(costs)
    first_term = _first_step_where(terminal, raw["chain_id"], raw["step"])
    return select(out, raw["step"] <= first_term)


# ----------------------------------------------------------------------------
# Filter
# ----------------------------------------------------------------------------
@torch.no_grad()
def filter_transitions(
    trans: SyntheticTransitions,
    valid_fn: Callable[[torch.Tensor], torch.Tensor],
    max_disagreement: float,
) -> torch.Tensor:
    """Start state valid; next state valid unless the transition is terminal (leaving
    the arena is valid information); disagreement under threshold; finite."""
    finite = torch.stack(
        [torch.isfinite(trans[k]).reshape(len(trans["costs"]), -1).all(dim=-1)
         for k in ("observations", "next_observations", "actions", "costs")]
    ).all(dim=0)
    obs_ok = valid_fn(trans["observations"])
    next_ok = valid_fn(trans["next_observations"]) | (trans["masks"] == 0)
    dis_ok = trans["disagreement"] <= max_disagreement
    return finite & obs_ok & next_ok & dis_ok


def truncate_chains(keep: torch.Tensor, chain_id: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """Once a step is rejected, reject every later step of that chain."""
    first_bad = _first_step_where(~keep, chain_id, step)
    return keep & (step < first_bad)
