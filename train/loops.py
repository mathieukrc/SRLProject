"""Training loops on top of MixedSampler."""

import os
from typing import Callable, Dict, Optional, Union

import torch

from data.mixed import MixedSampler
from FISOR.agent import FISORAgent, FISORConfig
from FISOR.ensemble_feasibility_critic import EnsembleFeasibilityCritic
from FISOR.feasibility_critic import FeasibilityCritic

Critic = Union[FeasibilityCritic, EnsembleFeasibilityCritic]
LogFn = Callable[[Dict[str, float], int], None]


def _atomic_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_feasibility_critic(
    critic: Critic,
    sampler: MixedSampler,
    num_steps: int,
    batch_size: int = 256,
    anchor_weight: float = 0.0,
    log_every: int = 1000,
    log_fn: Optional[LogFn] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: Optional[int] = None,
) -> Critic:
    start = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        ck = torch.load(checkpoint_path, map_location=critic.device, weights_only=False)
        critic.load_state_dict(ck["critic"])
        start = ck["step"] + 1

    for step in range(start, num_steps + 1):
        batch, anchors = sampler.sample(batch_size)
        info = critic.update(batch, anchors, anchor_weight)
        if log_fn is not None and step % log_every == 0:
            log_fn(info, step)
        if checkpoint_path and checkpoint_every and step % checkpoint_every == 0:
            _atomic_save({"critic": critic.state_dict(), "step": step}, checkpoint_path)

    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    return critic


def build_agent(
    obs_dim: int,
    act_dim: int,
    critic: EnsembleFeasibilityCritic,
    discount: float,
    device: str,
    **config_overrides,
) -> FISORAgent:
    config = FISORConfig(discount=discount, **config_overrides)
    return FISORAgent(obs_dim, act_dim, config, device=device, feasibility_critic=critic)


def train_fisor_agent(
    agent: FISORAgent,
    sampler: MixedSampler,
    num_steps: int,
    anchor_weight: float = 0.0,
    use_synthetic_actions: bool = False,
    log_every: int = 5000,
    log_fn: Optional[LogFn] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: Optional[int] = 50_000,
) -> FISORAgent:
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)

    while agent.step < num_steps:
        batch, anchors = sampler.sample(agent.config.batch_size)
        info = agent.update(batch, anchors, anchor_weight, use_synthetic_actions)
        if log_fn is not None and agent.step % log_every == 0:
            log_fn(info, agent.step)
        if checkpoint_path and checkpoint_every and agent.step % checkpoint_every == 0:
            agent.save(checkpoint_path)

    if checkpoint_path:
        agent.save(checkpoint_path)
    return agent
