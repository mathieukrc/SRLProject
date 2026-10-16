import copy
import warnings
from typing import Dict, Sequence

import numpy as np
import torch
from architectures.ensemble import Ensemble
from architectures.mlp import MLP
from data.batch import Batch

from .critic_config import CriticConfig
from .loss_functions import expectile_loss, soft_update

FORMAT_VERSION = 1
ARCH_KEYS = ("obs_dim", "act_dim", "hidden_dims", "num_qs")


class RewardCritic:
    critic: Ensemble
    target_critic: Ensemble
    value: MLP
    critic_opt: torch.optim.Optimizer
    value_opt: torch.optim.Optimizer
    device: torch.device
    num_updates: int
    config: CriticConfig

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        num_qs: int = 2,
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_batch_size: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = torch.device(device)

        self.config = CriticConfig(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=list(hidden_dims),
            num_qs=num_qs,
            discount=discount,
            tau=tau,
            expectile=expectile,
            critic_lr=critic_lr,
            value_lr=value_lr,
            critic_batch_size=critic_batch_size,
        )

        self.critic = Ensemble(
            lambda: MLP(obs_dim + act_dim, hidden_dims=hidden_dims), num=num_qs
        ).to(self.device)

        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.requires_grad_(False)

        self.value = MLP(obs_dim, hidden_dims=hidden_dims).to(self.device)

        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=value_lr)

        self.num_updates = 0

    def update_v(self, batch: Batch) -> Dict[str, float]:
        obs, act = batch["observations"], batch["actions"]

        with torch.no_grad():
            q = self.target_critic(torch.cat((obs, act), dim=-1)).min(dim=0).values

        v = self.value(obs)
        loss = expectile_loss(q - v, self.config["expectile"]).mean()

        self.value_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.value_opt.step()

        with torch.no_grad():
            return {"value_loss": loss.item(), "v": v.mean().item()}

    def update_q(self, batch: Batch) -> Dict[str, float]:
        obs, act, next_obs, rewards, masks = (
            batch["observations"],
            batch["actions"],
            batch["next_observations"],
            batch["rewards"],
            batch["masks"],
        )

        with torch.no_grad():
            next_v = self.value(next_obs)
            target_q = rewards + self.config["discount"] * masks * next_v

        qs = self.critic(torch.cat((obs, act), dim=-1))
        loss = ((qs - target_q) ** 2).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()

        soft_update(
            target=self.target_critic, source=self.critic, tau=self.config["tau"]
        )

        with torch.no_grad():
            return {"critic_loss": loss.item(), "q": qs.mean().item()}

    def update(self, batch: Batch) -> Dict[str, float]:
        mini_batch = {
            k: v[: self.config["critic_batch_size"]] for k, v in batch.items()
        }

        value_info = self.update_v(mini_batch)
        critic_info = self.update_q(mini_batch)

        self.num_updates += 1

        return {**critic_info, **value_info}

    @torch.no_grad()
    def q_r(
        self, obs: torch.Tensor, act: torch.Tensor, use_target: bool = True
    ) -> torch.Tensor:
        critic = self.critic if not use_target else self.target_critic

        q_values = critic(torch.cat((obs, act), dim=-1)).min(dim=0).values
        return q_values

    @torch.no_grad()
    def v_r(self, obs: torch.Tensor) -> torch.Tensor:
        v_values = self.value(obs)
        return v_values

    def state_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "config": dict(self.config),
            "num_updates": self.num_updates,
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "value": self.value.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "value_opt": self.value_opt.state_dict(),
        }

    def load_state_dict(self, state: dict, load_optimisers: bool = True) -> None:
        if state.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format: {state.get('format_version')}"
            )

        saved = state["config"]

        bad = {
            k: (saved[k], self.config[k])
            for k in ARCH_KEYS
            if list(np.atleast_1d(saved[k])) != list(np.atleast_1d(self.config[k]))
        }

        if bad:
            raise ValueError(f"Architecture mismatch (checkpoint, current): {bad}")

        for k in set(saved) - set(ARCH_KEYS):
            if saved[k] != self.config.get(k):
                warnings.warn(f"{k}: checkpoint={saved[k]}, current={self.config[k]}")

        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.value.load_state_dict(state["value"])

        if load_optimisers:
            self.critic_opt.load_state_dict(state["critic_opt"])
            self.value_opt.load_state_dict(state["value_opt"])

        self.num_updates = state.get("num_updates", 0)
