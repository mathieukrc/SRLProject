import copy
import warnings
from typing import Dict, List, Sequence, TypedDict

import numpy as np
import torch

from architectures.ensemble import Ensemble
from architectures.mlp import MLP
from data.batch import Batch

from .loss_functions import safe_expectile_loss, soft_update

FORMAT_VERSION = 1
ARCH_KEYS = ("obs_dim", "act_dim", "hidden_dims", "num_qs")


class FeasibilityCriticConfig(TypedDict):
    obs_dim: int
    act_dim: int
    hidden_dims: List[int]
    num_qs: int
    discount: float
    tau: float
    expectile: float
    critic_lr: float
    value_lr: float
    critic_batch_size: int


class FeasibilityCritic:
    safe_critic: Ensemble
    safe_target_critic: Ensemble
    safe_value: MLP
    critic_opt: torch.optim.Optimizer
    value_opt: torch.optim.Optimizer
    device: torch.device
    num_updates: int
    config: FeasibilityCriticConfig

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        num_qs: int = 2,
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.9,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_batch_size: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = torch.device(device)

        self.config = FeasibilityCriticConfig(
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

        self.safe_critic = Ensemble(
            lambda: MLP(obs_dim + act_dim, hidden_dims), num=num_qs
        ).to(self.device)

        self.safe_target_critic = copy.deepcopy(self.safe_critic)
        self.safe_target_critic.requires_grad_(False)

        self.safe_value = MLP(obs_dim, hidden_dims).to(self.device)

        self.critic_opt = torch.optim.Adam(self.safe_critic.parameters(), lr=critic_lr)
        self.value_opt = torch.optim.Adam(self.safe_value.parameters(), lr=value_lr)

        self.num_updates = 0

    def update_vc(self, batch: Batch) -> Dict[str, float]:
        obs, act = batch["observations"], batch["actions"]

        with torch.no_grad():
            qc = (
                self.safe_target_critic(torch.cat((obs, act), dim=-1)).max(dim=0).values
            )

        vc = self.safe_value(obs)
        loss = safe_expectile_loss(qc - vc, self.config["expectile"]).mean()

        self.value_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.value_opt.step()

        with torch.no_grad():
            return {
                "safe_value_loss": loss.item(),
                "vc": vc.mean().item(),
                "vc_max": vc.max().item(),
                "vc_min": vc.min().item(),
            }

    def update_qc(self, batch: Batch) -> Dict[str, float]:
        obs, act, next_obs, costs, masks = (
            batch["observations"],
            batch["actions"],
            batch["next_observations"],
            batch["costs"],
            batch["masks"],
        )

        with torch.no_grad():
            next_vc = self.safe_value(next_obs)
            qc_nonterminal = (1.0 - self.config["discount"]) * costs + self.config[
                "discount"
            ] * torch.maximum(costs, next_vc)
            target_qc = qc_nonterminal * masks + costs * (1.0 - masks)

        qcs = self.safe_critic(torch.cat((obs, act), dim=-1))
        safe_critic_loss = ((qcs - target_qc) ** 2).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        safe_critic_loss.backward()
        self.critic_opt.step()

        soft_update(
            source=self.safe_critic,
            target=self.safe_target_critic,
            tau=self.config["tau"],
        )

        with torch.no_grad():
            return {
                "safe_critic_loss": safe_critic_loss,
                "qc": qcs.mean(),
                "qc_max": qcs.max(),
                "qc_min": qcs.min(),
                "costs": costs.mean(),
            }

    def update(self, batch: Batch) -> Dict[str, float]:
        mini_batch = {
            k: v[: self.config["critic_batch_size"]] for k, v in batch.items()
        }

        value_info = self.update_vc(mini_batch)
        critic_info = self.update_qc(mini_batch)

        self.num_updates += 1

        return {**value_info, **critic_info}

    @torch.no_grad()
    def q_h(
        self, obs: torch.Tensor, act: torch.Tensor, use_target: bool = True
    ) -> torch.Tensor:
        critic = self.safe_critic if not use_target else self.safe_target_critic

        q_values = critic(torch.cat((obs, act), dim=-1))
        q_values = q_values.max(dim=0).values
        return q_values

    @torch.no_grad()
    def v_h(self, obs: torch.Tensor) -> torch.Tensor:
        v_values = self.safe_value(obs)
        return v_values

    @torch.no_grad()
    def is_feasible(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v_h(obs) <= 0.0

    def state_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "config": dict(self.config),
            "num_updates": self.num_updates,
            "safe_critic": self.safe_critic.state_dict(),
            "safe_target_critic": self.safe_target_critic.state_dict(),
            "safe_value": self.safe_value.state_dict(),
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

        self.safe_critic.load_state_dict(state["safe_critic"])
        self.safe_target_critic.load_state_dict(state["safe_target_critic"])
        self.safe_value.load_state_dict(state["safe_value"])

        if load_optimisers:
            self.critic_opt.load_state_dict(state["critic_opt"])
            self.value_opt.load_state_dict(state["value_opt"])

        self.num_updates = state.get("num_updates", 0)
