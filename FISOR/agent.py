import dataclasses
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from architectures.diffusion_actor import DiffusionActor
from architectures.diffusion_schedule import DiffusionSchedule
from data.batch import Batch

from .feasibility_critic import FeasibilityCritic
from .reward_critic import RewardCritic

ARCH_KEYS = (
    "critic_hidden_dims",
    "num_qs",
    "T",
    "time_dim",
    "actor_num_blocks",
    "actor_layer_norm",
)

CHECKPOINT_VERSION = 1


def _normalise(value: Any) -> Any:
    return tuple(value) if isinstance(value, (list, tuple)) else value


@torch.no_grad()
def feasibility_weights(
    q_r: torch.Tensor,
    v_r: torch.Tensor,
    q_h: torch.Tensor,
    v_h: torch.Tensor,
    cost_temperature: float,
    reward_temperature: float,
    cost_ub: float,
    reward_ub: float = 100.0,
) -> torch.Tensor:
    unsafe = (v_h > 0).to(v_h.dtype)
    safe = ((v_h <= 0) & (q_h <= 0)).to(v_h.dtype)

    cost_w = torch.exp(
        torch.clamp((v_h - q_h) * cost_temperature, max=math.log(cost_ub))
    )
    reward_w = torch.exp(
        torch.clamp((q_r - v_r) * reward_temperature, max=math.log(reward_ub))
    )

    return unsafe * cost_w + safe * reward_w


@dataclass
class FISORConfig:
    discount: float = 0.99
    tau: float = 0.005
    critic_hidden_dims: Sequence[int] = (256, 256)
    num_qs: int = 2
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    reward_expectile: float = 0.9  # critic_hyperparam
    cost_expectile: float = 0.9  # cost_critic_hyperparam
    actor_lr: float = 3e-4
    decay_steps: Optional[int] = int(3e6)
    actor_weight_decay: float = 0.0
    actor_tau: float = 0.001
    actor_num_blocks: int = 3
    actor_dropout_rate: float = 0.1
    actor_layer_norm: bool = True
    time_dim: int = 64
    T: int = 5
    N: int = 16  # candidate actions at inference
    M: int = 0  # repeated last denoising steps
    ddpm_temperature: float = 1.0
    clip_sampler: bool = True
    cost_temperature: float = 5.0
    reward_temperature: float = 3.0
    cost_ub: float = 150.0
    batch_size: int = 2048  # split in two for the actor
    critic_batch_size: int = 256


class FISORAgent:
    reward_critic: RewardCritic
    feasibility_critic: FeasibilityCritic
    actor: DiffusionActor
    generator: Optional[torch.Generator]
    device: torch.device
    step: int
    config: FISORConfig
    obs_dim: int
    act_dim: int

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: FISORConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        feasibility_critic=None,
        train_feasibility: bool = True,
    ) -> None:
        """
        feasibility_critic: an already trained critic to use instead of building a new
            FeasibilityCritic (e.g. an EnsembleFeasibilityCritic trained with synthetic
            data). Anything with q_h / v_h / update / state_dict works.
        train_feasibility: False keeps the feasibility critic frozen during policy
            training, so the synthetic-data information is not washed out by
            further updates on real data only.
        """
        self.device = torch.device(device)
        self.generator = torch.Generator(self.device)

        self.reward_critic = RewardCritic(
            obs_dim,
            act_dim,
            config.critic_hidden_dims,
            num_qs=config.num_qs,
            discount=config.discount,
            tau=config.tau,
            expectile=config.reward_expectile,
            critic_lr=config.critic_lr,
            value_lr=config.value_lr,
            device=device,
            critic_batch_size=config.critic_batch_size,
        )
        self.train_feasibility = train_feasibility
        self.feasibility_critic = feasibility_critic or FeasibilityCritic(
            obs_dim,
            act_dim,
            config.critic_hidden_dims,
            num_qs=config.num_qs,
            discount=config.discount,
            tau=config.tau,
            expectile=config.cost_expectile,
            critic_lr=config.critic_lr,
            value_lr=config.value_lr,
            device=device,
            critic_batch_size=config.critic_batch_size,
        )
        self.actor = DiffusionActor(
            obs_dim,
            act_dim,
            DiffusionSchedule(config.T, device),
            model_kwargs={
                "time_dim": config.time_dim,
                "num_blocks": config.actor_num_blocks,
                "dropout_rate": config.actor_dropout_rate,
                "use_layer_norm": config.actor_layer_norm,
            },
            weight_decay=config.actor_weight_decay,
            actor_tau=config.actor_tau,
            device=device,
            lr=config.actor_lr,
            decay_steps=config.decay_steps,
        )

        self.step = 0
        self.config = config
        self.act_dim = act_dim
        self.obs_dim = obs_dim

    @torch.no_grad()
    def compute_actor_weights(self, batch: Batch) -> torch.Tensor:
        obs, act = batch["observations"], batch["actions"]
        q_r = self.reward_critic.q_r(obs, act)
        v_r = self.reward_critic.v_r(obs)

        q_h = self.feasibility_critic.q_h(obs, act)
        v_h = self.feasibility_critic.v_h(obs)

        return feasibility_weights(
            q_r,
            v_r,
            q_h,
            v_h,
            self.config.cost_temperature,
            self.config.reward_temperature,
            self.config.cost_ub,
        )

    @staticmethod
    def _rows(batch: Batch, start: int, stop: Optional[int]) -> Batch:
        # bootstrap_weights is (K, B): slice the batch dimension, not the members.
        return {
            k: (v[:, start:stop] if k == "bootstrap_weights" else v[start:stop])
            for k, v in batch.items()
        }

    def update(self, batch: Batch) -> Dict[str, float]:
        half = self.config.batch_size // 2

        first_half = self._rows(batch, 0, half)
        second_half = self._rows(batch, half, None)

        weight1, weight2 = (
            self.compute_actor_weights(first_half),
            self.compute_actor_weights(second_half),
        )

        info1 = self.actor.update(first_half, weight1, self.generator)
        info2 = self.actor.update(second_half, weight2, self.generator)

        mini_batch = self._rows(batch, 0, self.config.critic_batch_size)

        info3 = self.reward_critic.update(mini_batch)
        info4 = (
            self.feasibility_critic.update(mini_batch) if self.train_feasibility else {}
        )

        self.step += 1

        return {
            **{"actor1/" + k: v for k, v in info1.items()},
            **{"actor2/" + k: v for k, v in info2.items()},
            **{"reward/" + k: v for k, v in info3.items()},
            **{"feas/" + k: v for k, v in info4.items()},
        }

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        single = obs_t.dim() == 1
        if single:
            obs_t = obs_t.unsqueeze(0)

        batch_size, num_candidates = obs_t.shape[0], self.config.N

        obs_rep = obs_t.repeat_interleave(num_candidates, dim=0)

        actions = self.actor.sample(
            obs_t,
            num_samples=self.config.N,
            use_target=True,
            temperature=self.config.ddpm_temperature,
            repeat_last_step=self.config.M,
            clip_sampler=self.config.clip_sampler,
        )
        actions = actions.reshape((batch_size * self.config.N, self.act_dim))

        q_h = self.feasibility_critic.q_h(obs_rep, actions, use_target=True)
        q_h = q_h.view(batch_size, num_candidates)

        best = q_h.argmin(dim=1)
        rows = torch.arange(batch_size, device=actions.device)
        chosen = actions.view(batch_size, num_candidates, self.act_dim)[rows, best]

        chosen = chosen.cpu().numpy()
        return chosen[0] if single else chosen

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        rng_state = {
            "torch_cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "generator": self.generator.get_state()
            if self.generator is not None
            else None,
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()

        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "config": dataclasses.asdict(self.config),
            "dims": {"obs_dim": self.obs_dim, "act_dim": self.act_dim},
            "step": self.step,
            "reward_critic": self.reward_critic.state_dict(),
            "feasibility_critic": self.feasibility_critic.state_dict(),
            "actor": self.actor.state_dict(),
            "rng": rng_state,
        }

        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)

    def load(self, path: str, restore_rng: bool = True) -> None:
        """Restore a checkpoint written by save() into this (already constructed)
        agent. Architecture fields must match; other hyperparameter differences
        only produce a warning, so you can fine-tune with changed settings."""
        # map_location='cpu': RNG states must stay on CPU; load_state_dict copies
        # parameters and optimiser states onto the devices of the existing modules.
        # weights_only=False is needed for the numpy RNG state (torch >= 2.6 defaults to True).
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        if ckpt.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version {ckpt.get('version')} in {path}"
            )

        self._check_compatible(ckpt, path)

        self.reward_critic.load_state_dict(ckpt["reward_critic"])
        self.feasibility_critic.load_state_dict(ckpt["feasibility_critic"])
        self.actor.load_state_dict(ckpt["actor"])
        self.step = ckpt["step"]

        if restore_rng:
            rng = ckpt["rng"]
            torch.set_rng_state(rng["torch_cpu"])
            np.random.set_state(rng["numpy"])
            if self.generator is not None and rng.get("generator") is not None:
                self.generator.set_state(rng["generator"])
            if torch.cuda.is_available() and "torch_cuda" in rng:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])

    def load_critics_only(self, path: str) -> None:
        """Load just the two critics, e.g. for feasibility-map evaluation.
        Optimiser states are restored too, so the critics can still be fine-tuned."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._check_compatible(ckpt, path)
        self.reward_critic.load_state_dict(ckpt["reward_critic"])
        self.feasibility_critic.load_state_dict(ckpt["feasibility_critic"])

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str = "cpu", seed: Optional[int] = None
    ) -> "FISORAgent":
        """Build an agent with the checkpoint's own config and dimensions, then load it."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config = FISORConfig(**ckpt["config"])  # noqa: F821 (defined in your agent module)
        agent = cls(
            ckpt["dims"]["obs_dim"],
            ckpt["dims"]["act_dim"],
            config,
            device=device,
        )
        agent.load(path)
        return agent

    def _check_compatible(self, ckpt: dict, path: str) -> None:
        dims = ckpt["dims"]
        if (dims["obs_dim"], dims["act_dim"]) != (self.obs_dim, self.act_dim):
            raise ValueError(
                f"{path}: checkpoint dims (obs={dims['obs_dim']}, act={dims['act_dim']}) "
                f"do not match agent (obs={self.obs_dim}, act={self.act_dim})"
            )

        saved = ckpt["config"]
        current = dataclasses.asdict(self.config)

        arch_mismatch = {
            k: (saved.get(k), current.get(k))
            for k in ARCH_KEYS
            if _normalise(saved.get(k)) != _normalise(current.get(k))
        }
        if arch_mismatch:
            raise ValueError(
                f"{path}: architecture mismatch (saved, current): {arch_mismatch}"
            )

        other_mismatch = {
            k: (saved.get(k), v)
            for k, v in current.items()
            if k not in ARCH_KEYS and _normalise(saved.get(k)) != _normalise(v)
        }
        if other_mismatch:
            warnings.warn(
                f"{path}: hyperparameters differ from checkpoint (saved, current): "
                f"{other_mismatch}"
            )
