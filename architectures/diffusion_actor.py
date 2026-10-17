import copy
import math
from typing import Dict, Optional, Tuple

import torch

from .ddpm import DDPMNoiseModel, ddpm_sample
from .diffusion_schedule import DiffusionSchedule


class DiffusionActor:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        schedule: DiffusionSchedule,
        model_kwargs: Optional[dict] = None,
        lr: float = 3e-4,
        decay_steps: Optional[int] = int(3e6),
        weight_decay: float = 0.0,
        actor_tau: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.schedule = schedule
        self.actor_tau = actor_tau
        self.device = torch.device(device)

        self.model = DDPMNoiseModel(obs_dim, act_dim, **(model_kwargs or {})).to(
            self.device
        )

        self.target_model = copy.deepcopy(self.model)
        self.target_model.requires_grad_(False)
        self.target_model.eval()

        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            (no_decay if name.endswith("bias") else decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        if decay_steps is not None:

            def cosine(step: int) -> float:
                s = min(step, decay_steps)
                return 0.5 * (1.0 + math.cos(math.pi * s / decay_steps))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, cosine)
        else:
            self.scheduler = None

    def loss(
        self,
        batch: dict,
        weights: torch.Tensor,
 w        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs, actions = batch["observations"], batch["actions"]
        B = actions.shape[0]

        t = self.schedule.sample_timesteps(B, generator)  # (B,) long
        noise = torch.randn(
            actions.shape,
            device=actions.device,
            dtype=actions.dtype,
            generator=generator,
        )
        noisy_actions = self.schedule.q_sample(actions, t, noise)

        eps_pred = self.model(obs, noisy_actions, t.to(actions.dtype).unsqueeze(1))

        per_sample = ((eps_pred - noise) ** 2).sum(dim=-1)  # sum over action dims
        loss = (per_sample * weights.detach()).mean()
        return loss, {"actor_loss": loss.item(), "weights": weights.mean().item()}

    def update(
        self,
        batch: dict,
        weights: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, float]:
        self.model.train()  # dropout on
        loss, info = self.loss(batch, weights, generator)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self._ema_update()
        info["actor_lr"] = self.optimizer.param_groups[0]["lr"]
        return info

    @torch.no_grad()
    def _ema_update(self) -> None:
        for p_t, p in zip(self.target_model.parameters(), self.model.parameters()):
            p_t.lerp_(p, self.actor_tau)
        for b_t, b in zip(self.target_model.buffers(), self.model.buffers()):
            b_t.copy_(b)

    @torch.no_grad()
    def sample(
        self,
        obs: torch.Tensor,
        num_samples: int,
        use_target: bool = True,
        temperature: float = 1.0,
        repeat_last_step: int = 0,
        clip_sampler: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """obs (obs_dim,) -> (N, act_dim);  obs (B, obs_dim) -> (B, N, act_dim)."""
        model = self.target_model if use_target else self.model
        was_training = model.training
        model.eval()  # dropout off

        single = obs.dim() == 1
        obs_b = obs.unsqueeze(0) if single else obs  # (B, obs_dim)
        B = obs_b.shape[0]
        obs_rep = obs_b.repeat_interleave(num_samples, dim=0)  # (B*N, obs_dim)

        actions = ddpm_sample(
            model,
            self.schedule,
            obs_rep,
            self.act_dim,
            temperature=temperature,
            repeat_last_step=repeat_last_step,
            clip_sampler=clip_sampler,
            generator=generator,
        )
        model.train(was_training)

        actions = actions.view(B, num_samples, self.act_dim)
        return actions[0] if single else actions

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.target_model.load_state_dict(state["target_model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
