from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .diffusion_schedule import DiffusionSchedule
from .fourier_features import FourierFeatures
from .mlp import MLP
from .resnet import MLPResNet


class DDPMNoiseModel(nn.Module):
    time_preprocess: FourierFeatures
    cond_encoder: MLP
    reverse_encoder: MLPResNet

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        time_dim: int = 64,
        cond_hidden_dims: Sequence[int] = (128, 128),
        num_blocks: int = 3,
        hidden_dim: int = 256,
        dropout_rate: Optional[float] = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.time_preprocess = FourierFeatures(
            in_dim=1, output_size=time_dim, learnable=True
        )
        self.cond_encoder = MLP(
            time_dim, cond_hidden_dims[-1], hidden_dims=cond_hidden_dims
        )
        self.reverse_encoder = MLPResNet(
            obs_dim + act_dim + cond_hidden_dims[-1],
            act_dim,
            num_blocks=num_blocks,
            hidden_dims=hidden_dim,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
        )

    def forward(
        self, obs: torch.Tensor, noisy_actions: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        B = noisy_actions.shape[0]
        t = t.float()

        if t.dim() == 0:
            t = t.expand(B)

        t_ff = self.time_preprocess(t.reshape(B, -1))
        cond = self.cond_encoder(t_ff)
        reverse_input = torch.cat((noisy_actions, obs, cond), dim=-1)

        return self.reverse_encoder(reverse_input)


@torch.no_grad()
def ddpm_sample(
    model: DDPMNoiseModel,
    schedule: DiffusionSchedule,
    obs: torch.Tensor,
    act_dim: int,
    temperature: float = 1.0,
    repeat_last_step: int = 0,
    clip_sampler: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    batch_size, device = obs.shape[0], obs.device
    was_training = model.training
    model.eval()

    def step(x: torch.Tensor, t: int) -> torch.Tensor:
        t_in = torch.full((batch_size, 1), float(t), device=device)
        eps_pred = model.forward(obs, x, t_in)

        alpha_1 = 1.0 / torch.sqrt(schedule.alphas[t])
        alpha_2 = (1.0 - schedule.alphas[t]) / (
            torch.sqrt(1.0 - schedule.alpha_hats[t])
        )
        x = alpha_1 * (x - alpha_2 * eps_pred)

        if t > 0:
            z = torch.randn(x.shape, generator=generator, device=device) * temperature
            x = x + torch.sqrt(schedule.betas[t]) * z

        return x.clamp(-1.0, 1.0) if clip_sampler else x

    x = torch.randn((batch_size, act_dim), generator=generator, device=device)
    for t in range(schedule.T - 1, -1, -1):
        x = step(x, t)
    for _ in range(repeat_last_step):
        x = step(x, 0)

    model.train(was_training)
    return x.clamp(-1.0, 1.0)
