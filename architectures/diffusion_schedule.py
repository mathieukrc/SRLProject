from typing import Optional

import torch


def vp_beta_schedule(T: int) -> torch.Tensor:
    t = torch.arange(1, T + 1)
    b_max = 10.0
    b_min = 0.1
    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
    betas = 1 - alpha
    return betas


class DiffusionSchedule:
    T: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_hats: torch.Tensor
    device: torch.device

    def __init__(
        self,
        T: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = torch.device(device)

        self.T = T
        self.betas = vp_beta_schedule(self.T).to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=-1)

    def sample_timesteps(
        self, batch_size: int, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        return torch.randint(
            0, self.T, (batch_size,), generator=generator, device=self.device
        )

    def q_sample(
        self, actions: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha_hat_t = self.alpha_hats[t].unsqueeze(-1)
        return torch.sqrt(alpha_hat_t) * actions + torch.sqrt(1 - alpha_hat_t) * noise
