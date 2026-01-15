"""DDPM scheduler implementation."""
from __future__ import annotations

import torch

from easyil.algos.diffusion_bc.schedulers.base import BaseScheduler


class DDPMScheduler(BaseScheduler):
    """DDPM (Denoising Diffusion Probabilistic Models) scheduler."""

    def __init__(
        self,
        train_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__(train_steps, beta_start, beta_end)

        self.posterior_variance = self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        self.posterior_mean_coef2 = (1.0 - self.alpha_bars_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bars)

    def step(
        self,
        xt: torch.Tensor,
        t: int,
        t_prev: int,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """One ancestral DDPM step: x_t -> x_{t-1}."""
        t_idx = torch.tensor([t], device=xt.device, dtype=torch.long)

        coef1 = self.posterior_mean_coef1[t_idx].view(1, 1, 1)
        coef2 = self.posterior_mean_coef2[t_idx].view(1, 1, 1)

        x0 = self.predict_x0_from_eps(xt, t_idx, eps)
        mean = coef1 * x0 + coef2 * xt

        if t == 0:
            return mean

        var = self.posterior_variance[t_idx].view(1, 1, 1)
        noise = torch.randn_like(xt)
        return mean + torch.sqrt(var) * noise
