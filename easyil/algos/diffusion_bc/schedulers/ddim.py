"""DDIM scheduler implementation."""
from __future__ import annotations

import torch

from easyil.algos.diffusion_bc.schedulers.base import BaseScheduler


class DDIMScheduler(BaseScheduler):
    """DDIM (Denoising Diffusion Implicit Models) scheduler."""

    def __init__(
        self,
        train_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        eta: float = 0.0,
    ) -> None:
        super().__init__(train_steps, beta_start, beta_end)
        self.eta = float(eta)

    def step(
        self,
        xt: torch.Tensor,
        t: int,
        t_prev: int,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """One DDIM step (optionally stochastic if eta>0)."""
        device = xt.device
        t_t = torch.tensor([t], device=device, dtype=torch.long)
        t_prev_t = torch.tensor([t_prev], device=device, dtype=torch.long)

        x0 = self.predict_x0_from_eps(xt, t_t, eps)

        ab_t = self.alpha_bars[t_t].view(1, 1, 1)
        ab_prev = self.alpha_bars[t_prev_t].view(1, 1, 1) if t_prev >= 0 else torch.ones_like(ab_t)

        sigma = self.eta * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(
            torch.clamp(1 - ab_t / ab_prev, min=0.0)
        )

        dir_xt = torch.sqrt(torch.clamp(1 - ab_prev - sigma**2, min=0.0)) * eps
        x_prev = torch.sqrt(ab_prev) * x0 + dir_xt

        if self.eta > 0 and t > 0:
            noise = torch.randn_like(xt)
            x_prev = x_prev + sigma * noise

        return x_prev
