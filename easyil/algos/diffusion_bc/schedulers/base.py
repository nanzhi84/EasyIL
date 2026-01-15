"""Base scheduler for diffusion models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class BaseScheduler(ABC):
    """Abstract base class for diffusion schedulers."""

    def __init__(
        self,
        train_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        self.T = int(train_steps)

        betas = torch.linspace(float(beta_start), float(beta_end), self.T, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

        alpha_bars_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alpha_bars[:-1]])
        self.alpha_bars_prev = alpha_bars_prev

        self._infer_timesteps: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "BaseScheduler":
        for name, val in list(self.__dict__.items()):
            if isinstance(val, torch.Tensor):
                setattr(self, name, val.to(device))
        return self

    @property
    def infer_timesteps(self) -> torch.Tensor:
        if self._infer_timesteps is None:
            self.set_inference_steps(self.T)
        return self._infer_timesteps

    def set_inference_steps(self, num_inference_steps: int) -> None:
        K = int(num_inference_steps)
        if K <= 0:
            raise ValueError("num_inference_steps must be > 0")
        steps = np.linspace(0, self.T - 1, K, dtype=np.float64)
        steps = np.round(steps).astype(np.int64)
        steps = np.unique(steps)
        steps = steps[::-1].copy()
        self._infer_timesteps = torch.from_numpy(steps).long()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*noise."""
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_omb * noise

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return (xt - sqrt_omb * eps) / torch.clamp(sqrt_ab, min=1e-8)

    @abstractmethod
    def step(
        self,
        xt: torch.Tensor,
        t: int,
        t_prev: int,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """One denoising step: x_t -> x_{t_prev}."""
        ...
