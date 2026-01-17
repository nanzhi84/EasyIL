"""Base scheduler for diffusion models (JAX)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import jax.numpy as jnp


@dataclass
class BaseScheduler(ABC):
    """Abstract base class for diffusion schedulers."""

    train_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # Precomputed tensors (initialized in __post_init__)
    betas: jnp.ndarray = field(init=False)
    alphas: jnp.ndarray = field(init=False)
    alpha_bars: jnp.ndarray = field(init=False)
    sqrt_alpha_bars: jnp.ndarray = field(init=False)
    sqrt_one_minus_alpha_bars: jnp.ndarray = field(init=False)
    alpha_bars_prev: jnp.ndarray = field(init=False)
    _infer_timesteps: Optional[jnp.ndarray] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.T = int(self.train_steps)

        betas = jnp.linspace(float(self.beta_start), float(self.beta_end), self.T, dtype=jnp.float32)
        alphas = 1.0 - betas
        alpha_bars = jnp.cumprod(alphas)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = jnp.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = jnp.sqrt(1.0 - alpha_bars)

        alpha_bars_prev = jnp.concatenate([jnp.array([1.0], dtype=jnp.float32), alpha_bars[:-1]])
        self.alpha_bars_prev = alpha_bars_prev

        self._infer_timesteps = None

    @property
    def infer_timesteps(self) -> jnp.ndarray:
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
        self._infer_timesteps = jnp.array(steps, dtype=jnp.int32)

    def q_sample(self, x0: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray) -> jnp.ndarray:
        """Forward diffusion: x_t = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*noise."""
        sqrt_ab = self.sqrt_alpha_bars[t].reshape(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_omb * noise

    def predict_x0_from_eps(self, xt: jnp.ndarray, t: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
        sqrt_ab = self.sqrt_alpha_bars[t].reshape(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1)
        return (xt - sqrt_omb * eps) / jnp.clip(sqrt_ab, a_min=1e-8)

    @abstractmethod
    def step(
        self,
        xt: jnp.ndarray,
        t: int,
        t_prev: int,
        eps: jnp.ndarray,
        rng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        """One denoising step: x_t -> x_{t_prev}."""
        ...
