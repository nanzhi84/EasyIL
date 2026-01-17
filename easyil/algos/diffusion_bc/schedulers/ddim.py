"""DDIM scheduler implementation (JAX)."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from easyil.algos.diffusion_bc.schedulers.base import BaseScheduler


@dataclass
class DDIMScheduler(BaseScheduler):
    """DDIM (Denoising Diffusion Implicit Models) scheduler."""

    eta: float = 0.0

    def step(
        self,
        xt: jnp.ndarray,
        t: int,
        t_prev: int,
        eps: jnp.ndarray,
        rng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        """One DDIM step (optionally stochastic if eta>0)."""
        t_t = jnp.array([t], dtype=jnp.int32)
        t_prev_t = jnp.array([t_prev], dtype=jnp.int32)

        x0 = self.predict_x0_from_eps(xt, t_t, eps)

        ab_t = self.alpha_bars[t_t].reshape(1, 1, 1)
        ab_prev = jnp.where(
            t_prev >= 0,
            self.alpha_bars[t_prev_t].reshape(1, 1, 1),
            jnp.ones_like(ab_t),
        )

        sigma = self.eta * jnp.sqrt((1 - ab_prev) / (1 - ab_t)) * jnp.sqrt(
            jnp.clip(1 - ab_t / ab_prev, a_min=0.0)
        )

        dir_xt = jnp.sqrt(jnp.clip(1 - ab_prev - sigma**2, a_min=0.0)) * eps
        x_prev = jnp.sqrt(ab_prev) * x0 + dir_xt

        if self.eta > 0 and t > 0:
            noise = jax.random.normal(rng_key, xt.shape)
            x_prev = x_prev + sigma * noise

        return x_prev
