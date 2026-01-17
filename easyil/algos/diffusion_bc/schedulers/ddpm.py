"""DDPM scheduler implementation (JAX)."""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from easyil.algos.diffusion_bc.schedulers.base import BaseScheduler


@dataclass
class DDPMScheduler(BaseScheduler):
    """DDPM (Denoising Diffusion Probabilistic Models) scheduler."""

    # Additional precomputed tensors
    posterior_variance: jnp.ndarray = field(init=False)
    posterior_log_variance_clipped: jnp.ndarray = field(init=False)
    posterior_mean_coef1: jnp.ndarray = field(init=False)
    posterior_mean_coef2: jnp.ndarray = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.posterior_variance = self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        self.posterior_log_variance_clipped = jnp.log(jnp.clip(self.posterior_variance, a_min=1e-20))
        self.posterior_mean_coef1 = self.betas * jnp.sqrt(self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        self.posterior_mean_coef2 = (1.0 - self.alpha_bars_prev) * jnp.sqrt(self.alphas) / (1.0 - self.alpha_bars)

    def step(
        self,
        xt: jnp.ndarray,
        t: int,
        t_prev: int,
        eps: jnp.ndarray,
        rng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        """One ancestral DDPM step: x_t -> x_{t-1}."""
        t_idx = jnp.array([t], dtype=jnp.int32)

        coef1 = self.posterior_mean_coef1[t_idx].reshape(1, 1, 1)
        coef2 = self.posterior_mean_coef2[t_idx].reshape(1, 1, 1)

        x0 = self.predict_x0_from_eps(xt, t_idx, eps)
        mean = coef1 * x0 + coef2 * xt

        if t == 0:
            return mean

        var = self.posterior_variance[t_idx].reshape(1, 1, 1)
        noise = jax.random.normal(rng_key, xt.shape)
        return mean + jnp.sqrt(var) * noise
