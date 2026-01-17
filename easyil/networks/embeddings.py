"""Common embedding layers for neural networks (JAX/Flax)."""
from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by MLP projection."""

    dim: int
    hidden_dim: int

    @staticmethod
    def sinusoidal_embedding(timesteps: jnp.ndarray, dim: int) -> jnp.ndarray:
        """Create sinusoidal timestep embeddings. (B,) -> (B, dim)"""
        half = dim // 2
        freqs = jnp.exp(-math.log(10000) * jnp.arange(0, half, dtype=jnp.float32) / half)
        args = timesteps.astype(jnp.float32)[:, None] * freqs[None, :]
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=1)
        if dim % 2 == 1:
            emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=1)
        return emb

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        x = self.sinusoidal_embedding(t, self.dim)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.silu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        return x


class ObsEncoder(nn.Module):
    """Observation encoder that flattens and projects observation history."""

    obs_dim: int
    obs_horizon: int
    emb_dim: int
    hidden: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode observation history.

        Args:
            obs: Observation tensor of shape (B, obs_horizon, obs_dim).

        Returns:
            Embedding of shape (B, emb_dim).
        """
        b = obs.shape[0]
        x = obs.reshape(b, -1)
        x = nn.Dense(features=self.hidden)(x)
        x = nn.silu(x)
        x = nn.Dense(features=self.emb_dim)(x)
        return x
