"""MLP backbone for neural networks (JAX/Flax)."""
from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import flax.linen as nn


class MLPBackbone(nn.Module):
    """A simple MLP backbone with configurable depth and width."""

    in_dim: int
    out_dim: int
    hidden: int = 512
    depth: int = 4
    dropout: float = 0.0
    activation: str = "silu"

    def _get_activation(self):
        act_map = {"silu": nn.silu, "relu": nn.relu, "gelu": nn.gelu}
        return act_map.get(self.activation.lower(), nn.silu)

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_dim).
            training: Whether in training mode (for dropout).

        Returns:
            Output tensor of shape (B, out_dim).
        """
        act_fn = self._get_activation()
        d = self.in_dim

        for i in range(self.depth - 1):
            x = nn.Dense(features=self.hidden)(x)
            x = act_fn(x)
            if self.dropout > 0:
                x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)

        x = nn.Dense(features=self.out_dim)(x)
        return x
