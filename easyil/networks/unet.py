"""1D UNet backbone for sequence modeling (JAX/Flax)."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


class ResBlock1D(nn.Module):
    """1D Residual block with embedding conditioning."""

    out_ch: int
    emb_dim: int
    groups: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray) -> jnp.ndarray:
        in_ch = x.shape[-1]
        out_ch = self.out_ch
        groups = min(self.groups, in_ch)
        out_groups = min(self.groups, out_ch)

        # First conv
        h = nn.GroupNorm(num_groups=groups)(x)
        h = nn.silu(h)
        h = nn.Conv(features=out_ch, kernel_size=(3,), padding="SAME")(h)

        # Add embedding
        emb_proj = nn.Dense(features=out_ch)(emb)
        h = h + emb_proj[:, None, :]

        # Second conv
        h = nn.GroupNorm(num_groups=out_groups)(h)
        h = nn.silu(h)
        h = nn.Conv(features=out_ch, kernel_size=(3,), padding="SAME")(h)

        # Skip connection
        if in_ch != out_ch:
            x = nn.Conv(features=out_ch, kernel_size=(1,))(x)

        return h + x


class Downsample1D(nn.Module):
    """1D downsampling via strided convolution."""

    out_ch: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Conv(features=self.out_ch, kernel_size=(4,), strides=(2,), padding="SAME")(x)


class Upsample1D(nn.Module):
    """1D upsampling via transposed convolution."""

    out_ch: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.ConvTranspose(features=self.out_ch, kernel_size=(4,), strides=(2,), padding="SAME")(x)


class UNet1DBackbone(nn.Module):
    """A compact 1D U-Net backbone for sequence modeling.

    Treats sequence length as the 1D spatial axis and feature dimension as channels.
    Designed to be used with conditioning embeddings (e.g., time + observation).
    """

    in_channels: int
    out_channels: int
    emb_dim: int = 256
    base_channels: int = 128
    channel_mults: Sequence[int] = (1, 2, 2)
    groups: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, seq_len, in_channels).
            emb: Conditioning embedding of shape (B, emb_dim).

        Returns:
            Output tensor of shape (B, seq_len, out_channels).
        """
        channel_mults = list(self.channel_mults)

        # Initial projection
        x = nn.Conv(features=self.base_channels, kernel_size=(3,), padding="SAME")(x)

        # Down path
        skips = []
        cur_ch = self.base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = self.base_channels * mult
            x = ResBlock1D(out_ch=out_ch, emb_dim=self.emb_dim, groups=self.groups)(x, emb)
            skips.append(x)
            x = Downsample1D(out_ch=out_ch)(x)
            cur_ch = out_ch

        # Mid
        x = ResBlock1D(out_ch=cur_ch, emb_dim=self.emb_dim, groups=self.groups)(x, emb)
        x = ResBlock1D(out_ch=cur_ch, emb_dim=self.emb_dim, groups=self.groups)(x, emb)

        # Up path
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = self.base_channels * mult
            x = Upsample1D(out_ch=out_ch)(x)
            skip = skips.pop()
            # Handle size mismatch
            min_len = min(x.shape[1], skip.shape[1])
            x = x[:, :min_len, :]
            skip = skip[:, :min_len, :]
            x = jnp.concatenate([x, skip], axis=-1)
            x = ResBlock1D(out_ch=out_ch, emb_dim=self.emb_dim, groups=self.groups)(x, emb)
            cur_ch = out_ch

        # Output projection
        x = nn.GroupNorm(num_groups=min(self.groups, cur_ch))(x)
        x = nn.silu(x)
        x = nn.Conv(features=self.out_channels, kernel_size=(3,), padding="SAME")(x)

        return x
