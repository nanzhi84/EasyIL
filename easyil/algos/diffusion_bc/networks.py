"""Noise predictor networks for Diffusion BC (JAX/Flax)."""
from __future__ import annotations

import inspect
from typing import Callable, Dict, List, Optional, Sequence

import jax.numpy as jnp
import flax.linen as nn
from omegaconf import DictConfig

from easyil.networks import ObsEncoder, TimestepEmbedding, UNet1DBackbone

MODEL_REGISTRY: Dict[str, type] = {}
_RESERVED_MODEL_KEYS = {"obs_dim", "act_dim", "obs_horizon", "action_horizon"}


def register_model(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_noise_predictor(
    cfg: DictConfig,
    obs_dim: int,
    act_dim: int,
    obs_horizon: int,
    action_horizon: int,
) -> nn.Module:
    """Build noise predictor from config."""
    model_type = str(cfg.type).lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model.type: {model_type} (expected: {list(MODEL_REGISTRY.keys())})")
    model_cls = MODEL_REGISTRY[model_type]

    def get_model_fields(cls: type) -> List[str]:
        if hasattr(cls, "__dataclass_fields__"):
            return list(cls.__dataclass_fields__.keys())
        sig = inspect.signature(cls)
        return [name for name in sig.parameters if name != "self"]

    cfg_dict = dict(cfg)
    model_fields = set(get_model_fields(model_cls))
    allowed_keys = model_fields - _RESERVED_MODEL_KEYS

    unknown_keys = [k for k in cfg_dict.keys() if k != "type" and k not in allowed_keys]
    if unknown_keys:
        unknown_keys_sorted = sorted(unknown_keys)
        raise ValueError(
            f"Unknown model config keys for type '{model_type}': {unknown_keys_sorted}"
        )

    model_kwargs = {k: v for k, v in cfg_dict.items() if k in allowed_keys}
    return model_cls(
        obs_dim=obs_dim,
        act_dim=act_dim,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        **model_kwargs,
    )


@register_model("mlp")
class MLPNoisePredictor(nn.Module):
    """MLP-based noise predictor for diffusion."""

    obs_dim: int
    act_dim: int
    obs_horizon: int
    action_horizon: int
    time_emb_dim: int = 64
    emb_dim: int = 256
    hidden: int = 512
    depth: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        xt: jnp.ndarray,
        t: jnp.ndarray,
        obs: jnp.ndarray,
        obs_emb: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        b = xt.shape[0]
        in_dim = self.act_dim * self.action_horizon

        # Flatten action input
        x = xt.reshape(b, -1)

        # Embeddings
        t_emb = TimestepEmbedding(dim=self.time_emb_dim, hidden_dim=self.emb_dim)(t)
        if obs_emb is not None:
            o_emb = obs_emb
        else:
            o_emb = ObsEncoder(
                obs_dim=self.obs_dim,
                obs_horizon=self.obs_horizon,
                emb_dim=self.emb_dim,
                hidden=self.emb_dim,
            )(obs)

        emb = t_emb + o_emb

        # MLP
        h = jnp.concatenate([x, emb], axis=1)
        for _ in range(self.depth - 1):
            h = nn.Dense(features=self.hidden)(h)
            h = nn.silu(h)
            if self.dropout > 0:
                h = nn.Dropout(rate=self.dropout, deterministic=not training)(h)
        out = nn.Dense(features=in_dim)(h)

        return out.reshape(b, self.action_horizon, self.act_dim)

    def encode_obs(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Pre-encode observations for inference speedup."""
        return ObsEncoder(
            obs_dim=self.obs_dim,
            obs_horizon=self.obs_horizon,
            emb_dim=self.emb_dim,
            hidden=self.emb_dim,
        )(obs)


@register_model("unet")
class UNet1DNoisePredictor(nn.Module):
    """UNet-based noise predictor for diffusion."""

    obs_dim: int
    act_dim: int
    obs_horizon: int
    action_horizon: int
    time_emb_dim: int = 64
    emb_dim: int = 256
    base_channels: int = 128
    channel_mults: Sequence[int] = (1, 2, 2)
    groups: int = 8

    @nn.compact
    def __call__(
        self,
        xt: jnp.ndarray,
        t: jnp.ndarray,
        obs: jnp.ndarray,
        obs_emb: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        # Embeddings
        t_emb = TimestepEmbedding(dim=self.time_emb_dim, hidden_dim=self.emb_dim)(t)
        if obs_emb is not None:
            o_emb = obs_emb
        else:
            o_emb = ObsEncoder(
                obs_dim=self.obs_dim,
                obs_horizon=self.obs_horizon,
                emb_dim=self.emb_dim,
                hidden=self.emb_dim,
            )(obs)

        emb = t_emb + o_emb

        # UNet backbone - input is (B, seq_len, channels)
        out = UNet1DBackbone(
            in_channels=self.act_dim,
            out_channels=self.act_dim,
            emb_dim=self.emb_dim,
            base_channels=self.base_channels,
            channel_mults=self.channel_mults,
            groups=self.groups,
        )(xt, emb)

        return out

    def encode_obs(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Pre-encode observations for inference speedup."""
        return ObsEncoder(
            obs_dim=self.obs_dim,
            obs_horizon=self.obs_horizon,
            emb_dim=self.emb_dim,
            hidden=self.emb_dim,
        )(obs)
