"""Noise predictor networks for Diffusion BC."""
from __future__ import annotations

import inspect
from typing import Callable, Dict, List, Optional

import torch
from omegaconf import DictConfig
from torch import nn

from easyil.networks import ObsEncoder, TimestepEmbedding, UNet1DBackbone

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    def decorator(cls: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
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

    sig = inspect.signature(model_cls.__init__)
    valid_params = {p for p in sig.parameters if p != "self"}
    model_kwargs = {k: v for k, v in dict(cfg).items() if k != "type" and k in valid_params}
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

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        obs_horizon: int,
        action_horizon: int,
        time_emb_dim: int = 64,
        emb_dim: int = 256,
        hidden: int = 512,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.act_dim = int(act_dim)
        self.action_horizon = int(action_horizon)
        self.in_dim = self.act_dim * self.action_horizon

        self.time_embed = TimestepEmbedding(time_emb_dim, emb_dim)
        self.obs_embed = ObsEncoder(obs_dim, obs_horizon, emb_dim, hidden=emb_dim)

        layers: List[nn.Module] = []
        d = self.in_dim + emb_dim
        for _ in range(int(depth) - 1):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, self.in_dim))
        self.mlp = nn.Sequential(*layers)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.obs_embed(obs)

    def forward(
        self, xt: torch.Tensor, t: torch.Tensor, obs: torch.Tensor, obs_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b = xt.shape[0]
        x = xt.reshape(b, -1)
        oemb = obs_emb if obs_emb is not None else self.obs_embed(obs)
        emb = self.time_embed(t) + oemb
        h = torch.cat([x, emb], dim=1)
        out = self.mlp(h)
        return out.view(b, self.action_horizon, self.act_dim)


@register_model("unet")
class UNet1DNoisePredictor(nn.Module):
    """UNet-based noise predictor for diffusion."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        obs_horizon: int,
        action_horizon: int,
        time_emb_dim: int = 64,
        emb_dim: int = 256,
        base_channels: int = 128,
        channel_mults: Optional[List[int]] = None,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.act_dim = int(act_dim)
        self.action_horizon = int(action_horizon)

        if channel_mults is None:
            channel_mults = [1, 2, 2]

        self.time_embed = TimestepEmbedding(time_emb_dim, emb_dim)
        self.obs_embed = ObsEncoder(obs_dim, obs_horizon, emb_dim, hidden=emb_dim)

        self.backbone = UNet1DBackbone(
            in_channels=act_dim,
            out_channels=act_dim,
            emb_dim=emb_dim,
            base_channels=base_channels,
            channel_mults=channel_mults,
            groups=groups,
        )

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.obs_embed(obs)

    def forward(
        self, xt: torch.Tensor, t: torch.Tensor, obs: torch.Tensor, obs_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        oemb = obs_emb if obs_emb is not None else self.obs_embed(obs)
        emb = self.time_embed(t) + oemb

        x = xt.transpose(1, 2)  # (B, act_dim, Ta)
        out = self.backbone(x, emb)
        return out.transpose(1, 2)  # (B, Ta, act_dim)
