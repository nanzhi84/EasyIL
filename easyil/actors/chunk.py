"""Chunk-based actor for BC-style action prediction."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from easyil.networks import MLPBackbone


@dataclass
class ChunkActorConfig:
    """Configuration for ChunkActor."""

    obs_dim: int
    act_dim: int
    obs_horizon: int = 1
    action_horizon: int = 1
    hidden: int = 512
    depth: int = 4
    dropout: float = 0.0
    action_clip: float = 1.0


class ChunkActor(nn.Module):
    """
    MLP-based actor that predicts action chunks from observation history.

    Used for Behavior Cloning with temporal action prediction.
    """

    def __init__(self, cfg: ChunkActorConfig) -> None:
        super().__init__()
        self.cfg = cfg

        in_dim = cfg.obs_dim * cfg.obs_horizon
        out_dim = cfg.act_dim * cfg.action_horizon

        self.backbone = MLPBackbone(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=cfg.hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )

    @property
    def action_dim(self) -> int:
        return self.cfg.act_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            obs: (B, obs_horizon, obs_dim) or (B, obs_horizon * obs_dim)

        Returns:
            (B, action_horizon, act_dim) action chunk
        """
        B = obs.shape[0]
        x = obs.reshape(B, -1)
        out = self.backbone(x)
        actions = out.view(B, self.cfg.action_horizon, self.cfg.act_dim)

        if self.cfg.action_clip > 0:
            actions = torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip)

        return actions

    def predict_chunk(self, obs: torch.Tensor) -> torch.Tensor:
        """Alias for forward, returns action chunk."""
        return self.forward(obs)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action (first action of chunk)."""
        chunk = self.forward(obs)
        return chunk[:, 0, :]
