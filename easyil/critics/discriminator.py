"""Discriminator for GAIL."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from easyil.networks import MLPBackbone


@dataclass
class DiscriminatorConfig:
    """Configuration for GAIL Discriminator."""

    obs_dim: int
    act_dim: int
    hidden: int = 256
    depth: int = 2


class Discriminator(nn.Module):
    """
    Discriminator for GAIL.

    Classifies (s, a) pairs as expert (high) or policy (low).
    """

    def __init__(self, cfg: DiscriminatorConfig) -> None:
        super().__init__()
        self.cfg = cfg

        in_dim = cfg.obs_dim + cfg.act_dim

        self.net = MLPBackbone(
            in_dim=in_dim,
            out_dim=1,
            hidden=cfg.hidden,
            depth=cfg.depth,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator logits.

        Args:
            obs: (B, obs_dim)
            action: (B, act_dim)

        Returns:
            logits: (B, 1) raw logits (before sigmoid)
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)

    def predict_proba(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict probability of being expert."""
        logits = self.forward(obs, action)
        return torch.sigmoid(logits)

    def reward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute GAIL reward: -log(1 - D(s,a)).

        Higher reward when discriminator thinks it's expert.
        """
        logits = self.forward(obs, action)
        # Reward: -log(1 - sigmoid(logits)) = logits + softplus(-logits)
        return -torch.nn.functional.logsigmoid(-logits)

    def reward_airl(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        AIRL-style reward: log(D) - log(1-D) = logits.

        More stable gradient flow.
        """
        return self.forward(obs, action)
