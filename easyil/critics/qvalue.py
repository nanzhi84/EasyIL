"""Q-value critic for actor-critic algorithms."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from easyil.networks import MLPBackbone


@dataclass
class QCriticConfig:
    """Configuration for Q-Critic."""

    obs_dim: int
    act_dim: int
    hidden: int = 256
    depth: int = 2


class QCritic(nn.Module):
    """
    Q(s, a) critic network.

    Used for SAC, SQIL, and other Q-learning based algorithms.
    Implements twin Q-networks for reduced overestimation.
    """

    def __init__(self, cfg: QCriticConfig) -> None:
        super().__init__()
        self.cfg = cfg

        in_dim = cfg.obs_dim + cfg.act_dim

        self.q1 = MLPBackbone(
            in_dim=in_dim,
            out_dim=1,
            hidden=cfg.hidden,
            depth=cfg.depth,
        )
        self.q2 = MLPBackbone(
            in_dim=in_dim,
            out_dim=1,
            hidden=cfg.hidden,
            depth=cfg.depth,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values from both networks.

        Args:
            obs: (B, obs_dim)
            action: (B, act_dim)

        Returns:
            q1: (B, 1)
            q2: (B, 1)
        """
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q1 value only."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute minimum of Q1 and Q2."""
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)
