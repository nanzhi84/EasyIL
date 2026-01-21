"""Common embedding layers for neural networks."""
from __future__ import annotations

import torch
from torch import nn


class ObsEncoder(nn.Module):
    """Observation encoder that flattens and projects observation history."""

    def __init__(self, obs_dim: int, obs_horizon: int, emb_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.obs_horizon = int(obs_horizon)
        self.in_dim = self.obs_dim * self.obs_horizon
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation history.

        Args:
            obs: Observation tensor of shape (B, obs_horizon, obs_dim).

        Returns:
            Embedding of shape (B, emb_dim).
        """
        b = obs.shape[0]
        x = obs.reshape(b, -1)
        return self.net(x)
