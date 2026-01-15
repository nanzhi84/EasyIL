"""Common embedding layers for neural networks."""
from __future__ import annotations

import math

import torch
from torch import nn


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by MLP projection."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal timestep embeddings. (B,) -> (B, dim)"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.sinusoidal_embedding(t, self.dim)
        return self.net(x)


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
