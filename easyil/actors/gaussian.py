"""Gaussian actor for stochastic policies (SAC-style)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal

from easyil.networks import MLPBackbone


@dataclass
class GaussianActorConfig:
    """Configuration for GaussianActor."""

    obs_dim: int
    act_dim: int
    hidden: int = 256
    depth: int = 2
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    action_clip: float = 1.0


class GaussianActor(nn.Module):
    """
    Gaussian policy actor that outputs mean and log_std for actions.

    Used for SAC, PPO, and other stochastic policy algorithms.
    """

    def __init__(self, cfg: GaussianActorConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.backbone = MLPBackbone(
            in_dim=cfg.obs_dim,
            out_dim=cfg.hidden,
            hidden=cfg.hidden,
            depth=cfg.depth - 1,
            dropout=0.0,
        )

        self.mean_head = nn.Linear(cfg.hidden, cfg.act_dim)
        self.log_std_head = nn.Linear(cfg.hidden, cfg.act_dim)

    @property
    def action_dim(self) -> int:
        return self.cfg.act_dim

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean and log_std.

        Args:
            obs: (B, obs_dim)

        Returns:
            mean: (B, act_dim)
            log_std: (B, act_dim)
        """
        h = self.backbone(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.cfg.log_std_min, self.cfg.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action with reparameterization trick.

        Returns:
            action: (B, act_dim) sampled action
            log_prob: (B,) log probability
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Reparameterized sample
        x = dist.rsample()

        # Squash through tanh
        action = torch.tanh(x) * self.cfg.action_clip

        # Log prob with tanh correction
        log_prob = dist.log_prob(x).sum(dim=-1)
        log_two = torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
        log_prob -= (2 * (log_two - x - torch.nn.functional.softplus(-2 * x))).sum(dim=-1)

        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action (mean passed through tanh)."""
        mean, _ = self.forward(obs)
        return torch.tanh(mean) * self.cfg.action_clip

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of given action."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Inverse tanh to get pre-squash action
        action_clipped = torch.clamp(action / self.cfg.action_clip, -0.999, 0.999)
        x = torch.atanh(action_clipped)

        log_prob = dist.log_prob(x).sum(dim=-1)
        log_two = torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
        log_prob -= (2 * (log_two - x - torch.nn.functional.softplus(-2 * x))).sum(dim=-1)

        return log_prob
