"""MLP-based policy for direct action chunk prediction."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MLPPolicyConfig:
    action_clip: float = 1.0
    use_amp: bool = True


class MLPPolicy(nn.Module):
    """Policy that directly predicts action chunks via MLP forward pass."""

    def __init__(self, net: nn.Module, cfg: MLPPolicyConfig) -> None:
        super().__init__()
        self.net = net
        self.cfg = cfg

    @torch.no_grad()
    def sample_action_chunk(
        self,
        obs: torch.Tensor,
        act_dim: int,
        action_horizon: int,
    ) -> torch.Tensor:
        """Predict an action chunk conditioned on obs history.

        Args:
            obs: Observation tensor of shape (B, obs_horizon, obs_dim).
            act_dim: Action dimension (unused, for interface compatibility).
            action_horizon: Number of actions to predict (unused, determined by net).

        Returns:
            Action chunk of shape (B, action_horizon, act_dim).
        """
        device = obs.device
        amp_ctx = torch.autocast(
            device_type=str(device.type),
            dtype=torch.float16,
            enabled=bool(self.cfg.use_amp),
        )

        with amp_ctx:
            actions = self.net(obs)

        if self.cfg.action_clip is not None:
            actions = torch.clamp(
                actions, -float(self.cfg.action_clip), float(self.cfg.action_clip)
            )
        return actions
