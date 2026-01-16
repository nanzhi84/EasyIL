"""Base protocols for behavior cloning algorithms."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class BasePolicy(Protocol):
    """Protocol that all policies must implement.

    A policy takes observation history and outputs action chunks.
    """

    def sample_action_chunk(
        self,
        obs: torch.Tensor,
        act_dim: int,
        action_horizon: int,
    ) -> torch.Tensor:
        """Sample an action chunk conditioned on observation history.

        Args:
            obs: Observation tensor of shape (B, obs_horizon, obs_dim).
            act_dim: Action dimension.
            action_horizon: Number of actions to predict.

        Returns:
            Action chunk of shape (B, action_horizon, act_dim).
        """
        ...

    def eval(self) -> nn.Module:
        """Set policy to evaluation mode."""
        ...

    def train(self, mode: bool = True) -> nn.Module:
        """Set policy to training mode."""
        ...


@runtime_checkable
class BCModule(Protocol):
    """Protocol that all BC modules must implement.

    A BCModule encapsulates the network, EMA network, policy, and training logic.
    """

    net: nn.Module
    ema_net: nn.Module
    policy: BasePolicy
    obs_dim: int
    act_dim: int
    obs_horizon: int
    action_horizon: int
    ema_decay: float

    def to(self, device: torch.device) -> "BCModule":
        """Move module to device."""
        ...

    def update_ema(self) -> None:
        """Update EMA network with current network weights."""
        ...

    def use_ema_for_inference(self) -> None:
        """Switch policy to use EMA network for inference."""
        ...

    def use_train_net_for_inference(self) -> None:
        """Switch policy to use training network for inference."""
        ...

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss given a batch of data."""
        ...
