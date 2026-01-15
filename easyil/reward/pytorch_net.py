from __future__ import annotations

import torch
import torch.nn as nn


class RewardNet(nn.Module):
    """Simple MLP reward network for PyTorch-based training."""

    def __init__(self, s_dim: int, a_dim: int, hidden_dim: int = 256):
        super().__init__()
        inp = int(s_dim) + int(a_dim)
        h = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(inp, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)
