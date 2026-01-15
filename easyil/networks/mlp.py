"""MLP backbone for neural networks."""
from __future__ import annotations

from typing import List

import torch
from torch import nn


class MLPBackbone(nn.Module):
    """A simple MLP backbone with configurable depth and width."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 512,
        depth: int = 4,
        dropout: float = 0.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        act_cls = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}.get(activation.lower(), nn.SiLU)

        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), act_cls()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_dim).

        Returns:
            Output tensor of shape (B, out_dim).
        """
        return self.net(x)
