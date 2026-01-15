"""1D UNet backbone for sequence modeling."""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn


class ResBlock1D(nn.Module):
    """1D Residual block with embedding conditioning."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_ch), num_channels=in_ch)
        self.norm2 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.emb_proj(emb).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class UNet1DBackbone(nn.Module):
    """A compact 1D U-Net backbone for sequence modeling.

    Treats sequence length as the 1D spatial axis and feature dimension as channels.
    Designed to be used with conditioning embeddings (e.g., time + observation).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int = 256,
        base_channels: int = 128,
        channel_mults: Optional[List[int]] = None,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        if channel_mults is None:
            channel_mults = [1, 2, 2]

        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        # Down path
        downs = []
        cur = base_channels
        self._skip_chs: List[int] = []
        for mult in channel_mults:
            out_ch = base_channels * int(mult)
            downs.append(ResBlock1D(cur, out_ch, emb_dim, groups=groups))
            downs.append(nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1))
            self._skip_chs.append(out_ch)
            cur = out_ch
        self.downs = nn.ModuleList(downs)

        # Mid
        self.mid1 = ResBlock1D(cur, cur, emb_dim, groups=groups)
        self.mid2 = ResBlock1D(cur, cur, emb_dim, groups=groups)

        # Up path
        ups = []
        for mult in reversed(channel_mults):
            out_ch = base_channels * int(mult)
            ups.append(nn.ConvTranspose1d(cur, out_ch, kernel_size=4, stride=2, padding=1))
            ups.append(ResBlock1D(out_ch + out_ch, out_ch, emb_dim, groups=groups))
            cur = out_ch
        self.ups = nn.ModuleList(ups)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=min(groups, cur), num_channels=cur),
            nn.SiLU(),
            nn.Conv1d(cur, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, seq_len).
            emb: Conditioning embedding of shape (B, emb_dim).

        Returns:
            Output tensor of shape (B, out_channels, seq_len).
        """
        x = self.in_conv(x)

        skips = []
        i = 0
        while i < len(self.downs):
            x = self.downs[i](x, emb)
            skips.append(x)
            x = self.downs[i + 1](x)
            i += 2

        x = self.mid1(x, emb)
        x = self.mid2(x, emb)

        j = 0
        while j < len(self.ups):
            x = self.ups[j](x)  # upsample
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                min_len = min(x.shape[-1], skip.shape[-1])
                x = x[..., :min_len]
                skip = skip[..., :min_len]
            x = torch.cat([x, skip], dim=1)
            x = self.ups[j + 1](x, emb)
            j += 2

        return self.out_conv(x)
