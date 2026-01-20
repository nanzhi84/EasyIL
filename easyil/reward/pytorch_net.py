from __future__ import annotations

from typing import Callable

import numpy as np
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


def load_reward_fn(
    pt_path: str,
    s_dim: int,
    a_dim: int,
    hidden_dim: int = 256,
    reward_scale: float = 1.0,
    device: str = "cuda",
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Load PyTorch reward model and return a reward function.

    Args:
        pt_path: Path to PyTorch model weights (.pth).
        s_dim: State/observation dimension.
        a_dim: Action dimension.
        hidden_dim: Hidden layer dimension.
        reward_scale: Reward scaling factor.
        device: Device to run inference on ("cuda" or "cpu").

    Returns:
        A function (obs, act) -> rewards, compatible with VecEnv interfaces.
    """
    print(f"[PyTorch] Loading reward model from {pt_path}...")

    device = device if torch.cuda.is_available() else "cpu"
    model = RewardNet(s_dim, a_dim, hidden_dim)
    model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    @torch.no_grad()
    def reward_fn(obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        single = obs.ndim == 1
        if single:
            obs, act = obs[np.newaxis, :], act[np.newaxis, :]

        obs_t = torch.from_numpy(obs).float().to(device)
        act_t = torch.from_numpy(act).float().to(device)
        r = model(obs_t, act_t).cpu().numpy() * reward_scale

        return r.squeeze() if single else r.squeeze(-1)

    print("[PyTorch] Reward model loaded.")
    return reward_fn
