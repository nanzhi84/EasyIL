from __future__ import annotations

from typing import Dict, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ChunkedExpertDataset(Dataset):
    """Samples (obs_history, action_chunk) pairs without crossing episodes."""

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        obs_horizon: int,
        action_horizon: int,
        obs_normalize: bool = False,
    ) -> None:
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)

        self.obs = np.asarray(data["obs"], dtype=np.float32)
        self.actions = np.asarray(data["actions"], dtype=np.float32)
        episode_starts = np.asarray(data["episode_starts"], dtype=np.int64)

        self.ep_ids = np.cumsum(episode_starts) - 1
        self.ep_start_idx = np.maximum.accumulate(np.where(episode_starts, np.arange(len(episode_starts)), 0))

        self.valid_starts = self._build_valid_starts()

        self.obs_mean: Optional[np.ndarray] = None
        self.obs_std: Optional[np.ndarray] = None
        if obs_normalize:
            self.obs_mean = self.obs.mean(axis=0)
            self.obs_std = self.obs.std(axis=0) + 1e-6

    def _build_valid_starts(self) -> np.ndarray:
        N = len(self.obs)
        candidates = np.arange(N - self.action_horizon + 1, dtype=np.int64)
        same_ep = self.ep_ids[candidates] == self.ep_ids[candidates + self.action_horizon - 1]
        enough_hist = (candidates - (self.obs_horizon - 1)) >= self.ep_start_idx[candidates]
        valid = candidates[same_ep & enough_hist]
        if len(valid) == 0:
            raise RuntimeError("No valid windows found. Check horizons and episode boundaries.")
        return valid

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.valid_starts[idx]
        obs_hist = self.obs[i - (self.obs_horizon - 1) : i + 1].copy()
        act_chunk = self.actions[i : i + self.action_horizon]

        if self.obs_mean is not None:
            obs_hist = (obs_hist - self.obs_mean) / self.obs_std

        return {"obs": torch.from_numpy(obs_hist), "actions": torch.from_numpy(act_chunk)}


def infinite_dataloader(dl: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    """Turn a finite DataLoader into an infinite iterator."""
    while True:
        yield from dl
