"""Chunked dataset for behavior cloning (pure NumPy)."""
from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import numpy as np


class ChunkedExpertDataset:
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

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        i = self.valid_starts[idx]
        obs_hist = self.obs[i - (self.obs_horizon - 1) : i + 1].copy()
        act_chunk = self.actions[i : i + self.action_horizon]

        if self.obs_mean is not None:
            obs_hist = (obs_hist - self.obs_mean) / self.obs_std

        return {"obs": obs_hist, "actions": act_chunk}


class DataLoader:
    """Simple data loader for JAX training (pure NumPy)."""

    def __init__(
        self,
        dataset: ChunkedExpertDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)

        n_batches = len(self)
        for i in range(n_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            batch_indices = indices[start:end]

            batch_obs = []
            batch_actions = []
            for idx in batch_indices:
                sample = self.dataset[idx]
                batch_obs.append(sample["obs"])
                batch_actions.append(sample["actions"])

            yield {
                "obs": np.stack(batch_obs, axis=0),
                "actions": np.stack(batch_actions, axis=0),
            }
