"""Transition dataset for actor-critic IL algorithms (pure NumPy)."""
from __future__ import annotations

from typing import Dict, Iterator, Optional

import numpy as np


class TransitionDataset:
    """Dataset for (s, a, s', done) transitions.

    Used by actor-critic IL algorithms like IQ-Learn, GAIL, AIRL.
    Unlike TrajectoryDataset, this provides single-step transitions.
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        obs_normalize: bool = False,
    ) -> None:
        """Initialize transition dataset.

        Args:
            data: Dict with keys 'obs', 'actions', 'next_obs', 'dones'.
            obs_normalize: If True, normalize observations.
        """
        self.obs = np.asarray(data["obs"], dtype=np.float32)
        self.actions = np.asarray(data["actions"], dtype=np.float32)
        self.next_obs = np.asarray(data["next_obs"], dtype=np.float32)
        self.dones = np.asarray(data["dones"], dtype=np.float32)

        if len(self.obs) != len(self.actions):
            raise ValueError(
                f"obs and actions must have same length, got {len(self.obs)} and {len(self.actions)}"
            )

        self.obs_mean: Optional[np.ndarray] = None
        self.obs_std: Optional[np.ndarray] = None

        if obs_normalize:
            all_obs = np.concatenate([self.obs, self.next_obs], axis=0)
            self.obs_mean = all_obs.mean(axis=0)
            self.obs_std = all_obs.std(axis=0) + 1e-6

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        obs = self.obs[idx].copy()
        next_obs = self.next_obs[idx].copy()

        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / self.obs_std
            next_obs = (next_obs - self.obs_mean) / self.obs_std

        return {
            "obs": obs,
            "actions": self.actions[idx],
            "next_obs": next_obs,
            "dones": self.dones[idx],
        }


class TransitionDataLoader:
    """Data loader for transition datasets (pure NumPy)."""

    def __init__(
        self,
        dataset: TransitionDataset,
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

            batch = {
                "obs": [],
                "actions": [],
                "next_obs": [],
                "dones": [],
            }
            for idx in batch_indices:
                sample = self.dataset[idx]
                for key in batch:
                    batch[key].append(sample[key])

            yield {key: np.stack(vals, axis=0) for key, vals in batch.items()}
