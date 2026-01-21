"""Unified buffer implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from easyil.buffers.samplers import Sampler


@dataclass
class BufferConfig:
    """Buffer configuration."""

    capacity: int = 0  # 0 = unlimited
    device: str = "cpu"


class UnifiedBuffer:
    """
    Unified dynamic buffer with pluggable sampling strategies.

    All data (expert, policy, mixed) stored in the same container.
    The 'source' field distinguishes data origin for algorithms like GAIL/SQIL.
    """

    CORE_FIELDS = ("obs", "actions", "rewards", "dones", "next_obs")
    META_FIELDS = ("source", "episode_id")

    def __init__(self, cfg: BufferConfig, sampler: Optional["Sampler"] = None) -> None:
        self.cfg = cfg
        self._chunks: Dict[str, List[np.ndarray]] = {}
        self._arrays: Dict[str, np.ndarray] = {}
        self._size = 0
        self._consolidated = False

        # Lazy import to avoid circular dependency
        if sampler is None:
            from easyil.buffers.samplers import RandomSampler

            sampler = RandomSampler()
        self.sampler = sampler
        self.sampler.set_buffer(self)

    def add(self, transition: Dict[str, np.ndarray]) -> None:
        """Add a single transition."""
        self._consolidated = False
        for key, val in transition.items():
            if key not in self._chunks:
                self._chunks[key] = []
            arr = np.asarray(val)
            # Ensure at least 1D for concatenation
            if arr.ndim == 0:
                arr = arr.reshape(1)
            else:
                arr = arr.reshape(1, *arr.shape)
            self._chunks[key].append(arr)
        self._size += 1
        self._enforce_capacity()

    def extend(self, data: Dict[str, np.ndarray]) -> None:
        """Add multiple transitions at once."""
        self._consolidated = False
        first_key = next(iter(data.keys()))
        n = len(data[first_key])

        for key, val in data.items():
            arr = np.asarray(val)
            if key not in self._chunks:
                self._chunks[key] = []
            self._chunks[key].append(arr)

        self._size += n
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        """Enforce capacity limit by removing oldest data."""
        if self.cfg.capacity <= 0 or self._size <= self.cfg.capacity:
            return

        self._consolidate()
        excess = self._size - self.cfg.capacity
        for key in self._arrays:
            self._arrays[key] = self._arrays[key][excess:]
        self._size = self.cfg.capacity
        self._chunks = {k: [v] for k, v in self._arrays.items()}

    def _consolidate(self) -> None:
        """Merge chunk lists into contiguous arrays (lazy)."""
        if self._consolidated:
            return

        for key, chunks in self._chunks.items():
            if not chunks:
                continue
            if len(chunks) == 1:
                self._arrays[key] = chunks[0]
            else:
                self._arrays[key] = np.concatenate(chunks, axis=0)

        self._chunks = {k: [v] for k, v in self._arrays.items()}
        self._consolidated = True

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor | np.ndarray]:
        """Sample a batch using the configured sampler."""
        self._consolidate()
        indices = self.sampler.sample_indices(batch_size)
        device = torch.device(self.cfg.device)

        result = {}
        for k, v in self._arrays.items():
            sampled = v[indices]
            # Keep string/object arrays as numpy, convert numeric to tensor
            if sampled.dtype.kind in ("U", "S", "O"):
                result[k] = sampled
            else:
                result[k] = torch.from_numpy(sampled).to(device)
        return result

    def sample_chunk(
        self,
        batch_size: int,
        obs_horizon: int,
        action_horizon: int,
    ) -> Dict[str, torch.Tensor]:
        """Sample observation history and action chunks for BC-style training."""
        self._consolidate()
        indices = self.sampler.sample_indices(batch_size)
        device = torch.device(self.cfg.device)

        obs = self._arrays["obs"]
        actions = self._arrays["actions"]

        obs_chunks = []
        act_chunks = []
        for i in indices:
            start = max(0, i - obs_horizon + 1)
            obs_hist = obs[start : i + 1]
            if len(obs_hist) < obs_horizon:
                pad = np.tile(obs_hist[0:1], (obs_horizon - len(obs_hist), 1))
                obs_hist = np.concatenate([pad, obs_hist], axis=0)
            obs_chunks.append(obs_hist)
            act_seq = actions[i : i + action_horizon]
            if len(act_seq) < action_horizon:
                pad = np.tile(act_seq[-1:], (action_horizon - len(act_seq), 1))
                act_seq = np.concatenate([act_seq, pad], axis=0)
            act_chunks.append(act_seq)

        return {
            "obs": torch.from_numpy(np.stack(obs_chunks)).float().to(device),
            "actions": torch.from_numpy(np.stack(act_chunks)).float().to(device),
        }

    def get_array(self, key: str) -> np.ndarray:
        """Get raw array (for sampler use)."""
        self._consolidate()
        return self._arrays.get(key, np.array([]))

    def has_field(self, key: str) -> bool:
        """Check if a field exists."""
        return key in self._chunks or key in self._arrays

    def __len__(self) -> int:
        return self._size

    def as_dataloader(
        self,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Return an infinite iterator yielding batches."""
        while True:
            yield self.sample(batch_size)
