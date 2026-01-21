"""Chunk sampler for BC-style training."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from easyil.buffers.samplers import register_sampler

if TYPE_CHECKING:
    from easyil.buffers.base import UnifiedBuffer


class ChunkSampler:
    """
    Sequential chunk sampling for BC-style training.

    Ensures obs_horizon + action_horizon windows don't cross episode boundaries.
    """

    def __init__(self, obs_horizon: int = 1, action_horizon: int = 1) -> None:
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.buffer: "UnifiedBuffer" = None  # type: ignore
        self._valid_indices: Optional[np.ndarray] = None

    def set_buffer(self, buffer: "UnifiedBuffer") -> None:
        self.buffer = buffer
        self._valid_indices = None

    def _compute_valid_indices(self) -> None:
        dones = self.buffer.get_array("dones").astype(bool)
        N = len(dones)

        if N < self.action_horizon:
            self._valid_indices = np.array([], dtype=np.int64)
            return

        episode_starts = np.zeros(N, dtype=bool)
        episode_starts[0] = True
        done_indices = np.where(dones)[0]
        if len(done_indices) > 0:
            next_starts = done_indices[:-1] + 1
            next_starts = next_starts[next_starts < N]
            episode_starts[next_starts] = True

        ep_ids = np.cumsum(episode_starts) - 1
        ep_start_idx = np.maximum.accumulate(np.where(episode_starts, np.arange(N), 0))

        candidates = np.arange(N - self.action_horizon + 1, dtype=np.int64)
        same_ep = ep_ids[candidates] == ep_ids[candidates + self.action_horizon - 1]
        enough_hist = (candidates - (self.obs_horizon - 1)) >= ep_start_idx[candidates]
        self._valid_indices = candidates[same_ep & enough_hist]

    def sample_indices(self, batch_size: int) -> np.ndarray:
        if self._valid_indices is None or len(self._valid_indices) == 0:
            self._compute_valid_indices()

        if len(self._valid_indices) == 0:
            return np.random.randint(0, len(self.buffer), size=batch_size)

        return np.random.choice(self._valid_indices, size=batch_size, replace=True)


@register_sampler("chunk")
def _build_chunk(obs_horizon: int = 1, action_horizon: int = 1, **kwargs) -> ChunkSampler:
    return ChunkSampler(obs_horizon=obs_horizon, action_horizon=action_horizon)
