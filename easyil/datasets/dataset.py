from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def load_npz(path: str | Path, num_trajs: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Load transitions from a .npz file.

    Expected keys: obs, actions, done

    Args:
        path: Path to the .npz file.
        num_trajs: If provided, only load the first `num_trajs` trajectories.

    Returns:
        Dict with keys: obs, actions, episode_starts
    """
    data = np.load(Path(path), allow_pickle=True)

    required_keys = ["obs", "actions", "done"]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise KeyError(f"Missing required keys in dataset '{path}': {missing}")

    done = np.asarray(data["done"], dtype=bool)
    episode_starts = np.zeros(len(done), dtype=np.int64)
    episode_starts[0] = 1
    done_indices = np.where(done)[0]
    next_ep_starts = done_indices[:-1] + 1
    episode_starts[next_ep_starts] = 1

    result = {
        "obs": data["obs"],
        "actions": data["actions"],
        "episode_starts": episode_starts,
    }

    if num_trajs is not None and num_trajs > 0:
        episode_starts = np.asarray(result["episode_starts"], dtype=np.int64)
        ep_start_indices = np.where(episode_starts)[0]
        if num_trajs < len(ep_start_indices):
            end_idx = ep_start_indices[num_trajs]
            result = {k: v[:end_idx] for k, v in result.items()}

    return result


class ChunkedDataset(Dataset):
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
