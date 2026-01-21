"""Offline data collector for loading expert demonstrations."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from easyil.buffers import UnifiedBuffer


def load_npz(path: str | Path, num_trajs: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Load transitions from a .npz file.

    Expected keys: obs, actions, done

    Args:
        path: Path to the .npz file.
        num_trajs: If provided, only load the first `num_trajs` trajectories.

    Returns:
        Dict with keys: obs, actions, rewards, dones, next_obs
    """
    data = np.load(Path(path), allow_pickle=True)

    required_keys = ["obs", "actions", "done"]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise KeyError(f"Missing required keys in dataset '{path}': {missing}")

    obs = np.asarray(data["obs"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    dones = np.asarray(data["done"], dtype=bool)

    # Create next_obs by shifting
    next_obs = np.roll(obs, -1, axis=0)
    next_obs[-1] = obs[-1]  # last obs has no next

    # Handle rewards: use zeros if not present
    if "rewards" in data:
        rewards = np.asarray(data["rewards"], dtype=np.float32)
    else:
        rewards = np.zeros(len(obs), dtype=np.float32)

    result = {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "next_obs": next_obs,
    }

    if num_trajs is not None and num_trajs > 0:
        done_indices = np.where(dones)[0]
        if num_trajs <= len(done_indices):
            end_idx = done_indices[num_trajs - 1] + 1
            result = {k: v[:end_idx] for k, v in result.items()}

    return result


class OfflineCollector:
    """Collector that loads offline data into buffer."""

    def __init__(self, path: str | Path, source_label: str = "expert") -> None:
        self.path = Path(path)
        self.source_label = source_label

    def collect_to_buffer(
        self,
        buffer: "UnifiedBuffer",
        num_trajs: Optional[int] = None,
        obs_normalize: bool = False,
    ) -> tuple[int, Optional[Dict[str, np.ndarray]]]:
        """
        Load data into buffer.

        Args:
            buffer: Target buffer.
            num_trajs: Number of trajectories to load.
            obs_normalize: Whether to normalize observations.

        Returns:
            (num_transitions, norm_stats or None)
        """
        data = load_npz(self.path, num_trajs=num_trajs)
        n = len(data["obs"])

        norm_stats = None
        if obs_normalize:
            obs_mean = data["obs"].mean(axis=0)
            obs_std = data["obs"].std(axis=0) + 1e-6
            data["obs"] = (data["obs"] - obs_mean) / obs_std
            data["next_obs"] = (data["next_obs"] - obs_mean) / obs_std
            norm_stats = {"mean": obs_mean, "std": obs_std}

        # Add source label
        data["source"] = np.full(n, self.source_label, dtype=object)

        buffer.extend(data)
        return n, norm_stats