"""Data loading utilities for IL datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


def load_trajectory_npz(
    path: str | Path,
    num_trajs: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Load trajectory data from a .npz file.

    Used by TrajectoryDataset for BC algorithms.

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


def load_transition_npz(
    path: str | Path,
    num_trajs: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Load transition data from a .npz file.

    Used by TransitionDataset for actor-critic IL algorithms.

    Expected keys: obs, actions, next_obs, dones
    OR: obs, actions, done (will compute next_obs from consecutive obs)

    Args:
        path: Path to the .npz file.
        num_trajs: If provided, only load the first `num_trajs` trajectories.

    Returns:
        Dict with keys: obs, actions, next_obs, dones
    """
    data = np.load(Path(path), allow_pickle=True)

    # Check if next_obs is provided directly
    if "next_obs" in data:
        result = {
            "obs": np.asarray(data["obs"], dtype=np.float32),
            "actions": np.asarray(data["actions"], dtype=np.float32),
            "next_obs": np.asarray(data["next_obs"], dtype=np.float32),
            "dones": np.asarray(data.get("dones", data.get("done")), dtype=np.float32),
        }
    else:
        # Compute next_obs from consecutive observations
        obs = np.asarray(data["obs"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        done = np.asarray(data["done"], dtype=bool)

        # next_obs[i] = obs[i+1], except at episode boundaries
        next_obs = np.roll(obs, -1, axis=0)
        next_obs[-1] = obs[-1]

        # At terminal states, next_obs doesn't matter for training
        done_indices = np.where(done)[0]
        for idx in done_indices:
            if idx + 1 < len(obs):
                next_obs[idx] = obs[idx]

        result = {
            "obs": obs,
            "actions": actions,
            "next_obs": next_obs,
            "dones": done.astype(np.float32),
        }

    if num_trajs is not None and num_trajs > 0:
        done_mask = result["dones"] > 0.5
        done_indices = np.where(done_mask)[0]
        if num_trajs <= len(done_indices):
            end_idx = done_indices[num_trajs - 1] + 1
            result = {k: v[:end_idx] for k, v in result.items()}

    return result
