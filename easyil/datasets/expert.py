from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


def load_expert_npz(path: str | Path, num_trajs: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Load expert transitions from a .npz file.

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
        raise KeyError(f"Missing required keys in expert dataset '{path}': {missing}")

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
