from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


def load_expert_npz(path: str | Path, num_trajs: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Load expert transitions from a .npz file.

    Expected keys: obs, action, done

    Args:
        path: Path to the .npz file.
        num_trajs: If provided, only load the first `num_trajs` trajectories.

    Returns:
        Dict with keys: obs, actions, episode_starts
    """
    data = np.load(Path(path), allow_pickle=True)

    done = np.asarray(data["done"], dtype=bool)
    episode_starts = np.zeros(len(done), dtype=np.int64)
    episode_starts[0] = 1
    done_indices = np.where(done)[0]
    next_ep_starts = done_indices[:-1] + 1
    episode_starts[next_ep_starts] = 1

    result = {
        "obs": data["obs"],
        "actions": data["action"],
        "episode_starts": episode_starts,
    }

    if num_trajs is not None and num_trajs > 0:
        episode_starts = np.asarray(result["episode_starts"], dtype=np.int64)
        ep_start_indices = np.where(episode_starts)[0]
        if num_trajs < len(ep_start_indices):
            end_idx = ep_start_indices[num_trajs]
            result = {k: v[:end_idx] for k, v in result.items()}

    return result
