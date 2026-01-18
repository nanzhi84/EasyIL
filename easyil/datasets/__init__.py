"""Dataset utilities for imitation learning."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Protocol, runtime_checkable

import numpy as np
from omegaconf import DictConfig

from easyil.datasets.loaders import load_trajectory_npz, load_transition_npz
from easyil.datasets.trajectory import TrajectoryDataLoader, TrajectoryDataset
from easyil.datasets.transition import TransitionDataLoader, TransitionDataset

if TYPE_CHECKING:
    from typing import Optional


@runtime_checkable
class ILDataset(Protocol):
    """Base protocol for all IL datasets."""

    obs_mean: Optional[np.ndarray]
    obs_std: Optional[np.ndarray]

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]: ...


def build_trajectory_dataset(
    dataset_cfg: DictConfig,
    obs_horizon: int,
    action_horizon: int,
) -> TrajectoryDataset:
    """Build a trajectory dataset for BC."""
    path = dataset_cfg.get("path")
    if path is None:
        raise KeyError("Missing required config: dataset.path")

    num_trajs = dataset_cfg.get("num_trajs", None)
    obs_normalize = dataset_cfg.get("obs_normalize", False)

    data = load_trajectory_npz(path, num_trajs=num_trajs)
    return TrajectoryDataset(
        data=data,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        obs_normalize=obs_normalize,
    )


def build_transition_dataset(dataset_cfg: DictConfig) -> TransitionDataset:
    """Build a transition dataset for actor-critic IL."""
    path = dataset_cfg.get("path")
    if path is None:
        raise KeyError("Missing required config: dataset.path")

    num_trajs = dataset_cfg.get("num_trajs", None)
    obs_normalize = dataset_cfg.get("obs_normalize", False)

    data = load_transition_npz(path, num_trajs=num_trajs)
    return TransitionDataset(data=data, obs_normalize=obs_normalize)


def build_dataset(
    dataset_cfg: DictConfig,
    obs_horizon: int = 1,
    action_horizon: int = 1,
) -> ILDataset:
    """Build a dataset from config using factory pattern.

    Supported types:
        - "trajectory" (default): TrajectoryDataset for BC
        - "transition": TransitionDataset for actor-critic IL
    """
    dataset_type = dataset_cfg.get("type", "trajectory")

    if dataset_type == "trajectory":
        return build_trajectory_dataset(dataset_cfg, obs_horizon, action_horizon)

    if dataset_type == "transition":
        return build_transition_dataset(dataset_cfg)

    raise ValueError(
        f"Unknown dataset type: '{dataset_type}'. "
        f"Supported types: 'trajectory', 'transition'"
    )


__all__ = [
    # Protocol
    "ILDataset",
    # Trajectory (BC)
    "TrajectoryDataLoader",
    "TrajectoryDataset",
    "build_trajectory_dataset",
    "load_trajectory_npz",
    # Transition (Actor-Critic IL)
    "TransitionDataLoader",
    "TransitionDataset",
    "build_transition_dataset",
    "load_transition_npz",
    # Factory
    "build_dataset",
]
