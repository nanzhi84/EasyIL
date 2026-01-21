"""Environment utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

DEFAULT_ACTION_CLIP = 1.0


def make_env(
    env_cfg: DictConfig,
    output_dir: Path,
    seed: int,
    n_envs: int = 1,
    training: bool = False,
    monitor_subdir: str = "eval",
) -> Any:
    """
    Create a vectorized environment.

    Args:
        env_cfg: Environment configuration.
        output_dir: Output directory for monitor logs.
        seed: Random seed for the environment.
        n_envs: Number of parallel environments.
        training: If True, enable reward normalization updates.
        monitor_subdir: Subdirectory name under output_dir/monitor.
    """
    env_id = str(env_cfg.id)
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env_kwargs = dict(env_cfg.get("kwargs") or {})
    monitor_dir = output_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_env_cls,
        env_kwargs=env_kwargs,
        monitor_dir=str(monitor_dir / monitor_subdir),
    )
    return env


def infer_env_dims(env: Any) -> Tuple[int, int, float]:
    """Infer observation/action dimensions and action clip from env spaces."""
    if not hasattr(env, "observation_space") or not hasattr(env, "action_space"):
        raise TypeError("Expected env with observation_space and action_space")

    obs_space = env.observation_space
    act_space = env.action_space
    if getattr(obs_space, "shape", None) is None or getattr(act_space, "shape", None) is None:
        raise ValueError("Environment spaces must define shape")

    obs_dim = int(obs_space.shape[0])
    act_dim = int(act_space.shape[0])
    act_clip = float(getattr(act_space, "high", [DEFAULT_ACTION_CLIP])[0])
    return obs_dim, act_dim, act_clip
