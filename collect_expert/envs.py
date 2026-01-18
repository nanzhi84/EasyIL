"""Simplified environment utilities for expert data collection.

This module provides only the environment creation functionality needed for
collecting expert trajectories. It does not include reward wrappers or
other training-specific features.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

DEFAULT_ACTION_CLIP = 1.0


def make_env(
    env_cfg: DictConfig,
    output_dir: Path,
    seed: int,
    n_envs: int = 1,
    training: bool = False,
    monitor_subdir: str = "eval",
):
    """
    Create a vectorized environment for data collection.

    Args:
        env_cfg: Environment configuration with 'id' and optional 'kwargs'.
        output_dir: Output directory for monitor logs.
        seed: Random seed for the environment.
        n_envs: Number of parallel environments.
        training: If True, enable reward normalization updates.
        monitor_subdir: Subdirectory name under output_dir/monitor.

    Returns:
        Vectorized environment instance.
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

    env = _wrap_normalize(env, env_cfg, training=training)
    return env


def _wrap_normalize(env, cfg: DictConfig, training: bool):
    """Wrap env with VecNormalize if enabled."""
    norm_cfg = cfg.get("normalize")
    if not norm_cfg or not norm_cfg.get("enabled"):
        return env
    wrapped = VecNormalize(
        env,
        norm_obs=bool(norm_cfg.get("norm_obs", True)),
        norm_reward=bool(norm_cfg.get("norm_reward", False)) and training,
        clip_obs=float(norm_cfg.get("clip_obs", 10.0)),
        clip_reward=float(norm_cfg.get("clip_reward", 10.0)),
    )
    wrapped.training = training
    return wrapped


def infer_env_dims(env) -> Tuple[int, int, float]:
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


def save_vecnormalize(env, path: Path) -> None:
    """Save VecNormalize stats if enabled."""
    if isinstance(env, VecNormalize):
        env.save(str(path))
