"""
Environment utilities.

Supports:
- Vectorized environments (DummyVecEnv, SubprocVecEnv)
- Observation/reward normalization (VecNormalize)
- Custom reward functions (learned reward models)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Tuple

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper, VecNormalize

DEFAULT_ACTION_CLIP = 1.0

# Sentinel value for using environment's original reward
ENV_REWS = "env_rews"

RewardFnType = Callable[[np.ndarray, np.ndarray], np.ndarray] | str


class RewardWrapper(VecEnvWrapper):
    """
    VecEnv wrapper that supports custom reward functions for training and comparison.

    Args:
        venv: The vectorized environment to wrap.
        reward_fn: Training reward function or "env_rews" for environment reward. Required.
        compare_reward_fn: Comparison reward function or "env_rews". Optional.

    The wrapper records compare_reward in info["compare_reward"] for tracking.
    """

    def __init__(
        self,
        venv: Any,
        reward_fn: RewardFnType,
        compare_reward_fn: RewardFnType | None = None,
    ):
        super().__init__(venv)
        self.reward_fn = reward_fn
        self.compare_reward_fn = compare_reward_fn
        self._last_obs: np.ndarray | None = None
        self._actions: np.ndarray | None = None

    def reset(self, **kwargs) -> np.ndarray:
        obs = self.venv.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        self._last_obs = obs
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions
        self.venv.step_async(actions)

    def _compute_reward(self, reward_fn: RewardFnType, env_rews: np.ndarray) -> np.ndarray:
        """Compute reward using reward_fn or return env_rews."""
        return env_rews if reward_fn == ENV_REWS else reward_fn(self._last_obs, self._actions)

    def step_wait(self):
        obs, env_rews, dones, infos = self.venv.step_wait()

        train_rews = self._compute_reward(self.reward_fn, env_rews)

        if self.compare_reward_fn is not None:
            compare_rews = self._compute_reward(self.compare_reward_fn, env_rews)
            for i, info in enumerate(infos):
                info["compare_reward"] = compare_rews[i]

        self._last_obs = obs
        return obs, train_rews, dones, infos


def _load_reward_fn(
    reward_cfg: DictConfig,
    obs_dim: int,
    action_dim: int,
) -> RewardFnType:
    """Load a reward function from config.

    Args:
        reward_cfg: Reward configuration with model_path or "env_rews".
        obs_dim: Observation dimension.
        action_dim: Action dimension.

    Returns:
        Callable reward function or ENV_REWS sentinel.
    """
    model_path = reward_cfg.get("model_path")
    if model_path is None:
        return ENV_REWS

    from easyil.reward import load_reward_fn

    return load_reward_fn(
        str(model_path),
        obs_dim,
        action_dim,
        int(reward_cfg.get("hidden_dim", 256)),
        float(reward_cfg.get("scale", 1.0)),
    )


def _wrap_reward(env: Any, env_cfg: DictConfig, env_id: str) -> Any:
    """Wrap env with RewardWrapper.

    Args:
        env: The vectorized environment.
        env_cfg: Environment configuration. Must contain "reward" config.
        env_id: Environment ID string.

    Returns:
        Environment wrapped with RewardWrapper.

    Raises:
        KeyError: If reward config is missing.
    """
    reward_cfg = env_cfg.get("reward")
    if reward_cfg is None:
        raise KeyError("Missing required config: env.reward")

    compare_cfg = env_cfg.get("compare_reward")

    temp_env = gym.make(env_id)
    obs_dim = int(np.prod(temp_env.observation_space.shape))
    action_dim = int(np.prod(temp_env.action_space.shape))
    temp_env.close()

    reward_fn = _load_reward_fn(reward_cfg, obs_dim, action_dim)
    compare_reward_fn = _load_reward_fn(compare_cfg, obs_dim, action_dim) if compare_cfg else None

    return RewardWrapper(
        env,
        reward_fn=reward_fn,
        compare_reward_fn=compare_reward_fn,
    )


def _wrap_normalize(env: Any, cfg: DictConfig, training: bool) -> Any:
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


def make_env(
    env_cfg: DictConfig,
    output_dir: Path,
    seed: int,
    n_envs: int = 1,
    training: bool = False,
    monitor_subdir: str = "eval",
) -> Any:
    """
    Create a vectorized environment with optional wrappers.

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
    env = _wrap_reward(env, env_cfg, env_id)
    env = _wrap_normalize(env, env_cfg, training=training)
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


def get_reward_wrapper(env: Any) -> RewardWrapper | None:
    """Get the RewardWrapper from env's wrapper chain, if present."""
    current = env
    while hasattr(current, "venv"):
        if isinstance(current, RewardWrapper):
            return current
        current = current.venv
    return current if isinstance(current, RewardWrapper) else None


def has_reward_wrapper(env: Any) -> bool:
    """Check if env has a RewardWrapper in its wrapper chain."""
    return get_reward_wrapper(env) is not None


def has_compare_reward(env: Any) -> bool:
    """Check if env has compare_reward tracking enabled."""
    wrapper = get_reward_wrapper(env)
    if wrapper is None:
        return False
    return wrapper.compare_reward_fn is not None


def save_vecnormalize(env: Any, path: Path) -> None:
    """Save VecNormalize stats if enabled."""
    if isinstance(env, VecNormalize):
        env.save(str(path))
