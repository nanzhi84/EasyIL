"""MLP reward network (JAX/Flax)."""
from __future__ import annotations

import pickle
from typing import Callable, Dict, Any

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class RewardNet(nn.Module):
    """MLP reward network."""

    hidden_dim: int = 256

    @nn.compact
    def __call__(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([states, actions], axis=-1)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x


def load_reward_fn(
    model_path: str,
    s_dim: int,
    a_dim: int,
    hidden_dim: int = 256,
    reward_scale: float = 1.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Load reward model and return a JIT-compiled reward function.

    Args:
        model_path: Path to model weights (.pkl).
        s_dim: State/observation dimension.
        a_dim: Action dimension.
        hidden_dim: Hidden layer dimension.
        reward_scale: Reward scaling factor.

    Returns:
        A function (obs, act) -> rewards, compatible with VecEnv interfaces.
    """
    print(f"[Reward] Loading model from {model_path}...")

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    params = data["params"] if "params" in data else data

    model = RewardNet(hidden_dim=hidden_dim)

    @jax.jit
    def predict(states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        return model.apply({"params": params}, states, actions)

    def reward_fn(obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        single = obs.ndim == 1
        if single:
            obs, act = obs[np.newaxis, :], act[np.newaxis, :]
        r = np.asarray(predict(obs, act)) * reward_scale
        return r.squeeze() if single else r.squeeze(-1)

    print("[Reward] Model loaded.")
    return reward_fn


def save_reward_params(params: Dict[str, Any], path: str) -> None:
    """Save reward network parameters to pickle file."""
    with open(path, "wb") as f:
        pickle.dump({"params": params}, f)
