from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import torch


class JaxRewardNet(nn.Module):
    """JAX/Flax reward network, mirroring the PyTorch RewardNet structure."""

    hidden_dim: int

    @nn.compact
    def __call__(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([states, actions], axis=-1)
        x = nn.Dense(features=self.hidden_dim, name="Dense_0")(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim, name="Dense_1")(x)
        x = nn.relu(x)
        x = nn.Dense(features=1, name="Dense_2")(x)
        return x


def load_jax_reward_fn(
    pt_path: str,
    s_dim: int,
    a_dim: int,
    hidden_dim: int = 256,
    reward_scale: float = 1.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Load PyTorch reward model and return a JIT-compiled JAX reward function.

    Args:
        pt_path: Path to PyTorch model weights (.pth).
        s_dim: State/observation dimension.
        a_dim: Action dimension.
        hidden_dim: Hidden layer dimension.
        reward_scale: Reward scaling factor.

    Returns:
        A function (obs, act) -> rewards, compatible with VecEnv interfaces.
    """
    print(f"[JAX] Loading reward model from {pt_path}...")

    pt_state = torch.load(pt_path, map_location="cpu", weights_only=True)

    model = JaxRewardNet(hidden_dim=hidden_dim)
    key = jax.random.PRNGKey(0)
    _ = model.init(key, jnp.ones((1, s_dim)), jnp.ones((1, a_dim)))

    def convert_layer(idx: int) -> dict:
        return {
            "kernel": jnp.array(pt_state[f"net.{idx}.weight"].numpy().T),
            "bias": jnp.array(pt_state[f"net.{idx}.bias"].numpy()),
        }

    params = {
        "Dense_0": convert_layer(0),
        "Dense_1": convert_layer(2),
        "Dense_2": convert_layer(4),
    }

    @jax.jit
    def predict(states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        return model.apply({"params": params}, states, actions)

    def reward_fn(obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        single = obs.ndim == 1
        if single:
            obs, act = obs[np.newaxis, :], act[np.newaxis, :]
        r = np.asarray(predict(obs, act)) * reward_scale
        return r.squeeze() if single else r.squeeze(-1)

    print("[JAX] Reward model loaded.")
    return reward_fn
