"""MLP-based policy for direct action chunk prediction (JAX)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import jax.numpy as jnp


@dataclass
class MLPPolicyConfig:
    action_clip: float = 1.0


def sample_action_chunk(
    obs: jnp.ndarray,
    net_apply: Callable,
    params: Dict[str, Any],
    action_clip: float,
) -> jnp.ndarray:
    """Predict an action chunk conditioned on obs history.

    Args:
        obs: Observation tensor of shape (B, obs_horizon, obs_dim).
        net_apply: Network apply function.
        params: Network parameters.
        action_clip: Action clipping value.

    Returns:
        Action chunk of shape (B, action_horizon, act_dim).
    """
    actions = net_apply(params, obs)

    if action_clip is not None:
        actions = jnp.clip(actions, -float(action_clip), float(action_clip))

    return actions
