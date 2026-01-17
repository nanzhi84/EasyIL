"""Diffusion-based policy for action chunk generation (JAX)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from easyil.algos.diffusion_bc.schedulers import BaseScheduler


@dataclass
class DiffusionPolicyConfig:
    action_clip: float = 1.0


def create_diffusion_sampler(
    net_apply: Callable,
    params: Dict[str, Any],
    scheduler: BaseScheduler,
    cfg: DiffusionPolicyConfig,
    act_dim: int,
    action_horizon: int,
) -> Callable:
    """Create a JIT-compiled diffusion sampling function.

    Args:
        net_apply: The network's apply function.
        params: Network parameters.
        scheduler: Diffusion scheduler.
        cfg: Policy configuration.
        act_dim: Action dimension.
        action_horizon: Action chunk length.

    Returns:
        A function (rng_key, obs) -> action_chunk
    """

    def sample_loop(rng_key: jnp.ndarray, obs: jnp.ndarray) -> jnp.ndarray:
        B = obs.shape[0]

        # Initialize from noise
        rng_key, init_key = jax.random.split(rng_key)
        xt = jax.random.normal(init_key, (B, action_horizon, act_dim))

        timesteps = scheduler.infer_timesteps
        t_list = list(timesteps.tolist())

        for i, t in enumerate(t_list):
            t_int = int(t)
            t_batch = jnp.full((B,), t_int, dtype=jnp.int32)
            eps = net_apply(params, xt, t_batch, obs)

            t_prev = int(t_list[i + 1]) if i + 1 < len(t_list) else 0
            rng_key, step_key = jax.random.split(rng_key)
            xt = scheduler.step(xt, t_int, t_prev, eps, step_key)

        x0 = xt

        if cfg.action_clip is not None:
            x0 = jnp.clip(x0, -float(cfg.action_clip), float(cfg.action_clip))

        return x0

    return sample_loop


def sample_action_chunk(
    rng_key: jnp.ndarray,
    obs: jnp.ndarray,
    net_apply: Callable,
    params: Dict[str, Any],
    scheduler: BaseScheduler,
    action_clip: float,
    act_dim: int,
    action_horizon: int,
) -> jnp.ndarray:
    """Sample an action chunk conditioned on obs history.

    Args:
        rng_key: JAX random key.
        obs: Observation tensor of shape (B, obs_horizon, obs_dim).
        net_apply: Network apply function.
        params: Network parameters.
        scheduler: Diffusion scheduler.
        action_clip: Action clipping value.
        act_dim: Action dimension.
        action_horizon: Number of actions to predict.

    Returns:
        Action chunk of shape (B, action_horizon, act_dim).
    """
    B = obs.shape[0]

    # Initialize from noise
    rng_key, init_key = jax.random.split(rng_key)
    xt = jax.random.normal(init_key, (B, action_horizon, act_dim))

    timesteps = scheduler.infer_timesteps
    t_list = list(timesteps.tolist())

    for i, t in enumerate(t_list):
        t_int = int(t)
        t_batch = jnp.full((B,), t_int, dtype=jnp.int32)
        eps = net_apply(params, xt, t_batch, obs)

        t_prev = int(t_list[i + 1]) if i + 1 < len(t_list) else 0
        rng_key, step_key = jax.random.split(rng_key)
        xt = scheduler.step(xt, t_int, t_prev, eps, step_key)

    x0 = xt

    if action_clip is not None:
        x0 = jnp.clip(x0, -float(action_clip), float(action_clip))

    return x0


def sample_action_chunk_at_step(
    rng_key: jnp.ndarray,
    obs: jnp.ndarray,
    net_apply: Callable,
    params: Dict[str, Any],
    scheduler: BaseScheduler,
    action_clip: float,
    act_dim: int,
    action_horizon: int,
    stop_step: int,
    return_x0: bool = False,
) -> jnp.ndarray:
    """Sample action chunk at a specific diffusion step for analysis.

    Args:
        rng_key: JAX random key.
        obs: Observation tensor of shape (B, obs_horizon, obs_dim).
        net_apply: Network apply function.
        params: Network parameters.
        scheduler: Diffusion scheduler.
        action_clip: Action clipping value.
        act_dim: Action dimension.
        action_horizon: Action chunk length.
        stop_step: Diffusion step to stop at (0 = fully denoised).
        return_x0: If True, return predicted x0; otherwise return noisy xt.

    Returns:
        Action chunk of shape (B, action_horizon, act_dim).
    """
    B = obs.shape[0]

    rng_key, init_key = jax.random.split(rng_key)
    xt = jax.random.normal(init_key, (B, action_horizon, act_dim))

    timesteps = scheduler.infer_timesteps
    t_list = list(timesteps.tolist())

    x0_pred = xt

    for i, t in enumerate(t_list):
        t_int = int(t)
        if t_int < stop_step:
            break

        t_batch = jnp.full((B,), t_int, dtype=jnp.int32)
        eps = net_apply(params, xt, t_batch, obs)

        x0_pred = scheduler.predict_x0_from_eps(xt, t_batch, eps)

        if t_int == stop_step:
            break

        t_prev = int(t_list[i + 1]) if i + 1 < len(t_list) else 0
        rng_key, step_key = jax.random.split(rng_key)
        xt = scheduler.step(xt, t_int, t_prev, eps, step_key)

    out = x0_pred if (return_x0 or stop_step == 0) else xt

    if action_clip is not None:
        out = jnp.clip(out, -float(action_clip), float(action_clip))

    return out
