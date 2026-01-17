"""Diffusion BC module implementation (JAX/Flax)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training import train_state
from omegaconf import DictConfig

from easyil.algos import register_algo
from easyil.envs import infer_env_dims
from easyil.algos.diffusion_bc.networks import build_noise_predictor
from easyil.algos.diffusion_bc.policy import (
    DiffusionPolicyConfig,
    sample_action_chunk,
    sample_action_chunk_at_step,
)
from easyil.algos.diffusion_bc.schedulers import BaseScheduler, build_scheduler


class TrainState(train_state.TrainState):
    """Extended train state with EMA parameters."""

    ema_params: Dict[str, Any] = struct.field(pytree_node=True)
    ema_decay: float = struct.field(pytree_node=False)


def create_train_state(
    rng_key: jnp.ndarray,
    net: Any,
    obs_dim: int,
    act_dim: int,
    obs_horizon: int,
    action_horizon: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    ema_decay: float = 0.999,
) -> TrainState:
    """Initialize training state with network parameters."""
    # Create dummy inputs for initialization
    dummy_xt = jnp.ones((1, action_horizon, act_dim))
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    dummy_obs = jnp.ones((1, obs_horizon, obs_dim))

    params = net.init(rng_key, dummy_xt, dummy_t, dummy_obs)

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    return TrainState.create(
        apply_fn=net.apply,
        params=params,
        tx=tx,
        ema_params=params.copy(),
        ema_decay=ema_decay,
    )


def update_ema(state: TrainState) -> TrainState:
    """Update EMA parameters."""
    new_ema = jax.tree.map(
        lambda ema, p: ema * state.ema_decay + p * (1 - state.ema_decay),
        state.ema_params,
        state.params,
    )
    return state.replace(ema_params=new_ema)


@dataclass
class DiffusionBCModule:
    """Module for Diffusion-based Behavior Cloning (JAX)."""

    net: Any  # Flax module
    scheduler: BaseScheduler
    obs_dim: int
    act_dim: int
    obs_horizon: int
    action_horizon: int
    action_clip: float = 1.0
    obs_noise: float = 0.0

    # Training state (set after initialization)
    state: Optional[TrainState] = field(default=None, repr=False)

    def init_state(
        self,
        rng_key: jnp.ndarray,
        learning_rate: float,
        weight_decay: float = 0.0,
        ema_decay: float = 0.999,
    ) -> None:
        """Initialize training state."""
        self.state = create_train_state(
            rng_key=rng_key,
            net=self.net,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            ema_decay=ema_decay,
        )

    def compute_loss(
        self,
        params: Dict[str, Any],
        batch: Dict[str, jnp.ndarray],
        rng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute diffusion loss."""
        obs = batch["obs"]  # (B, To, obs_dim)
        x0 = batch["actions"]  # (B, Ta, act_dim)

        B = x0.shape[0]

        rng_key, noise_key, t_key, obs_noise_key = jax.random.split(rng_key, 4)

        if self.obs_noise > 0.0:
            obs = obs + jax.random.normal(obs_noise_key, obs.shape) * self.obs_noise

        t = jax.random.randint(t_key, (B,), 0, int(self.scheduler.T))
        noise = jax.random.normal(noise_key, x0.shape)
        xt = self.scheduler.q_sample(x0, t, noise)
        pred = self.state.apply_fn(params, xt, t, obs)
        loss = jnp.mean((pred - noise) ** 2)
        return loss

    @jax.jit
    def train_step(
        self,
        state: TrainState,
        batch: Dict[str, jnp.ndarray],
        rng_key: jnp.ndarray,
    ) -> Tuple[TrainState, float]:
        """Single training step."""

        def loss_fn(params):
            return self.compute_loss(params, batch, rng_key)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        state = update_ema(state)
        return state, loss

    def sample_actions(
        self,
        rng_key: jnp.ndarray,
        obs: jnp.ndarray,
        use_ema: bool = True,
    ) -> jnp.ndarray:
        """Sample action chunk for inference."""
        params = self.state.ema_params if use_ema else self.state.params
        return sample_action_chunk(
            rng_key=rng_key,
            obs=obs,
            net_apply=self.state.apply_fn,
            params=params,
            scheduler=self.scheduler,
            action_clip=self.action_clip,
            act_dim=self.act_dim,
            action_horizon=self.action_horizon,
        )

    def sample_actions_at_step(
        self,
        rng_key: jnp.ndarray,
        obs: jnp.ndarray,
        stop_step: int,
        return_x0: bool = False,
        use_ema: bool = True,
    ) -> jnp.ndarray:
        """Sample action chunk at a specific diffusion step."""
        params = self.state.ema_params if use_ema else self.state.params
        return sample_action_chunk_at_step(
            rng_key=rng_key,
            obs=obs,
            net_apply=self.state.apply_fn,
            params=params,
            scheduler=self.scheduler,
            action_clip=self.action_clip,
            act_dim=self.act_dim,
            action_horizon=self.action_horizon,
            stop_step=stop_step,
            return_x0=return_x0,
        )

    def save(self, path: str) -> None:
        """Save model parameters."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "params": self.state.params,
                    "ema_params": self.state.ema_params,
                },
                f,
            )

    def load(self, path: str) -> None:
        """Load model parameters."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.state = self.state.replace(
            params=data["params"],
            ema_params=data["ema_params"],
        )


@register_algo("diffusion_bc")
def diffusion_bc(cfg: DictConfig, eval_env: Any, output_dir: str) -> DiffusionBCModule:
    obs_dim, act_dim, act_clip = infer_env_dims(eval_env)

    obs_horizon = int(cfg.get("obs_horizon", 1))
    action_horizon = int(cfg.get("action_horizon", 16))
    obs_noise = float(cfg.get("obs_noise", 0.0))
    action_clip = float(cfg.policy.get("action_clip", act_clip))

    scheduler = build_scheduler(cfg.scheduler)
    net = build_noise_predictor(cfg.model, obs_dim, act_dim, obs_horizon, action_horizon)

    return DiffusionBCModule(
        net=net,
        scheduler=scheduler,
        obs_dim=obs_dim,
        act_dim=act_dim,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        action_clip=action_clip,
        obs_noise=obs_noise,
    )
