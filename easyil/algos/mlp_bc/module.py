"""MLP BC module implementation (JAX/Flax)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training import train_state
import flax.linen as nn
from omegaconf import DictConfig

from easyil.algos import register_algo
from easyil.envs import infer_env_dims
from easyil.networks import MLPBackbone
from easyil.algos.mlp_bc.policy import MLPPolicyConfig, sample_action_chunk


class MLPActionPredictor(nn.Module):
    """MLP that directly predicts action chunks from observation history."""

    obs_dim: int
    act_dim: int
    obs_horizon: int
    action_horizon: int
    hidden: int = 512
    depth: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass.

        Args:
            obs: Observation tensor of shape (B, obs_horizon, obs_dim).
            training: Whether in training mode (for dropout).

        Returns:
            Action chunk of shape (B, action_horizon, act_dim).
        """
        B = obs.shape[0]
        in_dim = self.obs_dim * self.obs_horizon
        out_dim = self.act_dim * self.action_horizon

        x = obs.reshape(B, -1)

        # MLP layers
        for _ in range(self.depth - 1):
            x = nn.Dense(features=self.hidden)(x)
            x = nn.silu(x)
            if self.dropout > 0:
                x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)

        out = nn.Dense(features=out_dim)(x)
        return out.reshape(B, self.action_horizon, self.act_dim)


class TrainState(train_state.TrainState):
    """Extended train state with EMA parameters."""

    ema_params: Dict[str, Any] = struct.field(pytree_node=True)
    ema_decay: float = struct.field(pytree_node=False)


def create_train_state(
    rng_key: jnp.ndarray,
    net: nn.Module,
    obs_dim: int,
    obs_horizon: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    ema_decay: float = 0.999,
) -> TrainState:
    """Initialize training state with network parameters."""
    dummy_obs = jnp.ones((1, obs_horizon, obs_dim))
    params = net.init(rng_key, dummy_obs)

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


def _build_train_step(
    apply_fn: Callable,
    obs_noise: float,
) -> Callable[[TrainState, Dict[str, jnp.ndarray], jnp.ndarray], Tuple[TrainState, float]]:
    def train_step(
        state: TrainState,
        batch: Dict[str, jnp.ndarray],
        rng_key: jnp.ndarray,
    ) -> Tuple[TrainState, float]:
        obs = batch["obs"]
        actions = batch["actions"]

        if obs_noise > 0.0:
            obs = obs + jax.random.normal(rng_key, obs.shape) * obs_noise

        def loss_fn(params: Dict[str, Any]) -> jnp.ndarray:
            pred = apply_fn(params, obs)
            return jnp.mean((pred - actions) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        state = update_ema(state)
        return state, loss

    return jax.jit(train_step)


@dataclass
class MLPBCModule:
    """Module for MLP-based Behavior Cloning (JAX)."""

    net: nn.Module
    obs_dim: int
    act_dim: int
    obs_horizon: int
    action_horizon: int
    action_clip: float = 1.0
    obs_noise: float = 0.0

    # Training state (set after initialization)
    state: Optional[TrainState] = field(default=None, repr=False)
    _train_step_fn: Optional[
        Callable[[TrainState, Dict[str, jnp.ndarray], jnp.ndarray], Tuple[TrainState, float]]
    ] = field(default=None, init=False, repr=False)

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
            obs_horizon=self.obs_horizon,
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
        """Compute MSE loss."""
        obs = batch["obs"]  # (B, To, obs_dim)
        actions = batch["actions"]  # (B, Ta, act_dim)

        if self.obs_noise > 0.0:
            obs = obs + jax.random.normal(rng_key, obs.shape) * self.obs_noise

        pred = self.state.apply_fn(params, obs)
        loss = jnp.mean((pred - actions) ** 2)
        return loss

    def train_step(
        self,
        state: TrainState,
        batch: Dict[str, jnp.ndarray],
        rng_key: jnp.ndarray,
    ) -> Tuple[TrainState, float]:
        """Single training step."""
        if state is None:
            raise ValueError("Train state must be initialized before calling train_step.")
        if self._train_step_fn is None:
            self._train_step_fn = _build_train_step(
                apply_fn=state.apply_fn,
                obs_noise=self.obs_noise,
            )
        return self._train_step_fn(state, batch, rng_key)

    def sample_actions(
        self,
        obs: jnp.ndarray,
        use_ema: bool = True,
    ) -> jnp.ndarray:
        """Sample action chunk for inference."""
        params = self.state.ema_params if use_ema else self.state.params
        return sample_action_chunk(
            obs=obs,
            net_apply=self.state.apply_fn,
            params=params,
            action_clip=self.action_clip,
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


@register_algo("mlp_bc")
def mlp_bc(cfg: DictConfig, eval_env: Any, output_dir: str) -> MLPBCModule:
    obs_dim, act_dim, act_clip = infer_env_dims(eval_env)

    obs_horizon = int(cfg.get("obs_horizon", 1))
    action_horizon = int(cfg.get("action_horizon", 1))
    obs_noise = float(cfg.get("obs_noise", 0.0))
    action_clip = float(cfg.policy.get("action_clip", act_clip))

    net = MLPActionPredictor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        hidden=int(cfg.model.get("hidden", 512)),
        depth=int(cfg.model.get("depth", 4)),
        dropout=float(cfg.model.get("dropout", 0.0)),
    )

    return MLPBCModule(
        net=net,
        obs_dim=obs_dim,
        act_dim=act_dim,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        action_clip=action_clip,
        obs_noise=obs_noise,
    )
