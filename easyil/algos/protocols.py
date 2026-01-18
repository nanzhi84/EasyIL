"""Protocol definitions for IL algorithms (JAX)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import jax.numpy as jnp
from flax.training import train_state


@runtime_checkable
class ILModule(Protocol):
    """Base protocol for all imitation learning modules.

    This is the minimal interface that all IL algorithms must implement.
    Specific algorithm types (BC, Actor-Critic IL) extend this with
    additional requirements.
    """

    obs_dim: int
    act_dim: int

    def init_state(self, rng_key: jnp.ndarray, **kwargs: Any) -> None:
        """Initialize training state(s)."""
        ...

    def train_step(
        self,
        state: Any,
        batch: Dict[str, jnp.ndarray],
        rng_key: jnp.ndarray,
    ) -> Tuple[Any, Dict[str, float]]:
        """Single training step.

        Returns:
            Updated state and dict of metrics (e.g., {"loss": 0.5}).
        """
        ...

    def save(self, path: str) -> None:
        """Save model parameters."""
        ...

    def load(self, path: str) -> None:
        """Load model parameters."""
        ...


@runtime_checkable
class BCModule(ILModule, Protocol):
    """Protocol for behavior cloning modules.

    BC algorithms learn from (obs_history, action_chunk) pairs
    using supervised learning. They predict action sequences
    given observation history.
    """

    obs_horizon: int
    action_horizon: int
    state: Optional[train_state.TrainState]

    def init_state(
        self,
        rng_key: jnp.ndarray,
        learning_rate: float,
        weight_decay: float = 0.0,
        ema_decay: float = 0.999,
    ) -> None:
        """Initialize training state with optimizer."""
        ...

    def train_step(
        self,
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
        rng_key: jnp.ndarray,
    ) -> Tuple[train_state.TrainState, float]:
        """Single training step.

        Args:
            state: Current training state.
            batch: Dict with 'obs' (B, obs_horizon, obs_dim) and 'actions' (B, action_horizon, act_dim).
            rng_key: JAX random key.

        Returns:
            Updated state and scalar loss value.
        """
        ...

    def sample_actions(
        self,
        rng_key_or_obs: jnp.ndarray,
        obs_or_use_ema: Any = None,
        use_ema: bool = True,
    ) -> jnp.ndarray:
        """Sample action chunks from the policy.

        Note: Signature varies between deterministic (MLP) and stochastic (Diffusion) policies.
        """
        ...


@runtime_checkable
class ActorCriticILModule(ILModule, Protocol):
    """Protocol for actor-critic based IL modules (IQ-Learn, GAIL, AIRL, etc.).

    These algorithms learn from (s, a, s', done) transitions and maintain
    separate actor and critic networks.
    """

    gamma: float
    tau: float
    actor_state: Optional[train_state.TrainState]
    critic_state: Optional[train_state.TrainState]

    def init_state(
        self,
        rng_key: jnp.ndarray,
        actor_lr: float,
        critic_lr: float,
        **kwargs: Any,
    ) -> None:
        """Initialize actor and critic training states."""
        ...

    def train_step(
        self,
        state: Any,
        batch: Dict[str, jnp.ndarray],
        rng_key: jnp.ndarray,
    ) -> Tuple[Any, Dict[str, float]]:
        """Single training step updating both actor and critic.

        Args:
            state: Combined state (actor + critic).
            batch: Dict with 'obs', 'actions', 'next_obs', 'dones'.
            rng_key: JAX random key.

        Returns:
            Updated state and dict of metrics {"actor_loss": ..., "critic_loss": ...}.
        """
        ...

    def select_action(
        self,
        obs: jnp.ndarray,
        rng_key: jnp.ndarray,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        """Select action for given observation.

        Args:
            obs: Observation array (obs_dim,) or (B, obs_dim).
            rng_key: JAX random key.
            deterministic: If True, return mean action without noise.

        Returns:
            Action array (act_dim,) or (B, act_dim).
        """
        ...

    def update_critic(
        self,
        batch: Dict[str, jnp.ndarray],
        rng_key: jnp.ndarray,
    ) -> Tuple[train_state.TrainState, float]:
        """Update critic network.

        Returns:
            Updated critic state and critic loss.
        """
        ...

    def update_actor(
        self,
        batch: Dict[str, jnp.ndarray],
        rng_key: jnp.ndarray,
    ) -> Tuple[train_state.TrainState, float]:
        """Update actor network.

        Returns:
            Updated actor state and actor loss.
        """
        ...

    def soft_update_target(self) -> None:
        """Soft update target network parameters: target = tau * online + (1 - tau) * target."""
        ...
