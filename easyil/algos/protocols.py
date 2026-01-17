"""Protocol definitions for BC algorithms (JAX)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple, Union, runtime_checkable

import jax.numpy as jnp
from flax.training import train_state


@runtime_checkable
class BCModule(Protocol):
    """Protocol for behavior cloning modules."""

    obs_dim: int
    act_dim: int
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
        """Initialize training state."""
        ...

    def train_step(
        self,
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
        rng_key: jnp.ndarray,
    ) -> Tuple[train_state.TrainState, float]:
        """Single training step."""
        ...

    def save(self, path: str) -> None:
        """Save model parameters."""
        ...

    def load(self, path: str) -> None:
        """Load model parameters."""
        ...
