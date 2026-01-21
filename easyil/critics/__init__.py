"""Critic modules for value networks and discriminators."""
from __future__ import annotations

from typing import Any, Callable, Dict

from omegaconf import DictConfig

from easyil.critics.qvalue import QCritic, QCriticConfig
from easyil.critics.discriminator import Discriminator, DiscriminatorConfig

CRITIC_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_critic(name: str) -> Callable:
    """Register a critic builder."""

    def decorator(fn: Callable) -> Callable:
        CRITIC_REGISTRY[name] = fn
        return fn

    return decorator


def build_critic(cfg: DictConfig, obs_dim: int, act_dim: int) -> Any:
    """Build a critic from config."""
    critic_type = str(cfg.type)
    if critic_type not in CRITIC_REGISTRY:
        raise ValueError(f"Unknown critic: {critic_type}. Available: {list(CRITIC_REGISTRY.keys())}")
    return CRITIC_REGISTRY[critic_type](cfg, obs_dim, act_dim)


@register_critic("q")
def _build_q_critic(cfg: DictConfig, obs_dim: int, act_dim: int) -> QCritic:
    config = QCriticConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden=int(cfg.get("hidden", 256)),
        depth=int(cfg.get("depth", 2)),
    )
    return QCritic(config)


@register_critic("discriminator")
def _build_discriminator(cfg: DictConfig, obs_dim: int, act_dim: int) -> Discriminator:
    config = DiscriminatorConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden=int(cfg.get("hidden", 256)),
        depth=int(cfg.get("depth", 2)),
    )
    return Discriminator(config)


__all__ = [
    "QCritic",
    "QCriticConfig",
    "Discriminator",
    "DiscriminatorConfig",
    "build_critic",
    "register_critic",
    "CRITIC_REGISTRY",
]
