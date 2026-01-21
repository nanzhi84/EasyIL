"""Actor modules for policy networks."""
from __future__ import annotations

from typing import Any, Callable, Dict

from omegaconf import DictConfig

from easyil.actors.chunk import ChunkActor, ChunkActorConfig
from easyil.actors.gaussian import GaussianActor, GaussianActorConfig

ACTOR_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_actor(name: str) -> Callable:
    """Register an actor builder."""

    def decorator(fn: Callable) -> Callable:
        ACTOR_REGISTRY[name] = fn
        return fn

    return decorator


def build_actor(cfg: DictConfig, obs_dim: int, act_dim: int) -> Any:
    """Build an actor from config."""
    actor_type = str(cfg.type)
    if actor_type not in ACTOR_REGISTRY:
        raise ValueError(f"Unknown actor: {actor_type}. Available: {list(ACTOR_REGISTRY.keys())}")
    return ACTOR_REGISTRY[actor_type](cfg, obs_dim, act_dim)


@register_actor("chunk")
def _build_chunk_actor(cfg: DictConfig, obs_dim: int, act_dim: int) -> ChunkActor:
    config = ChunkActorConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        obs_horizon=int(cfg.get("obs_horizon", 1)),
        action_horizon=int(cfg.get("action_horizon", 1)),
        hidden=int(cfg.get("hidden", 512)),
        depth=int(cfg.get("depth", 4)),
        dropout=float(cfg.get("dropout", 0.0)),
        action_clip=float(cfg.get("action_clip", 1.0)),
    )
    return ChunkActor(config)


@register_actor("gaussian")
def _build_gaussian_actor(cfg: DictConfig, obs_dim: int, act_dim: int) -> GaussianActor:
    config = GaussianActorConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden=int(cfg.get("hidden", 256)),
        depth=int(cfg.get("depth", 2)),
        log_std_min=float(cfg.get("log_std_min", -20.0)),
        log_std_max=float(cfg.get("log_std_max", 2.0)),
        action_clip=float(cfg.get("action_clip", 1.0)),
    )
    return GaussianActor(config)


__all__ = [
    "ChunkActor",
    "ChunkActorConfig",
    "GaussianActor",
    "GaussianActorConfig",
    "build_actor",
    "register_actor",
    "ACTOR_REGISTRY",
]
