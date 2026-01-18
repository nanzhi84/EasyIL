"""RL algorithms for training expert policies."""
from __future__ import annotations

from typing import Any, Callable, Dict

from omegaconf import DictConfig

# Expert RL algorithm registry
EXPERT_RL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_expert_rl(name: str | None = None) -> Callable:
    """Register an expert RL algorithm builder function."""

    def decorator(fn: Callable) -> Callable:
        key = name if name is not None else fn.__name__
        EXPERT_RL_REGISTRY[key] = fn
        return fn

    return decorator


def build_expert_algo(cfg: DictConfig, env: Any, output_dir: str) -> Any:
    """Build an expert RL algorithm instance from config + registry."""
    algo_name = str(cfg.name)
    return EXPERT_RL_REGISTRY[algo_name](cfg, env, output_dir)


# Import algorithm modules to trigger registration
from collect_expert.rl.sac import sac  # noqa: F401, E402

__all__ = [
    "EXPERT_RL_REGISTRY",
    "build_expert_algo",
    "register_expert_rl",
    "sac",
]
