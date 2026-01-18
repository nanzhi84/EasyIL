"""Algorithm registry and factory (JAX)."""
from __future__ import annotations

from typing import Any, Callable, Dict, Union

from omegaconf import DictConfig

from easyil.algos.protocols import ActorCriticILModule, BCModule, ILModule

ALGO_REGISTRY: Dict[str, Callable[..., ILModule]] = {}


def register_algo(name: str) -> Callable[[Callable[..., ILModule]], Callable[..., ILModule]]:
    """Decorator to register an algorithm builder."""

    def decorator(fn: Callable[..., ILModule]) -> Callable[..., ILModule]:
        ALGO_REGISTRY[name] = fn
        return fn

    return decorator


def build_algo(cfg: DictConfig, eval_env: Any, output_dir: str) -> ILModule:
    """Build algorithm from config."""
    algo_name = str(cfg.name).lower()
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algo_name} (available: {list(ALGO_REGISTRY.keys())})")
    return ALGO_REGISTRY[algo_name](cfg, eval_env, output_dir)


# Import algorithms to register them
from easyil.algos.diffusion_bc.module import diffusion_bc
from easyil.algos.mlp_bc.module import mlp_bc

__all__ = [
    "ALGO_REGISTRY",
    "ActorCriticILModule",
    "BCModule",
    "ILModule",
    "build_algo",
    "diffusion_bc",
    "mlp_bc",
    "register_algo",
]
