"""Algorithm implementations for EasyIL."""
from __future__ import annotations

from typing import Any, Callable, Dict

from omegaconf import DictConfig

# Protocol definitions
from easyil.algos.protocols import BasePolicy, BCModule

# Algorithm registry
ALGO_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_algo(name: str | None = None) -> Callable:
    """Register an algorithm builder function."""

    def decorator(fn: Callable) -> Callable:
        key = name if name is not None else fn.__name__
        ALGO_REGISTRY[key] = fn
        return fn

    return decorator


def build_algo(cfg: DictConfig, eval_env: Any, output_dir: str) -> Any:
    """Build an algorithm instance from config + registry."""
    algo_name = str(cfg.name)
    if algo_name not in ALGO_REGISTRY:
        available = ", ".join(sorted(ALGO_REGISTRY.keys()))
        raise KeyError(f"Unknown algo '{algo_name}'. Available: {available}")
    return ALGO_REGISTRY[algo_name](cfg, eval_env, output_dir)


# Import algorithm modules to trigger registration
from easyil.algos.diffusion_bc import diffusion_bc  # noqa: F401, E402
from easyil.algos.mlp_bc import mlp_bc  # noqa: F401, E402
from easyil.algos.sac import sac  # noqa: F401, E402

__all__ = [
    "ALGO_REGISTRY",
    "build_algo",
    "register_algo",
    "BasePolicy",
    "BCModule",
    "diffusion_bc",
    "mlp_bc",
    "sac",
]
