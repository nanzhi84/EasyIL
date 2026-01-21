"""Algorithm implementations for EasyIL."""
from __future__ import annotations

from typing import Any, Callable, Dict

from omegaconf import DictConfig

ALGO_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_algo(name: str, fn: Callable) -> None:
    """Register an algorithm builder function."""
    ALGO_REGISTRY[name] = fn


def build_algo(cfg: DictConfig, obs_dim: int, act_dim: int, device: str = "cpu") -> Any:
    """Build an algorithm instance from config."""
    _ensure_registered()

    algo_name = str(cfg.name)
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algo_name}. Available: {list(ALGO_REGISTRY.keys())}")
    return ALGO_REGISTRY[algo_name](cfg, obs_dim, act_dim, device)


_registered = False


def _ensure_registered() -> None:
    """Ensure all algorithms are registered (called lazily)."""
    global _registered
    if _registered:
        return
    _registered = True

    from easyil.algos.bc import _build_bc
    from easyil.algos.gail import _build_gail
    from easyil.algos.sqil import _build_sqil

    register_algo("bc", _build_bc)
    register_algo("gail", _build_gail)
    register_algo("sqil", _build_sqil)


# Re-export for convenience (lazy imports)
def __getattr__(name: str) -> Any:
    if name in ("BC", "BCConfig"):
        from easyil.algos.bc import BC, BCConfig
        return BC if name == "BC" else BCConfig
    if name in ("GAIL", "GAILConfig"):
        from easyil.algos.gail import GAIL, GAILConfig
        return GAIL if name == "GAIL" else GAILConfig
    if name in ("SQIL", "SQILConfig"):
        from easyil.algos.sqil import SQIL, SQILConfig
        return SQIL if name == "SQIL" else SQILConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ALGO_REGISTRY",
    "build_algo",
    "register_algo",
    "BC",
    "BCConfig",
    "GAIL",
    "GAILConfig",
    "SQIL",
    "SQILConfig",
]
