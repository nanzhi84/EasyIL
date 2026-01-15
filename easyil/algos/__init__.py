"""Algorithm implementations for EasyIL."""
from __future__ import annotations

# Re-export registry functions from core
from easyil.core.registry import ALGO_REGISTRY, build_algo, register_algo

# Import algorithm modules to trigger registration
from easyil.algos.diffusion_bc import diffusion_bc  # noqa: F401
from easyil.algos.mlp_bc import mlp_bc  # noqa: F401
from easyil.algos.sac import sac  # noqa: F401

__all__ = ["ALGO_REGISTRY", "build_algo", "register_algo", "diffusion_bc", "mlp_bc", "sac"]
