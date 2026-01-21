"""Pluggable sampling strategies for UnifiedBuffer."""
from __future__ import annotations

from typing import Any, Callable, Dict, Protocol, TYPE_CHECKING

import numpy as np
from omegaconf import DictConfig

if TYPE_CHECKING:
    from easyil.buffers.base import UnifiedBuffer


class Sampler(Protocol):
    """Sampler protocol: defines how to sample indices from buffer."""

    def set_buffer(self, buffer: "UnifiedBuffer") -> None: ...

    def sample_indices(self, batch_size: int) -> np.ndarray: ...


SAMPLER_REGISTRY: Dict[str, Callable[..., Sampler]] = {}


def register_sampler(name: str) -> Callable:
    """Register a sampler builder."""

    def decorator(fn: Callable[..., Sampler]) -> Callable[..., Sampler]:
        SAMPLER_REGISTRY[name] = fn
        return fn

    return decorator


def build_sampler(cfg: DictConfig | Dict[str, Any] | None) -> Sampler:
    """Build a sampler from config."""
    # Lazy import to trigger registration
    from easyil.buffers.samplers import random, chunk, balanced  # noqa: F401

    if cfg is None:
        return SAMPLER_REGISTRY["random"]()

    if isinstance(cfg, DictConfig):
        cfg = dict(cfg)

    sampler_type = cfg.pop("type", "random")
    if sampler_type not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler: {sampler_type}. Available: {list(SAMPLER_REGISTRY.keys())}")

    return SAMPLER_REGISTRY[sampler_type](**cfg)


__all__ = [
    "Sampler",
    "SAMPLER_REGISTRY",
    "register_sampler",
    "build_sampler",
]
