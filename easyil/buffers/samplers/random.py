"""Random sampler implementation."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from easyil.buffers.samplers import register_sampler

if TYPE_CHECKING:
    from easyil.buffers.base import UnifiedBuffer


class RandomSampler:
    """Uniform random sampling."""

    def __init__(self) -> None:
        self.buffer: "UnifiedBuffer" = None  # type: ignore

    def set_buffer(self, buffer: "UnifiedBuffer") -> None:
        self.buffer = buffer

    def sample_indices(self, batch_size: int) -> np.ndarray:
        return np.random.randint(0, len(self.buffer), size=batch_size)


@register_sampler("random")
def _build_random(**kwargs) -> RandomSampler:
    return RandomSampler()
