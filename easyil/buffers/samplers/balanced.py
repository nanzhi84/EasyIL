"""Balanced sampler for GAIL/SQIL-style training."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from easyil.buffers.samplers import register_sampler

if TYPE_CHECKING:
    from easyil.buffers.base import UnifiedBuffer


class BalancedSampler:
    """
    Balanced sampling: maintains ratio between expert and policy data.

    Used for GAIL, SQIL where we need to compare expert vs policy.
    """

    def __init__(self, expert_ratio: float = 0.5) -> None:
        self.expert_ratio = expert_ratio
        self.buffer: "UnifiedBuffer" = None  # type: ignore

    def set_buffer(self, buffer: "UnifiedBuffer") -> None:
        self.buffer = buffer

    def sample_indices(self, batch_size: int) -> np.ndarray:
        source = self.buffer.get_array("source")

        expert_mask = source == "expert"
        policy_mask = ~expert_mask

        expert_indices = np.where(expert_mask)[0]
        policy_indices = np.where(policy_mask)[0]

        n_expert = int(batch_size * self.expert_ratio)
        n_policy = batch_size - n_expert

        if len(expert_indices) == 0:
            return np.random.choice(policy_indices, size=batch_size, replace=True)
        if len(policy_indices) == 0:
            return np.random.choice(expert_indices, size=batch_size, replace=True)

        sampled_expert = np.random.choice(expert_indices, size=n_expert, replace=True)
        sampled_policy = np.random.choice(policy_indices, size=n_policy, replace=True)

        indices = np.concatenate([sampled_expert, sampled_policy])
        np.random.shuffle(indices)
        return indices


@register_sampler("balanced")
def _build_balanced(expert_ratio: float = 0.5, **kwargs) -> BalancedSampler:
    return BalancedSampler(expert_ratio=expert_ratio)
