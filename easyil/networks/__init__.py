"""Reusable network architectures for EasyIL."""

from easyil.networks.embeddings import ObsEncoder
from easyil.networks.mlp import MLPBackbone

__all__ = [
    "ObsEncoder",
    "MLPBackbone",
]
