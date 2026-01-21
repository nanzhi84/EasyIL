"""Data collectors for offline and online learning."""
from __future__ import annotations

from easyil.collectors.offline import OfflineCollector, load_npz
from easyil.collectors.online import OnlineCollector

__all__ = [
    "OfflineCollector",
    "OnlineCollector",
    "load_npz",
]
