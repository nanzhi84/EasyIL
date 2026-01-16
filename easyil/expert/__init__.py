"""Expert training and data collection module.

This module provides tools for:
1. Training expert policies using RL algorithms (e.g., SAC)
2. Collecting expert demonstrations from trained policies
"""
from __future__ import annotations

from easyil.expert.collector import ExpertCollector, ExpertData, load_model_for_collection

__all__ = ["ExpertCollector", "ExpertData", "load_model_for_collection"]
