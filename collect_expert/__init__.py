"""Standalone expert data collection module.

This module provides tools for collecting expert demonstrations from trained
RL policies (e.g., SAC trained with SBX). It is decoupled from the main easyil
package and can be used independently.

Usage:
    python -m collect_expert model_path=path/to/model.zip n_episodes=100
"""
from __future__ import annotations

from collect_expert.collector import ExpertCollector, ExpertData, load_model_for_collection

__all__ = ["ExpertCollector", "ExpertData", "load_model_for_collection"]
