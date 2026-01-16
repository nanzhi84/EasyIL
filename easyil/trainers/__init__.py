from __future__ import annotations

from pathlib import Path
from typing import Protocol

from omegaconf import DictConfig

from easyil.trainers.offline import OfflineTrainer
from easyil.trainers.online import OnlineTrainer


class Trainer(Protocol):
    """Protocol for trainer classes."""

    def train(self) -> None: ...
    def save(self) -> None: ...
    def close(self) -> None: ...


def build_trainer(cfg: DictConfig, output_dir: Path) -> Trainer:
    """Build a trainer based on the training mode in config.

    Args:
        cfg: Full configuration with algo, env, train, etc.
        output_dir: Output directory for logs and checkpoints.

    Returns:
        Trainer instance (OfflineTrainer or OnlineTrainer).

    Raises:
        ValueError: If train.mode is not recognized.
    """
    mode = cfg.train.get("mode")
    if mode is None:
        raise KeyError("Missing required config: train.mode (must be 'offline' or 'online')")

    if mode == "offline":
        return OfflineTrainer(cfg, output_dir)
    if mode == "online":
        return OnlineTrainer(cfg, output_dir)

    raise ValueError(f"Unknown training mode: {mode}. Must be 'offline' or 'online'.")


__all__ = ["OfflineTrainer", "OnlineTrainer", "Trainer", "build_trainer"]
