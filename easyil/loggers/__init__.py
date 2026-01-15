from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from omegaconf import DictConfig


@runtime_checkable
class Logger(Protocol):
    """Abstract logger protocol for training metrics."""

    def log(self, metrics: dict[str, Any], step: int) -> None: ...
    def finish(self) -> None: ...


class NullLogger:
    """No-op logger used when logging is disabled."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def finish(self) -> None:
        pass


LOGGER_REGISTRY: dict[str, Any] = {}


def register_logger(fn):
    """Register a logger builder function."""
    LOGGER_REGISTRY[fn.__name__] = fn
    return fn


def build_logger(logger_cfg: DictConfig, output_dir: Path, full_cfg: DictConfig) -> Logger:
    """Build a logger instance from config + registry."""
    if not bool(logger_cfg.get("enabled", False)):
        return NullLogger()

    logger_type = str(logger_cfg.get("type", "swanlab"))
    if logger_type not in LOGGER_REGISTRY:
        available = ", ".join(sorted(LOGGER_REGISTRY.keys()))
        raise KeyError(f"Unknown logger '{logger_type}'. Available: {available}")

    return LOGGER_REGISTRY[logger_type](logger_cfg, output_dir, full_cfg)


# Import logger modules so that the decorator runs at import time.
from easyil.loggers.swanlab import swanlab  # noqa: E402, F401


__all__ = ["Logger", "NullLogger", "register_logger", "build_logger", "swanlab"]
