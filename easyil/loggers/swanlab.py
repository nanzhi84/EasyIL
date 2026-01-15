from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import swanlab
from omegaconf import DictConfig, OmegaConf

from easyil.loggers import Logger, register_logger


def _to_float(x: Any) -> float | None:
    """Convert to float, filtering out non-numeric types."""
    if x is None or isinstance(x, (bool, str)):
        return None
    val = x.item() if hasattr(x, "item") else x
    fval = float(val)
    return fval if np.isfinite(fval) else None


class SwanLabLogger(Logger):
    """SwanLab logger implementation."""

    def __init__(self, run: Any):
        self._run = run

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self._run is None:
            return
        clean = {k: fv for k, v in metrics.items() if (fv := _to_float(v)) is not None}
        if clean:
            swanlab.log(clean, step=step)

    def finish(self) -> None:
        if self._run is not None:
            swanlab.finish()


@register_logger
def swanlab(cfg: DictConfig, output_dir: Path, full_cfg: DictConfig) -> SwanLabLogger:
    """Build a SwanLabLogger from config."""
    exp_name = cfg.get("experiment_name") or f"{full_cfg.algo.name}-{full_cfg.env.id}-{output_dir.name}"

    # Merge user tags with auto-generated algo/env tags
    user_tags = list(cfg.get("tags") or [])
    auto_tags = [str(full_cfg.algo.name), str(full_cfg.env.id)]
    tags = sorted(set(user_tags + auto_tags))

    run = swanlab.init(
        project=str(cfg.project),
        experiment_name=str(exp_name),
        description=cfg.get("description"),
        group=cfg.get("group"),
        tags=tags,
        config=OmegaConf.to_container(full_cfg, resolve=True),
        logdir=str(cfg.logdir),
        mode=str(cfg.mode),
    )
    return SwanLabLogger(run)
