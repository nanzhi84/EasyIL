"""Diffusion schedulers (JAX)."""
from typing import Any

from omegaconf import DictConfig

from easyil.algos.diffusion_bc.schedulers.base import BaseScheduler
from easyil.algos.diffusion_bc.schedulers.ddim import DDIMScheduler
from easyil.algos.diffusion_bc.schedulers.ddpm import DDPMScheduler

__all__ = [
    "BaseScheduler",
    "DDPMScheduler",
    "DDIMScheduler",
    "build_scheduler",
]


def build_scheduler(cfg: DictConfig) -> BaseScheduler:
    """Build a scheduler from config."""
    sched_type = str(cfg.type).lower()

    train_steps = int(cfg.get("train_steps", 100))
    beta_start = float(cfg.get("beta_start", 1e-4))
    beta_end = float(cfg.get("beta_end", 2e-2))

    if sched_type == "ddpm":
        return DDPMScheduler(
            train_steps=train_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
    elif sched_type == "ddim":
        eta = float(cfg.get("eta", 0.0))
        return DDIMScheduler(
            train_steps=train_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            eta=eta,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")
