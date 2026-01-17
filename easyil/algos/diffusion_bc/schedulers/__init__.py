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
    if "num_inference_steps" not in cfg or cfg.num_inference_steps is None:
        raise ValueError("scheduler.num_inference_steps is required and must be > 0.")
    num_inference_steps = int(cfg.num_inference_steps)

    if sched_type == "ddpm":
        scheduler = DDPMScheduler(
            train_steps=train_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        scheduler.set_inference_steps(num_inference_steps)
        return scheduler
    elif sched_type == "ddim":
        eta = float(cfg.get("eta", 0.0))
        scheduler = DDIMScheduler(
            train_steps=train_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            eta=eta,
        )
        scheduler.set_inference_steps(num_inference_steps)
        return scheduler
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")
