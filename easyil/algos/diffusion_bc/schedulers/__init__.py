"""Diffusion schedulers for noise scheduling."""
from __future__ import annotations

import inspect
from typing import Callable, Dict

from omegaconf import DictConfig

from easyil.algos.diffusion_bc.schedulers.base import BaseScheduler
from easyil.algos.diffusion_bc.schedulers.ddpm import DDPMScheduler
from easyil.algos.diffusion_bc.schedulers.ddim import DDIMScheduler

SCHEDULER_REGISTRY: Dict[str, Callable[..., BaseScheduler]] = {}


def register_scheduler(name: str) -> Callable[[Callable[..., BaseScheduler]], Callable[..., BaseScheduler]]:
    def decorator(cls: Callable[..., BaseScheduler]) -> Callable[..., BaseScheduler]:
        SCHEDULER_REGISTRY[name] = cls
        return cls

    return decorator


def build_scheduler(cfg: DictConfig) -> BaseScheduler:
    """Build a scheduler from config."""
    scheduler_type = str(cfg.type).lower()
    if scheduler_type not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler type: {scheduler_type} (expected: {list(SCHEDULER_REGISTRY.keys())})")

    scheduler_cls = SCHEDULER_REGISTRY[scheduler_type]

    exclude = {"type", "num_inference_steps"}
    sig = inspect.signature(scheduler_cls.__init__)
    valid_params = {p for p in sig.parameters if p != "self"}
    kwargs = {k: v for k, v in dict(cfg).items() if k not in exclude and k in valid_params}
    scheduler = scheduler_cls(**kwargs)

    if "num_inference_steps" in cfg:
        scheduler.set_inference_steps(int(cfg.num_inference_steps))

    return scheduler


# Register built-in schedulers
register_scheduler("ddpm")(DDPMScheduler)
register_scheduler("ddim")(DDIMScheduler)

__all__ = [
    "SCHEDULER_REGISTRY",
    "register_scheduler",
    "build_scheduler",
    "BaseScheduler",
    "DDPMScheduler",
    "DDIMScheduler",
]
