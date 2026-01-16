from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def pick_device(device: str = "auto") -> torch.device:
    """Pick torch device based on string preference."""
    want = device.lower()
    if want in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_plain_dict(cfg: Any) -> dict[str, Any]:
    """Convert OmegaConf or dict-like config into a resolved python dict."""
    if cfg is None:
        return {}
    if isinstance(cfg, (DictConfig, ListConfig)):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected dict-like config, got: {type(cfg)}")
    return dict(cfg)


def drop_none(d: dict[str, Any]) -> dict[str, Any]:
    """Drop keys with value None."""
    return {k: v for k, v in d.items() if v is not None}


def save_resolved_config(cfg: DictConfig, output_dir: Path) -> None:
    """Save resolved Hydra config to output directory."""
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg), encoding="utf-8"
    )

def seed_everything(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)