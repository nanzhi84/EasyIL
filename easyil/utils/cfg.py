from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def save_resolved_config(cfg: DictConfig, output_dir: Path) -> None:
    """Save resolved Hydra config to output directory."""
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg), encoding="utf-8"
    )
