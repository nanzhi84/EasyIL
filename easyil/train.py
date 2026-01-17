"""Training entry point (JAX)."""
from __future__ import annotations

import os
import random
from pathlib import Path

import hydra
import jax
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from easyil.trainers import build_trainer


def _seed_everything(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Note: JAX uses explicit RNG keys, no global seed needed


@hydra.main(config_path="conf", config_name="diffusion_bc", version_base="1.3")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    _seed_everything(cfg.seed)

    # Print JAX backend info
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    trainer = build_trainer(cfg, output_dir)
    trainer.train()
    trainer.save()
    trainer.close()


if __name__ == "__main__":
    main()
