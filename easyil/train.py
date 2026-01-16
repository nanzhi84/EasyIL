from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call
from omegaconf import DictConfig

from easyil.utils.cfg import save_resolved_config, seed_everything


@hydra.main(config_path="conf", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_resolved_config(cfg, output_dir)
    seed_everything(cfg.seed)

    call(cfg.runner, cfg=cfg, output_dir=output_dir)


if __name__ == "__main__":
    main()
