"""Expert data collection entry point."""
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from easyil.data_collection.collector import ExpertCollector, load_model_for_collection
from easyil.utils.cfg import save_resolved_config


@hydra.main(config_path="conf", config_name="collect", version_base="1.3")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_resolved_config(cfg, output_dir)

    model_path = Path(cfg.model_path)
    vecnormalize_path = model_path.parent / "vecnormalize.pkl" if cfg.get("use_vecnormalize", True) else None

    model = load_model_for_collection(
        model_path=model_path,
        env_cfg=cfg.env,
        output_dir=output_dir,
        seed=cfg.seed,
    )

    collector = ExpertCollector(
        model=model,
        env_cfg=cfg.env,
        output_dir=output_dir,
        vecnormalize_path=vecnormalize_path,
    )

    data = collector.collect(
        n_episodes=cfg.n_episodes,
        deterministic=cfg.get("deterministic", True),
        seed=cfg.seed,
        show_progress=cfg.get("show_progress", True),
    )

    save_path = Path(cfg.output_path)
    data.save(save_path)

    print(f"\nCollection complete:")
    print(f"  Episodes: {data.n_trajectories}")
    print(f"  Transitions: {data.n_transitions}")
    print(f"  Saved to: {save_path}")


if __name__ == "__main__":
    main()
