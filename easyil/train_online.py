"""Online RL training entry point (SAC, etc.)."""
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from easyil.algos import build_algo
from easyil.envs import make_env
from easyil.loggers import build_logger
from easyil.trainers import OnlineTrainer
from easyil.utils.cfg import save_resolved_config, seed_everything


@hydra.main(config_path="conf", config_name="train_online", version_base="1.3")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_resolved_config(cfg, output_dir)
    seed_everything(cfg.seed)

    logger = build_logger(cfg.logger, output_dir, cfg)
    train_env = make_env(cfg.env, output_dir, seed=cfg.seed, n_envs=cfg.env.num_envs, training=True, monitor_subdir="train")
    eval_env = make_env(cfg.env, output_dir, seed=cfg.seed + 1, n_envs=1, training=False, monitor_subdir="eval")
    model = build_algo(cfg.algo, train_env, str(output_dir))

    trainer = OnlineTrainer(
        model=model,
        train_env=train_env,
        eval_env=eval_env,
        logger=logger,
        output_dir=output_dir,
        train_cfg=cfg.train,
        env_cfg=cfg.env,
    )

    trainer.train()

    train_env.close()
    eval_env.close()
    logger.finish()


if __name__ == "__main__":
    main()