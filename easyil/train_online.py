"""Online RL training runner (SAC, etc.)."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from easyil.algos import build_algo
from easyil.envs import make_env
from easyil.loggers import build_logger
from easyil.trainers import OnlineTrainer


def run(cfg: DictConfig, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
