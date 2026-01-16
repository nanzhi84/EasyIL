"""Online trainer for reinforcement learning algorithms."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from easyil.algos import build_algo
from easyil.callbacks import OnlineTrainCallback
from easyil.envs import make_env, save_vecnormalize
from easyil.loggers import build_logger


class OnlineTrainer:
    """Trainer for online RL algorithms (SAC, TD3, etc.)."""

    def __init__(self, cfg: DictConfig, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir

        self.logger = build_logger(cfg.logger, output_dir, cfg)
        self.train_env = make_env(
            cfg.env,
            output_dir,
            seed=cfg.seed,
            n_envs=cfg.env.num_envs,
            training=True,
            monitor_subdir="train",
        )
        self.eval_env = make_env(
            cfg.env,
            output_dir,
            seed=cfg.seed + 1,
            n_envs=1,
            training=False,
            monitor_subdir="eval",
        )
        self.model = build_algo(cfg.algo, self.train_env, str(output_dir))

    def train(self) -> None:
        callback = OnlineTrainCallback(
            logger=self.logger,
            output_dir=self.output_dir,
            train_env=self.train_env,
            eval_env=self.eval_env,
            train_cfg=self.cfg.train,
            env_cfg=self.cfg.env,
        )

        self.model.learn(
            total_timesteps=int(self.cfg.train.total_timesteps),
            callback=callback,
            progress_bar=bool(self.cfg.train.get("progress_bar", True)),
            log_interval=int(self.cfg.train.get("log_interval", 100)),
        )

    def save(self) -> None:
        self.model.save(str(self.output_dir / "final_model"))
        save_vecnormalize(self.train_env, self.output_dir / "vecnormalize.pkl")

    def close(self) -> None:
        self.train_env.close()
        self.eval_env.close()
        self.logger.finish()
