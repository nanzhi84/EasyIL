from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from easyil.callbacks import OnlineTrainCallback
from easyil.envs import save_vecnormalize
from easyil.loggers import Logger


class OnlineTrainer:
    """Trainer for online RL algorithms (SAC, TD3, etc.)."""

    def __init__(
        self,
        model: Any,
        train_env: Any,
        eval_env: Any,
        logger: Logger,
        output_dir: Path,
        train_cfg: DictConfig,
        env_cfg: DictConfig,
    ):
        self.model = model
        self.train_env = train_env
        self.eval_env = eval_env
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.train_cfg = train_cfg
        self.env_cfg = env_cfg

    def train(self) -> None:
        callback = OnlineTrainCallback(
            logger=self.logger,
            output_dir=self.output_dir,
            train_env=self.train_env,
            eval_env=self.eval_env,
            train_cfg=self.train_cfg,
            env_cfg=self.env_cfg,
        )

        self.model.learn(
            total_timesteps=int(self.train_cfg.total_timesteps),
            callback=callback,
            progress_bar=bool(self.train_cfg.get("progress_bar", True)),
            log_interval=int(self.train_cfg.get("log_interval", 100)),
        )

        self.model.save(str(self.output_dir / "final_model"))
        save_vecnormalize(self.train_env, self.output_dir / "vecnormalize.pkl")
