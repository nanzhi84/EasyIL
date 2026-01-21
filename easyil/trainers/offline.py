"""Offline trainer for behavior cloning algorithms."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from easyil.algos import build_algo
from easyil.buffers import UnifiedBuffer, BufferConfig, build_sampler
from easyil.callbacks import OfflineCallback
from easyil.collectors import OfflineCollector
from easyil.envs import make_env, infer_env_dims
from easyil.loggers import build_logger
from easyil.utils import pick_device


class OfflineTrainer:
    """Trainer for offline learning algorithms (BC)."""

    def __init__(self, cfg: DictConfig, output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        self.seed = cfg.seed

        self.device = pick_device(str(cfg.train.get("device", "auto")))
        self.logger = build_logger(cfg.logger, output_dir, cfg)

        # Create eval env to infer dimensions
        self.eval_env = make_env(cfg.env, output_dir, seed=cfg.seed + 1)
        obs_dim, act_dim, _ = infer_env_dims(self.eval_env)

        # Build buffer with sampler from config
        sampler_cfg = cfg.train.get("sampler")
        sampler = build_sampler(sampler_cfg) if sampler_cfg else None
        buffer_cfg = BufferConfig(
            capacity=int(cfg.train.get("buffer_size", 0)),
            device=str(self.device),
        )
        self.buffer = UnifiedBuffer(buffer_cfg, sampler=sampler)

        # Load offline data
        collector = OfflineCollector(cfg.train.dataset.path)
        n_samples, self.obs_norm_stats = collector.collect_to_buffer(
            self.buffer,
            num_trajs=cfg.train.dataset.get("num_trajs"),
            obs_normalize=cfg.train.dataset.get("obs_normalize", False),
        )
        print(f"Loaded {n_samples} samples from {cfg.train.dataset.path}")

        # Save norm stats if present
        if self.obs_norm_stats is not None:
            np.savez(
                self.output_dir / "obs_norm_stats.npz",
                mean=self.obs_norm_stats["mean"],
                std=self.obs_norm_stats["std"],
            )

        # Build algorithm
        self.algo = build_algo(cfg.algo, obs_dim, act_dim, str(self.device))

        # Build callback
        self.callback = OfflineCallback(
            logger=self.logger,
            output_dir=self.output_dir,
            eval_env=self.eval_env,
            device=self.device,
            train_cfg=cfg.train,
            algo_cfg=cfg.algo,
            env_cfg=cfg.env,
            obs_norm_stats=self.obs_norm_stats,
        )

        self.total_updates = int(cfg.train.total_updates)
        self.batch_size = int(cfg.train.batch_size)

    def train(self) -> None:
        """Run training loop."""
        obs_horizon = self.cfg.algo.get("obs_horizon", 1)
        action_horizon = self.cfg.algo.get("action_horizon", 1)

        pbar = tqdm(
            range(1, self.total_updates + 1),
            disable=not self.cfg.train.get("progress_bar", True),
            dynamic_ncols=True,
        )

        for update in pbar:
            # Sample batch with chunking
            batch = self.buffer.sample_chunk(self.batch_size, obs_horizon, action_horizon)

            # Update algorithm
            metrics = self.algo.update(batch)

            # Callbacks
            self.callback.log_train(update, metrics)
            self.callback.save_checkpoint(update, self.algo)
            self.callback.maybe_eval(update, self.algo, seed=self.seed)

            pbar.set_postfix(loss=metrics.get("loss", 0.0))

        self.callback.on_training_end(self.total_updates, self.algo, seed=self.seed)

    def save(self) -> None:
        """Save final model."""
        torch.save(
            self.algo.inference_actor().state_dict(),
            self.output_dir / "final_model.pt",
        )

    def close(self) -> None:
        """Clean up resources."""
        self.eval_env.close()
        self.logger.finish()
