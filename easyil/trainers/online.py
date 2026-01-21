"""Online trainer for imitation learning algorithms (GAIL, SQIL)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from easyil.algos import build_algo
from easyil.buffers import UnifiedBuffer, BufferConfig, build_sampler
from easyil.callbacks import OnlineCallback
from easyil.collectors import OfflineCollector, OnlineCollector
from easyil.envs import make_env, infer_env_dims
from easyil.loggers import build_logger
from easyil.utils import pick_device


class OnlineTrainer:
    """Trainer for online imitation learning algorithms (GAIL, SQIL)."""

    def __init__(self, cfg: DictConfig, output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        self.seed = cfg.seed

        self.device = pick_device(str(cfg.train.get("device", "auto")))
        self.logger = build_logger(cfg.logger, output_dir, cfg)

        # Create environments
        self.train_env = make_env(
            cfg.env,
            output_dir,
            seed=cfg.seed,
            n_envs=1,
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

        obs_dim, act_dim, _ = infer_env_dims(self.train_env)

        # Build buffer with sampler from config
        sampler_cfg = cfg.train.get("sampler")
        sampler = build_sampler(sampler_cfg) if sampler_cfg else None
        buffer_cfg = BufferConfig(
            capacity=int(cfg.train.get("buffer_size", 1_000_000)),
            device=str(self.device),
        )
        self.buffer = UnifiedBuffer(buffer_cfg, sampler=sampler)

        # Load expert data
        self.obs_norm_stats: Optional[dict] = None
        if cfg.train.get("expert_data"):
            collector = OfflineCollector(cfg.train.expert_data.path, source_label="expert")
            n_expert, self.obs_norm_stats = collector.collect_to_buffer(
                self.buffer,
                num_trajs=cfg.train.expert_data.get("num_trajs"),
                obs_normalize=cfg.train.expert_data.get("obs_normalize", False),
            )
            print(f"Loaded {n_expert} expert samples")

            if self.obs_norm_stats is not None:
                np.savez(
                    self.output_dir / "obs_norm_stats.npz",
                    mean=self.obs_norm_stats["mean"],
                    std=self.obs_norm_stats["std"],
                )

        # Build algorithm
        self.algo = build_algo(cfg.algo, obs_dim, act_dim, str(self.device))

        # Build online collector
        self.collector = OnlineCollector(
            env=self.train_env,
            actor=self.algo.inference_actor(),
            device=self.device,
            source_label="policy",
            obs_norm_stats=self.obs_norm_stats,
        )

        # Build callback
        self.callback = OnlineCallback(
            logger=self.logger,
            output_dir=self.output_dir,
            eval_env=self.eval_env,
            device=self.device,
            train_cfg=cfg.train,
            algo_cfg=cfg.algo,
            env_cfg=cfg.env,
            obs_norm_stats=self.obs_norm_stats,
        )

        # Training params
        self.total_timesteps = int(cfg.train.total_timesteps)
        self.batch_size = int(cfg.train.batch_size)
        self.update_after = int(cfg.train.get("update_after", 1000))
        self.update_every = int(cfg.train.get("update_every", 50))
        self.collect_per_step = int(cfg.train.get("collect_per_step", 1))

    def train(self) -> None:
        """Run training loop."""
        timesteps = 0
        update_count = 0

        pbar = tqdm(
            total=self.total_timesteps,
            disable=not self.cfg.train.get("progress_bar", True),
            dynamic_ncols=True,
        )

        while timesteps < self.total_timesteps:
            # Collect experience
            collect_stats = self.collector.collect_steps(
                self.buffer,
                n_steps=self.collect_per_step,
                deterministic=False,
            )
            timesteps += collect_stats["n_steps"]
            pbar.update(collect_stats["n_steps"])

            # Update policy
            if timesteps >= self.update_after and timesteps % self.update_every == 0:
                batch = self.buffer.sample(self.batch_size)
                metrics = self.algo.update(batch)
                update_count += 1

                self.callback.log_train(timesteps, metrics)
                pbar.set_postfix(**{k: f"{v:.3f}" for k, v in metrics.items()})

            # Evaluation and checkpointing
            self.callback.maybe_eval(timesteps, self.algo, seed=self.seed)
            self.callback.save_checkpoint(timesteps, self.algo)

        self.callback.on_training_end(timesteps, self.algo, seed=self.seed)
        pbar.close()

    def save(self) -> None:
        """Save final model."""
        torch.save(
            self.algo.inference_actor().state_dict(),
            self.output_dir / "final_model.pt",
        )
        torch.save(self.algo.state_dict(), self.output_dir / "algo_state.pt")

    def close(self) -> None:
        """Clean up resources."""
        self.train_env.close()
        self.eval_env.close()
        self.logger.finish()
