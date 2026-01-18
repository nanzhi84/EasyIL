"""Offline trainer for behavior cloning algorithms (JAX)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from easyil.algos import build_algo
from easyil.callbacks import OfflineTrainCallback
from easyil.datasets import ILDataset, TrajectoryDataLoader, build_dataset
from easyil.envs import VecEnvProtocol, make_env, save_vecnormalize
from easyil.loggers import build_logger

if TYPE_CHECKING:
    from easyil.algos import BCModule


class OfflineTrainer:
    """Trainer for offline learning algorithms (Diffusion BC, MLP BC, etc.) using JAX."""

    eval_env: VecEnvProtocol

    def __init__(self, cfg: DictConfig, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir
        self.seed = cfg.seed

        self.logger = build_logger(cfg.logger, output_dir, cfg)
        self.eval_env = make_env(cfg.env, output_dir, seed=cfg.seed + 1)

        self.module = self._build_module()
        self.dataloader, self.obs_norm_stats = self._build_dataloader()
        self.callback = self._build_callback()

        self.total_updates = int(cfg.train.total_updates)
        self.rng_key = jax.random.PRNGKey(self.seed)

    def _build_module(self) -> "BCModule":
        module = build_algo(self.cfg.algo, self.eval_env, str(self.output_dir))

        # Initialize training state
        self.rng_key, init_key = jax.random.split(jax.random.PRNGKey(self.seed))
        module.init_state(
            rng_key=init_key,
            learning_rate=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.get("weight_decay", 0.0),
            ema_decay=self.cfg.algo.get("ema_decay", 0.999),
        )

        return module

    def _build_dataloader(self) -> tuple[TrajectoryDataLoader, Dict[str, np.ndarray] | None]:
        dataset_cfg = self.cfg.train.dataset
        ds = build_dataset(
            dataset_cfg,
            obs_horizon=self.cfg.algo.obs_horizon,
            action_horizon=self.cfg.algo.action_horizon,
        )

        obs_norm_stats = self._extract_obs_norm_stats(ds)

        dl = TrajectoryDataLoader(
            ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            seed=self.seed,
        )

        return dl, obs_norm_stats

    def _extract_obs_norm_stats(self, ds: ILDataset) -> Dict[str, np.ndarray] | None:
        """Extract and save observation normalization stats from dataset."""
        if ds.obs_mean is None:
            return None

        obs_norm_stats = {"mean": ds.obs_mean, "std": ds.obs_std}
        np.savez(self.output_dir / "obs_norm_stats.npz", mean=ds.obs_mean, std=ds.obs_std)
        return obs_norm_stats

    def _build_callback(self) -> OfflineTrainCallback:
        return OfflineTrainCallback(
            logger=self.logger,
            output_dir=self.output_dir,
            eval_env=self.eval_env,
            train_cfg=self.cfg.train,
            algo_cfg=self.cfg.algo,
            env_cfg=self.cfg.env,
            obs_norm_stats=self.obs_norm_stats,
        )

    def _infinite_dataloader(self) -> Iterator[Dict[str, np.ndarray]]:
        while True:
            yield from self.dataloader

    def train(self) -> None:
        data_iter = self._infinite_dataloader()
        pbar = tqdm(
            range(1, self.total_updates + 1),
            disable=not self.cfg.train.get("progress_bar", True),
            dynamic_ncols=True,
        )

        state = self.module.state

        for update in pbar:
            # Get batch and convert to JAX arrays
            batch_np = next(data_iter)
            batch = {k: jnp.array(v) for k, v in batch_np.items()}

            # Training step
            self.rng_key, step_key = jax.random.split(self.rng_key)
            state, loss = self.module.train_step(state, batch, step_key)
            self.module.state = state

            loss_val = float(loss)
            lr = self.cfg.train.learning_rate

            self.callback.log_train(update, loss_val, lr)
            self.callback.save_checkpoint(update, self.module)
            self.callback.maybe_eval(update, self.module, seed=self.seed, rng_key=self.rng_key)
            pbar.set_postfix(loss=loss_val)

        self.callback.on_training_end(self.total_updates, self.module, seed=self.seed, rng_key=self.rng_key)

    def save(self) -> None:
        self.module.save(str(self.output_dir / "final_model.pkl"))
        save_vecnormalize(self.eval_env, self.output_dir / "vecnormalize.pkl")

    def close(self) -> None:
        self.eval_env.close()
        self.logger.finish()
