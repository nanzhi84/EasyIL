"""Offline trainer for behavior cloning algorithms."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from easyil.algos import build_algo
from easyil.callbacks import OfflineTrainCallback
from easyil.datasets import ChunkedDataset, load_npz
from easyil.envs import make_env, save_vecnormalize
from easyil.loggers import build_logger
from easyil.utils.cfg import pick_device

if TYPE_CHECKING:
    from easyil.algos import BCModule


class OfflineTrainer:
    """Trainer for offline learning algorithms (MLP BC, etc.)."""

    def __init__(self, cfg: DictConfig, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir
        self.seed = cfg.seed

        self.device = pick_device(str(cfg.train.get("device", "auto")))
        self.logger = build_logger(cfg.logger, output_dir, cfg)
        self.eval_env = make_env(cfg.env, output_dir, seed=cfg.seed + 1)

        self.module = self._build_module()
        self.dataloader, self.obs_norm_stats = self._build_dataloader()
        self.optimizer = self._build_optimizer()
        self.callback = self._build_callback()

        self.use_amp = cfg.train.get("use_amp", True) and self.device.type == "cuda"
        self.grad_clip = cfg.train.get("grad_clip_norm", 0.0)
        self.total_updates = int(cfg.train.total_updates)

    def _build_module(self) -> "BCModule":
        module = build_algo(self.cfg.algo, self.eval_env, str(self.output_dir))
        module.to(self.device)

        use_compile = self.cfg.train.get("use_compile", True)
        if use_compile and self.device.type == "cuda":
            module.net = torch.compile(module.net, mode="reduce-overhead")
            module.ema_net = torch.compile(module.ema_net, mode="reduce-overhead")
            module.policy.net = module.ema_net

        return module

    def _build_dataloader(self) -> tuple[DataLoader, Dict[str, np.ndarray] | None]:
        dataset_cfg = self.cfg.train.dataset
        num_trajs = dataset_cfg.get("num_trajs", None)
        data = load_npz(dataset_cfg.path, num_trajs=num_trajs)

        obs_normalize = dataset_cfg.get("obs_normalize", False)
        ds = ChunkedDataset(
            data=data,
            obs_horizon=self.cfg.algo.obs_horizon,
            action_horizon=self.cfg.algo.action_horizon,
            obs_normalize=obs_normalize,
        )

        obs_norm_stats = None
        if obs_normalize and ds.obs_mean is not None:
            obs_norm_stats = {"mean": ds.obs_mean, "std": ds.obs_std}
            np.savez(self.output_dir / "obs_norm_stats.npz", mean=ds.obs_mean, std=ds.obs_std)

        num_workers = self.cfg.train.get("num_workers", 4)
        dl = DataLoader(
            ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.cfg.train.get("pin_memory", True),
            persistent_workers=num_workers > 0,
            drop_last=True,
        )

        return dl, obs_norm_stats

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.module.net.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.get("weight_decay", 0.0),
        )

    def _build_callback(self) -> OfflineTrainCallback:
        return OfflineTrainCallback(
            logger=self.logger,
            output_dir=self.output_dir,
            eval_env=self.eval_env,
            device=self.device,
            train_cfg=self.cfg.train,
            algo_cfg=self.cfg.algo,
            env_cfg=self.cfg.env,
            obs_norm_stats=self.obs_norm_stats,
        )

    def _infinite_dataloader(self):
        while True:
            yield from self.dataloader

    def train(self) -> None:
        scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        data_iter = self._infinite_dataloader()
        pbar = tqdm(
            range(1, self.total_updates + 1),
            disable=not self.cfg.train.get("progress_bar", True),
            dynamic_ncols=True,
        )

        for update in pbar:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in next(data_iter).items()}

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                loss = self.module.compute_loss(batch)

            scaler.scale(loss).backward()
            if self.grad_clip > 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.module.net.parameters(), self.grad_clip)
            scaler.step(self.optimizer)
            scaler.update()

            self.module.update_ema()

            self.callback.log_train(update, loss.item(), self.optimizer.param_groups[0]["lr"])
            self.callback.save_checkpoint(update, self.module, self.optimizer)
            self.callback.maybe_eval(update, self.module, seed=self.seed)
            pbar.set_postfix(loss=loss.item())

        self.callback.on_training_end(self.total_updates, self.module, seed=self.seed)

    def save(self) -> None:
        torch.save(self.module.ema_net.state_dict(), self.output_dir / "final_model.pt")
        save_vecnormalize(self.eval_env, self.output_dir / "vecnormalize.pkl")

    def close(self) -> None:
        self.eval_env.close()
        self.logger.finish()
