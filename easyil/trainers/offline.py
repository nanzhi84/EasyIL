"""Offline trainer for behavior cloning algorithms."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from easyil.callbacks import OfflineTrainCallback

if TYPE_CHECKING:
    from easyil.core.base import BCModule


class OfflineTrainer:
    """Trainer for offline learning algorithms (Diffusion BC, MLP BC, etc.)."""

    def __init__(
        self,
        module: "BCModule",
        dataloader: Any,
        optimizer: torch.optim.Optimizer,
        callback: OfflineTrainCallback,
        device: torch.device,
        train_cfg: DictConfig,
        seed: int,
    ):
        self.module = module
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.callback = callback
        self.device = device
        self.train_cfg = train_cfg
        self.seed = seed

        self.use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
        self.grad_clip = train_cfg.get("grad_clip_norm", 0.0)
        self.total_updates = int(train_cfg.total_updates)

    def _infinite_dataloader(self):
        while True:
            yield from self.dataloader

    def train(self) -> None:
        scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        data_iter = self._infinite_dataloader()
        pbar = tqdm(
            range(1, self.total_updates + 1),
            disable=not self.train_cfg.get("progress_bar", True),
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
