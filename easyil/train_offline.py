"""Offline training entry point for behavior cloning algorithms."""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from easyil.algos import build_algo
from easyil.callbacks import OfflineTrainCallback
from easyil.datasets import ChunkedExpertDataset, load_expert_npz
from easyil.envs import make_env, save_vecnormalize
from easyil.loggers import build_logger
from easyil.trainers import OfflineTrainer
from easyil.utils.cfg import pick_device, save_resolved_config, seed_everything


@hydra.main(config_path="conf", config_name="train_offline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_resolved_config(cfg, output_dir)
    seed_everything(cfg.seed)

    device = pick_device(str(cfg.train.get("device", "auto")))
    logger = build_logger(cfg.logger, output_dir, cfg)
    eval_env = make_env(cfg.env, output_dir, seed=cfg.seed + 1)

    module = build_algo(cfg.algo, eval_env, str(output_dir))
    module.to(device)

    # Compile networks for faster training (works for all BC algorithms)
    use_compile = cfg.train.get("use_compile", True)
    if use_compile and device.type == "cuda":
        module.net = torch.compile(module.net, mode="reduce-overhead")
        module.ema_net = torch.compile(module.ema_net, mode="reduce-overhead")
        module.policy.net = module.ema_net

    # Dataset
    num_trajs = cfg.train.dataset.get("num_trajs", None)
    data = load_expert_npz(cfg.train.dataset.path, num_trajs=num_trajs)
    obs_normalize = cfg.train.dataset.get("obs_normalize", False)
    ds = ChunkedExpertDataset(
        data=data,
        obs_horizon=cfg.algo.obs_horizon,
        action_horizon=cfg.algo.action_horizon,
        obs_normalize=obs_normalize,
    )

    obs_norm_stats = None
    if obs_normalize and ds.obs_mean is not None:
        obs_norm_stats = {"mean": ds.obs_mean, "std": ds.obs_std}
        np.savez(output_dir / "obs_norm_stats.npz", mean=ds.obs_mean, std=ds.obs_std)

    num_workers = cfg.train.get("num_workers", 4)
    dl = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cfg.train.get("pin_memory", True),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        module.net.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.get("weight_decay", 0.0),
    )

    callback = OfflineTrainCallback(
        logger=logger,
        output_dir=output_dir,
        eval_env=eval_env,
        device=device,
        train_cfg=cfg.train,
        algo_cfg=cfg.algo,
        env_cfg=cfg.env,
        obs_norm_stats=obs_norm_stats,
    )

    trainer = OfflineTrainer(
        module=module,
        dataloader=dl,
        optimizer=optimizer,
        callback=callback,
        device=device,
        train_cfg=cfg.train,
        seed=cfg.seed,
    )

    trainer.train()

    torch.save(module.ema_net.state_dict(), output_dir / "final_model.pt")
    save_vecnormalize(eval_env, output_dir / "vecnormalize.pkl")

    eval_env.close()
    logger.finish()


if __name__ == "__main__":
    main()
