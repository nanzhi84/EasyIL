"""Standalone evaluation script for trained policy models."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from easyil.actors import build_actor
from easyil.envs import make_env, infer_env_dims
from easyil.utils import pick_device


@torch.no_grad()
def run_eval(
    actor: Any,
    eval_env: Any,
    n_episodes: int,
    device: torch.device,
    obs_horizon: int = 1,
    obs_norm_stats: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """Run evaluation episodes."""
    actor.eval()

    obs_mean = obs_norm_stats["mean"] if obs_norm_stats else None
    obs_std = obs_norm_stats["std"] if obs_norm_stats else None

    returns = []
    total_steps = 0
    t0 = time.perf_counter()

    for _ in range(n_episodes):
        obs = eval_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = np.asarray(obs, dtype=np.float32)

        obs_dim = obs.shape[-1]
        obs_hist = np.zeros((obs_horizon, obs_dim), dtype=np.float32)
        obs_hist[-1] = obs

        ep_return = 0.0
        done = False

        while not done:
            obs_normalized = obs_hist
            if obs_mean is not None:
                obs_normalized = (obs_hist - obs_mean) / obs_std

            obs_t = torch.from_numpy(obs_normalized).float().unsqueeze(0).to(device)
            action = actor.deterministic(obs_t)
            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, done, info = eval_env.step(action_np)
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            next_obs = np.asarray(next_obs, dtype=np.float32)

            ep_return += float(reward)
            total_steps += 1

            obs_hist[:-1] = obs_hist[1:]
            obs_hist[-1] = next_obs

        returns.append(ep_return)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "episodes": n_episodes,
        "env_steps": total_steps,
        "wall_time_sec": time.perf_counter() - t0,
    }


@hydra.main(config_path="conf", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_dir = Path(cfg.run_dir)
    device = pick_device(cfg.device)

    train_config_path = run_dir / "resolved_config.yaml"
    if not train_config_path.exists():
        raise FileNotFoundError(f"Training config not found: {train_config_path}")

    train_cfg = OmegaConf.load(train_config_path)

    if cfg.seed is not None:
        seed = cfg.seed
    else:
        seed = train_cfg.seed

    eval_env = make_env(train_cfg.env, run_dir, seed=seed + 1)
    obs_dim, act_dim, _ = infer_env_dims(eval_env)

    # Build actor from config
    actor_cfg = train_cfg.algo.get("actor", {})
    actor_cfg["type"] = "chunk" if train_cfg.algo.name == "bc" else "gaussian"
    actor_cfg["obs_horizon"] = train_cfg.algo.get("obs_horizon", 1)
    actor_cfg["action_horizon"] = train_cfg.algo.get("action_horizon", 1)

    actor = build_actor(DictConfig(actor_cfg), obs_dim, act_dim)
    actor.to(device)

    ckpt_path = run_dir / cfg.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    actor.load_state_dict(state_dict)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Environment: {train_cfg.env.id}")
    print(f"Device: {device}")

    obs_norm_stats = None
    norm_stats_path = run_dir / "obs_norm_stats.npz"
    if norm_stats_path.exists():
        norm_data = np.load(norm_stats_path)
        obs_norm_stats = {"mean": norm_data["mean"], "std": norm_data["std"]}
        print(f"Loaded observation normalization stats from {norm_stats_path}")

    print(f"Running {cfg.n_episodes} evaluation episodes...")

    result = run_eval(
        actor=actor,
        eval_env=eval_env,
        n_episodes=cfg.n_episodes,
        device=device,
        obs_horizon=train_cfg.algo.get("obs_horizon", 1),
        obs_norm_stats=obs_norm_stats,
    )

    eval_env.close()

    result_dict = {
        "run_dir": str(run_dir),
        "checkpoint": cfg.ckpt,
        "env_id": str(train_cfg.env.id),
        "n_episodes": result["episodes"],
        "mean_return": result["mean_return"],
        "std_return": result["std_return"],
        "env_steps": result["env_steps"],
        "wall_time_sec": result["wall_time_sec"],
        "seed": seed,
    }

    print(f"\n{'='*50}")
    print(f"Mean Return: {result['mean_return']:.2f} +/- {result['std_return']:.2f}")
    print(f"Episodes: {result['episodes']}")
    print(f"Env Steps: {result['env_steps']}")
    print(f"Wall Time: {result['wall_time_sec']:.2f}s")
    print(f"{'='*50}")

    if cfg.output:
        output_path = Path(cfg.output)
        output_path.write_text(json.dumps(result_dict, indent=2))
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
