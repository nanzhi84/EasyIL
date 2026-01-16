"""Standalone evaluation script and utilities for trained policy models."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from easyil.algos import build_algo
from easyil.envs import make_env

if TYPE_CHECKING:
    from easyil.algos import BCModule


@dataclass
class EvalResult:
    """Evaluation result container."""

    mean_return: float
    std_return: float
    episodes: int
    env_steps: int
    wall_time_sec: float


def load_eval_context(
    run_dir: Path,
    ckpt: str,
    device: torch.device,
    seed: int,
) -> Tuple["BCModule", Any, DictConfig, Optional[Dict[str, np.ndarray]]]:
    """Load model, environment, config, and normalization stats for evaluation.

    Args:
        run_dir: Path to training run directory.
        ckpt: Checkpoint filename (relative to run_dir).
        device: Torch device.
        seed: Random seed for environment.

    Returns:
        Tuple of (module, eval_env, train_cfg, obs_norm_stats).
    """
    train_config_path = run_dir / "resolved_config.yaml"
    if not train_config_path.exists():
        raise FileNotFoundError(f"Training config not found: {train_config_path}")

    train_cfg = OmegaConf.load(train_config_path)

    eval_env = make_env(train_cfg.env, run_dir, seed=seed + 1, training=False)
    module = build_algo(train_cfg.algo, eval_env, str(run_dir))
    module.to(device)

    ckpt_path = run_dir / ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    # Handle torch.compile() prefix
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        cleaned_state_dict[new_key] = v

    module.ema_net.load_state_dict(cleaned_state_dict)
    module.net.load_state_dict(cleaned_state_dict)
    module.policy.net = module.ema_net

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Environment: {train_cfg.env.id}")
    print(f"Device: {device}")

    obs_norm_stats = None
    norm_stats_path = run_dir / "obs_norm_stats.npz"
    if norm_stats_path.exists():
        norm_data = np.load(norm_stats_path)
        obs_norm_stats = {"mean": norm_data["mean"], "std": norm_data["std"]}
        print(f"Loaded observation normalization stats from {norm_stats_path}")

    return module, eval_env, train_cfg, obs_norm_stats


@torch.no_grad()
def run_eval(
    module: "BCModule",
    eval_env: Any,
    n_episodes: int,
    *,
    obs_horizon: int,
    action_horizon: int,
    exec_horizon: int,
    device: torch.device,
    seed: int,
    obs_norm_stats: Optional[Dict[str, np.ndarray]] = None,
) -> EvalResult:
    """Run evaluation episodes and return aggregated statistics.

    Args:
        module: The BCModule containing the policy.
        eval_env: Vectorized evaluation environment.
        n_episodes: Number of episodes to run.
        obs_horizon: Observation history length.
        action_horizon: Action chunk length.
        exec_horizon: Number of actions to execute before replanning.
        device: Torch device for inference.
        seed: Random seed for reproducibility.
        obs_norm_stats: Optional dict with 'mean' and 'std' for observation normalization.

    Returns:
        EvalResult with mean/std return and timing info.
    """
    module.policy.eval()

    obs_mean = obs_norm_stats["mean"] if obs_norm_stats else None
    obs_std = obs_norm_stats["std"] if obs_norm_stats else None

    n_envs = getattr(eval_env, "num_envs", 1)
    obs = np.asarray(eval_env.reset(), dtype=np.float32)
    obs_dim = obs.shape[-1]
    act_dim = module.act_dim

    obs_normalized = (obs - obs_mean) / obs_std if obs_mean is not None else obs

    obs_hist = np.zeros((n_envs, obs_horizon, obs_dim), dtype=np.float32)
    for e in range(n_envs):
        obs_hist[e, :, :] = obs_normalized[e]

    action_queues = [np.zeros((0, act_dim), dtype=np.float32) for _ in range(n_envs)]
    exec_left = np.zeros((n_envs,), dtype=np.int64)
    ep_returns = np.zeros((n_envs,), dtype=np.float64)
    finished_returns: List[float] = []

    env_steps = 0
    t0 = time.perf_counter()

    while len(finished_returns) < n_episodes:
        need_plan = [i for i in range(n_envs) if exec_left[i] <= 0 or len(action_queues[i]) == 0]
        if need_plan:
            obs_batch = torch.from_numpy(obs_hist[need_plan]).to(device=device)
            chunks = module.policy.sample_action_chunk(obs=obs_batch, act_dim=act_dim, action_horizon=action_horizon)
            chunks_np = chunks.detach().cpu().numpy().astype(np.float32)
            for k, env_i in enumerate(need_plan):
                action_queues[env_i] = chunks_np[k]
                exec_left[env_i] = exec_horizon

        actions = np.zeros((n_envs, act_dim), dtype=np.float32)
        for i in range(n_envs):
            actions[i] = action_queues[i][0]
            action_queues[i] = action_queues[i][1:]
            exec_left[i] -= 1

        obs, rewards, dones, infos = eval_env.step(actions)
        obs = np.asarray(obs, dtype=np.float32)
        env_steps += n_envs
        ep_returns += rewards

        obs_normalized = (obs - obs_mean) / obs_std if obs_mean is not None else obs

        obs_hist[:, :-1, :] = obs_hist[:, 1:, :]
        obs_hist[:, -1, :] = obs_normalized

        for i, d in enumerate(dones.tolist()):
            if not d:
                continue
            finished_returns.append(float(ep_returns[i]))
            ep_returns[i] = 0.0
            obs_hist[i, :, :] = obs_normalized[i]
            action_queues[i] = np.zeros((0, act_dim), dtype=np.float32)
            exec_left[i] = 0
            if len(finished_returns) >= n_episodes:
                break

    rets = np.asarray(finished_returns[:n_episodes], dtype=np.float64)
    return EvalResult(
        mean_return=float(rets.mean()),
        std_return=float(rets.std(ddof=0)),
        episodes=n_episodes,
        env_steps=env_steps,
        wall_time_sec=time.perf_counter() - t0,
    )


@hydra.main(config_path="conf", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_dir = Path(cfg.run_dir)
    device = torch.device("cuda")

    if cfg.seed is not None:
        seed = cfg.seed
    else:
        pre_cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
        seed = pre_cfg.seed

    module, eval_env, train_cfg, obs_norm_stats = load_eval_context(
        run_dir=run_dir,
        ckpt=cfg.ckpt,
        device=device,
        seed=seed,
    )

    print(f"Running {cfg.n_episodes} evaluation episodes...")

    obs_horizon = train_cfg.algo.obs_horizon
    action_horizon = train_cfg.algo.action_horizon
    exec_horizon = train_cfg.algo.get("exec_horizon", action_horizon)

    result = run_eval(
        module=module,
        eval_env=eval_env,
        n_episodes=cfg.n_episodes,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        exec_horizon=exec_horizon,
        device=device,
        seed=seed,
        obs_norm_stats=obs_norm_stats,
    )

    eval_env.close()

    result_dict = {
        "run_dir": str(run_dir),
        "checkpoint": cfg.ckpt,
        "env_id": str(train_cfg.env.id),
        "n_episodes": result.episodes,
        "mean_return": result.mean_return,
        "std_return": result.std_return,
        "env_steps": result.env_steps,
        "wall_time_sec": result.wall_time_sec,
        "seed": seed,
    }

    print(f"\n{'='*50}")
    print(f"Mean Return: {result.mean_return:.2f} ± {result.std_return:.2f}")
    print(f"Episodes: {result.episodes}")
    print(f"Env Steps: {result.env_steps}")
    print(f"Wall Time: {result.wall_time_sec:.2f}s")
    print(f"{'='*50}")

    if cfg.output:
        output_path = Path(cfg.output)
        output_path.write_text(json.dumps(result_dict, indent=2))
        print(f"Results saved to: {output_path}")
    else:
        print("\nFull results JSON:")
        print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
