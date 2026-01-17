"""Monotonicity evaluation for diffusion policy.

Evaluates policy performance at different diffusion denoising steps to analyze
the relationship between noise level and action quality.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from easyil.eval import load_eval_context
from easyil.utils.cfg import pick_device

if TYPE_CHECKING:
    from easyil.algos.diffusion_bc import DiffusionBCModule


@dataclass
class MonotonicityStats:
    """Statistics container for monotonicity evaluation."""

    k_values: List[int] = field(default_factory=list)
    mean_returns: List[float] = field(default_factory=list)
    std_returns: List[float] = field(default_factory=list)
    min_returns: List[float] = field(default_factory=list)
    max_returns: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert to JSON-friendly dictionary."""
        return {
            str(k): {
                "mean": m,
                "std": s,
                "min": mn,
                "max": mx,
            }
            for k, m, s, mn, mx in zip(
                self.k_values, self.mean_returns, self.std_returns, self.min_returns, self.max_returns
            )
        }


@torch.no_grad()
def evaluate_monotonicity(
    module: "DiffusionBCModule",
    eval_env: Any,
    *,
    use_k: List[int],
    n_episodes: int,
    max_episode_length: int,
    obs_horizon: int,
    action_horizon: int,
    exec_horizon: int,
    device: torch.device,
    seed: int,
    return_x0: bool = False,
    obs_norm_stats: Optional[Dict[str, np.ndarray]] = None,
    show_progress: bool = True,
) -> MonotonicityStats:
    """Evaluate diffusion policy at different denoising steps using VecEnv parallelism.

    Args:
        module: The DiffusionBCModule containing the policy.
        eval_env: Vectorized evaluation environment.
        use_k: List of diffusion steps to evaluate (0 = fully denoised).
        n_episodes: Number of episodes per diffusion step.
        max_episode_length: Maximum steps per episode (unused, VecEnv auto-resets).
        obs_horizon: Observation history length.
        action_horizon: Action chunk length.
        exec_horizon: Number of actions to execute before replanning.
        device: Torch device for inference.
        seed: Random seed for reproducibility.
        return_x0: If True, use predicted x0 at each step; otherwise use noisy xt.
        obs_norm_stats: Optional dict with 'mean' and 'std' for observation normalization.
        show_progress: Whether to show progress bars.

    Returns:
        MonotonicityStats with evaluation results per diffusion step.
    """
    module.policy.eval()

    obs_mean = obs_norm_stats["mean"] if obs_norm_stats else None
    obs_std = obs_norm_stats["std"] if obs_norm_stats else None

    obs_dim = module.obs_dim
    act_dim = module.act_dim
    n_envs = getattr(eval_env, "num_envs", 1)

    # Sort k values in descending order (most noisy to least noisy)
    use_k = sorted(use_k, reverse=True)

    stats = MonotonicityStats()

    mode_str = "Predicted x0" if return_x0 else "Noisy xt"
    print(f"Starting Monotonicity Evaluation (mode: {mode_str})")
    print(f"Evaluating diffusion steps: {use_k}")
    print(f"Using {n_envs} parallel environments")

    k_iter = tqdm(use_k, desc="Diffusion steps", disable=not show_progress)

    for k in k_iter:
        k_iter.set_description(f"k={k}")

        # Reset all environments
        obs = np.asarray(eval_env.reset(), dtype=np.float32)

        # Apply normalization
        obs_normalized = (obs - obs_mean) / obs_std if obs_mean is not None else obs

        # Initialize observation history for all envs: (n_envs, obs_horizon, obs_dim)
        obs_hist = np.zeros((n_envs, obs_horizon, obs_dim), dtype=np.float32)
        for e in range(n_envs):
            obs_hist[e, :, :] = obs_normalized[e]

        # Action queues and exec counters for each env
        action_queues: List[np.ndarray] = [np.zeros((0, act_dim), dtype=np.float32) for _ in range(n_envs)]
        exec_left = np.zeros((n_envs,), dtype=np.int64)

        # Episode tracking
        ep_returns = np.zeros((n_envs,), dtype=np.float64)
        finished_returns: List[float] = []

        pbar = tqdm(total=n_episodes, desc=f"Episodes (k={k})", leave=False, disable=not show_progress)

        while len(finished_returns) < n_episodes:
            # Find envs that need replanning
            need_plan = [i for i in range(n_envs) if exec_left[i] <= 0 or len(action_queues[i]) == 0]

            if need_plan:
                # Batch inference for envs that need planning
                obs_batch = torch.from_numpy(obs_hist[need_plan]).to(device=device)
                chunks = module.policy.sample_action_chunk_at_step(
                    obs=obs_batch,
                    act_dim=act_dim,
                    action_horizon=action_horizon,
                    stop_step=k,
                    return_x0=return_x0,
                )
                chunks_np = chunks.detach().cpu().numpy().astype(np.float32)

                for idx, env_i in enumerate(need_plan):
                    action_queues[env_i] = chunks_np[idx]
                    exec_left[env_i] = exec_horizon

            # Prepare actions for all envs
            actions = np.zeros((n_envs, act_dim), dtype=np.float32)
            for i in range(n_envs):
                actions[i] = action_queues[i][0]
                action_queues[i] = action_queues[i][1:]
                exec_left[i] -= 1

            # Step all envs
            obs, rewards, dones, infos = eval_env.step(actions)
            obs = np.asarray(obs, dtype=np.float32)
            ep_returns += rewards

            # Apply normalization
            obs_normalized = (obs - obs_mean) / obs_std if obs_mean is not None else obs

            # Update observation history
            obs_hist[:, :-1, :] = obs_hist[:, 1:, :]
            obs_hist[:, -1, :] = obs_normalized

            # Handle episode completions
            for i, done in enumerate(dones.tolist()):
                if not done:
                    continue

                finished_returns.append(float(ep_returns[i]))
                pbar.update(1)

                # Reset this env's state
                ep_returns[i] = 0.0
                obs_hist[i, :, :] = obs_normalized[i]
                action_queues[i] = np.zeros((0, act_dim), dtype=np.float32)
                exec_left[i] = 0

                if len(finished_returns) >= n_episodes:
                    break

        pbar.close()

        # Compute statistics
        returns_arr = np.array(finished_returns[:n_episodes])
        stats.k_values.append(k)
        stats.mean_returns.append(float(returns_arr.mean()))
        stats.std_returns.append(float(returns_arr.std(ddof=0)))
        stats.min_returns.append(float(returns_arr.min()))
        stats.max_returns.append(float(returns_arr.max()))

        print(f"k={k}: Return = {returns_arr.mean():.2f} ± {returns_arr.std():.2f}")

    return stats


def plot_monotonicity(
    stats: MonotonicityStats,
    output_path: Path,
    env_id: str,
    return_x0: bool,
) -> None:
    """Plot monotonicity evaluation results.

    Args:
        stats: MonotonicityStats from evaluate_monotonicity.
        output_path: Path to save the plot.
        env_id: Environment ID for title.
        return_x0: Whether x0 or xt was used (for title).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k_values = np.array(stats.k_values)
    means = np.array(stats.mean_returns)
    stds = np.array(stats.std_returns)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    color = "#3366CC"
    ax.plot(k_values, means, marker="o", linestyle="-", color=color, linewidth=2, label="Mean Return")
    ax.fill_between(k_values, means - stds, means + stds, color=color, alpha=0.2, label="Std Dev")

    mode_str = "Predicted x0" if return_x0 else "Noisy xt"
    ax.set_title(f"Diffusion Policy Monotonicity Check: {env_id}\n({mode_str})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Diffusion Step k (0 = Fully Denoised)", fontsize=12)
    ax.set_ylabel("Average Episode Return", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="best", fontsize=10)
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def run_monotonicity_eval(
    run_dir: Path,
    ckpt: str,
    use_k: List[int],
    n_episodes: int,
    max_episode_length: int,
    return_x0: bool,
    device: torch.device,
    seed: int,
    output_subdir: str = "monotonicity",
) -> MonotonicityStats:
    """Run full monotonicity evaluation with model loading and result saving.

    Args:
        run_dir: Path to training run directory.
        ckpt: Checkpoint filename.
        use_k: List of diffusion steps to evaluate.
        n_episodes: Number of episodes per step.
        max_episode_length: Maximum episode length.
        return_x0: Whether to use predicted x0 or noisy xt.
        device: Torch device.
        seed: Random seed.
        output_subdir: Subdirectory name for outputs.

    Returns:
        MonotonicityStats with results.
    """
    # Load model, environment, config, and normalization stats
    module, eval_env, train_cfg, obs_norm_stats = load_eval_context(
        run_dir=run_dir,
        ckpt=ckpt,
        device=device,
        seed=seed,
    )

    env_id = str(train_cfg.env.id)
    obs_horizon = train_cfg.algo.obs_horizon
    action_horizon = train_cfg.algo.action_horizon
    exec_horizon = train_cfg.algo.get("exec_horizon", action_horizon)

    stats = evaluate_monotonicity(
        module=module,
        eval_env=eval_env,
        use_k=use_k,
        n_episodes=n_episodes,
        max_episode_length=max_episode_length,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        exec_horizon=exec_horizon,
        device=device,
        seed=seed,
        return_x0=return_x0,
        obs_norm_stats=obs_norm_stats,
    )

    eval_env.close()

    # Save results
    output_dir = run_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_suffix = "x0" if return_x0 else "xt"
    json_path = output_dir / f"monotonicity_{mode_suffix}.json"
    json_path.write_text(json.dumps(stats.to_dict(), indent=2))
    print(f"Results saved to {json_path}")

    plot_path = output_dir / f"monotonicity_{mode_suffix}.png"
    plot_monotonicity(stats, plot_path, env_id, return_x0)

    return stats


@hydra.main(config_path="conf", config_name="monotonicity", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for monotonicity evaluation."""
    run_dir = Path(cfg.run_dir)
    device = pick_device(cfg.device)

    # Parse use_k from config
    use_k = list(cfg.use_k) if cfg.use_k else list(range(cfg.max_k + 1))

    run_monotonicity_eval(
        run_dir=run_dir,
        ckpt=cfg.ckpt,
        use_k=use_k,
        n_episodes=cfg.n_episodes,
        max_episode_length=cfg.max_episode_length,
        return_x0=cfg.return_x0,
        device=device,
        seed=cfg.seed,
        output_subdir=cfg.output_subdir,
    )


if __name__ == "__main__":
    main()
