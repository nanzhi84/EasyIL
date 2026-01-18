"""
Callback for online IL training with support for:
- Standard training metrics logging
- Evaluation and checkpointing
- Dual reward tracking (when using learned reward models)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

from easyil.envs import save_vecnormalize, has_compare_reward
from easyil.loggers import Logger


class OnlineTrainCallback(BaseCallback):
    """
    Callback for online IL training: logging, evaluation, checkpointing.

    When the environment has a RewardWrapper (reward shaping enabled),
    automatically tracks both shaped and true rewards for comparison.
    """

    def __init__(
        self,
        *,
        logger: Logger,
        output_dir: Path,
        train_env: Any,
        eval_env: Any,
        train_cfg: DictConfig,
        env_cfg: DictConfig,
    ):
        super().__init__()
        self._logger = logger
        self.output_dir = Path(output_dir)
        self.train_env = train_env
        self.eval_env = eval_env
        self.env_id = str(env_cfg.id)

        self.log_freq = train_cfg.get("log_freq", 1000)
        self.eval_freq = train_cfg.get("eval_freq", 10_000)
        self.n_eval_episodes = train_cfg.get("n_eval_episodes", 10)
        self.deterministic_eval = train_cfg.get("deterministic_eval", True)
        self.save_best = train_cfg.get("save_best", True)
        self.checkpoint_freq = train_cfg.get("checkpoint_freq", 100_000)
        self.plot_enabled = train_cfg.get("plot_enabled", True)
        self.plot_every_eval = train_cfg.get("plot_every_eval", True)
        self.final_eval_episodes = train_cfg.get("final_eval_episodes", 10)
        self.final_deterministic_eval = train_cfg.get("final_deterministic_eval", True)

        self.best_mean_reward = -math.inf
        self._last_logged = self._last_eval = self._last_ckpt = 0
        self._eval_history: list[dict[str, float]] = self._load_eval_history()

        self._dual_reward_mode = has_compare_reward(train_env)
        if self._dual_reward_mode:
            print("[callback] Dual reward tracking enabled (train vs compare reward)")
            self._ep_train_ret = 0.0
            self._ep_compare_ret = 0.0
            self._dual_eval_history: list[dict[str, float]] = self._load_dual_eval_history()

            # Determine plot labels: custom > auto-detect based on reward.enabled
            reward_enabled = env_cfg.get("reward", {}).get("enabled", False)
            labels = env_cfg.get("plot_labels", {})
            default_train, default_compare = ("RM Return", "True Return") if reward_enabled else ("True Return", "RM Return")
            self._train_label = labels.get("train") or default_train
            self._compare_label = labels.get("compare") or default_compare
            print(f"[callback] Plot labels: train={self._train_label}, compare={self._compare_label}")

    @property
    def _eval_csv_path(self) -> Path:
        return self.output_dir / "eval_curve.csv"

    @property
    def _dual_eval_csv_path(self) -> Path:
        return self.output_dir / "dual_returns.csv"

    def _load_eval_history(self) -> list[dict[str, float]]:
        if not self._eval_csv_path.exists():
            return []
        with self._eval_csv_path.open() as f:
            return sorted(
                [
                    {
                        "timesteps": float(r["timesteps"]),
                        "mean_reward": float(r["mean_reward"]),
                        "std_reward": float(r["std_reward"]),
                    }
                    for r in csv.DictReader(f)
                ],
                key=lambda d: d["timesteps"],
            )

    def _load_dual_eval_history(self) -> list[dict[str, float]]:
        if not self._dual_eval_csv_path.exists():
            return []
        with self._dual_eval_csv_path.open() as f:
            return sorted(
                [
                    {
                        "timesteps": float(r["timesteps"]),
                        "train_return": float(r["train_return"]),
                        "compare_return": float(r["compare_return"]),
                    }
                    for r in csv.DictReader(f)
                ],
                key=lambda d: d["timesteps"],
            )

    def _log(self, metrics: dict[str, Any]) -> None:
        self._logger.log(metrics, step=self.num_timesteps)

    def _on_step(self) -> bool:
        self._log_episode_info()
        self._log_scalars()
        self._eval_and_save()
        self._checkpoint()
        return True

    def _log_episode_info(self) -> None:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        rewards = self.locals.get("rewards")
        if infos is None or dones is None or rewards is None:
            return

        for i, info in enumerate(infos):
            if self._dual_reward_mode:
                self._ep_train_ret += rewards[i]
                self._ep_compare_ret += info.get("compare_reward", 0)

                if dones[i]:
                    self._log({
                        "train/episode_return_train": self._ep_train_ret,
                        "train/episode_return_compare": self._ep_compare_ret,
                    })
                    self._ep_train_ret = 0.0
                    self._ep_compare_ret = 0.0

            elif ep := (info.get("episode") if isinstance(info, dict) else None):
                self._log({"train/episode_return": ep.get("r"), "train/episode_length": ep.get("l")})

    def _log_scalars(self) -> None:
        if self.log_freq <= 0 or (self.num_timesteps - self._last_logged) < self.log_freq:
            return
        self._last_logged = self.num_timesteps
        metrics = {"time/total_timesteps": self.num_timesteps}
        if sb3_logger := getattr(self.model, "logger", None):
            metrics.update(getattr(sb3_logger, "name_to_value", {}))
        self._log(metrics)

    def _checkpoint(self) -> None:
        if self.checkpoint_freq <= 0 or (self.num_timesteps - self._last_ckpt) < self.checkpoint_freq:
            return
        self._last_ckpt = self.num_timesteps
        self.model.save(str(self.output_dir / f"checkpoint_{self.num_timesteps}"))
        save_vecnormalize(self.train_env, self.output_dir / "vecnormalize.pkl")

    def _eval_and_save(self) -> None:
        if self.eval_freq <= 0 or (self.num_timesteps - self._last_eval) < self.eval_freq:
            return
        self._last_eval = self.num_timesteps

        if self._dual_reward_mode:
            train_ret, compare_ret = self._run_dual_eval()
            self._record_dual_eval(train_ret, compare_ret)
            mean = train_ret
        else:
            mean, std = self._run_eval(self.n_eval_episodes, self.deterministic_eval, "eval")
            self._record_eval(mean, std)

        if self.plot_every_eval:
            self._plot()

        if self.save_best and mean > self.best_mean_reward:
            self.best_mean_reward = mean
            self.model.save(str(self.output_dir / "best_model"))
            save_vecnormalize(self.train_env, self.output_dir / "vecnormalize.pkl")
            self._log({"eval/best_mean_reward": mean})

    def _run_eval(self, n_episodes: int, deterministic: bool, prefix: str) -> tuple[float, float]:
        if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            sync_envs_normalization(self.train_env, self.eval_env)
        mean, std = evaluate_policy(self.model, self.eval_env, n_eval_episodes=n_episodes, deterministic=deterministic)
        self._log({f"{prefix}/mean_reward": mean, f"{prefix}/std_reward": std})
        return float(mean), float(std)

    def _run_dual_eval(self) -> tuple[float, float]:
        """Run evaluation and compute both train and compare returns."""
        if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            sync_envs_normalization(self.train_env, self.eval_env)

        total_train = 0.0
        total_compare = 0.0

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_train = 0.0
            ep_compare = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic_eval)
                obs, rewards, dones, infos = self.eval_env.step(action)
                ep_train += rewards[0]
                ep_compare += infos[0].get("compare_reward", 0)
                done = dones[0]

            total_train += ep_train
            total_compare += ep_compare

        avg_train = total_train / self.n_eval_episodes
        avg_compare = total_compare / self.n_eval_episodes

        self._log({
            "eval/avg_train_return": avg_train,
            "eval/avg_compare_return": avg_compare,
        })
        print(f"Eval @ {self.num_timesteps}: train={avg_train:.2f}, compare={avg_compare:.2f}")

        return avg_train, avg_compare

    def _record_eval(self, mean: float, std: float) -> None:
        row = {"timesteps": float(self.num_timesteps), "mean_reward": mean, "std_reward": std}
        self._eval_history.append(row)
        write_header = not self._eval_csv_path.exists()
        with self._eval_csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timesteps", "mean_reward", "std_reward"])
            if write_header:
                w.writeheader()
            w.writerow(row)

    def _record_dual_eval(self, train_ret: float, compare_ret: float) -> None:
        row = {
            "timesteps": float(self.num_timesteps),
            "train_return": train_ret,
            "compare_return": compare_ret,
        }
        self._dual_eval_history.append(row)
        write_header = not self._dual_eval_csv_path.exists()
        with self._dual_eval_csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timesteps", "train_return", "compare_return"])
            if write_header:
                w.writeheader()
            w.writerow(row)

    def _plot(self) -> None:
        if not self.plot_enabled:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self._dual_reward_mode and len(self._dual_eval_history) >= 2:
            self._plot_dual()
        elif self._eval_history:
            self._plot_single()

    def _plot_single(self) -> None:
        import matplotlib.pyplot as plt

        xs = np.array([r["timesteps"] for r in self._eval_history])
        ys = np.array([r["mean_reward"] for r in self._eval_history])
        stds = np.array([r["std_reward"] for r in self._eval_history])

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        color = "#2E86AB"
        ax.plot(xs, ys, color=color, linewidth=2.5, label="Online IL", zorder=3)
        ax.fill_between(xs, ys - stds, ys + stds, color=color, alpha=0.2, zorder=2)

        ax.set_xlabel("Environment Steps", fontsize=12, fontweight="medium")
        ax.set_ylabel("Return", fontsize=12, fontweight="medium")
        ax.set_title(self.env_id, fontsize=14, fontweight="bold", pad=10)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
        ax.grid(True, linestyle="-", alpha=0.3, zorder=1)
        ax.set_axisbelow(True)

        fig.tight_layout()
        fig.savefig(self.output_dir / "learning_curve.png", bbox_inches="tight")
        plt.close(fig)

    def _plot_dual(self) -> None:
        import matplotlib.pyplot as plt

        xs = np.array([r["timesteps"] for r in self._dual_eval_history])
        train_ys = np.array([r["train_return"] for r in self._dual_eval_history])
        compare_ys = np.array([r["compare_return"] for r in self._dual_eval_history])

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

        color1 = "#2E86AB"
        ax1.set_xlabel("Environment Steps", fontsize=12, fontweight="medium")
        ax1.set_ylabel(self._train_label, color=color1, fontsize=12, fontweight="medium")
        ax1.plot(xs, train_ys, color=color1, linewidth=2.5, label=self._train_label)
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "#E94F37"
        ax2.set_ylabel(self._compare_label, color=color2, fontsize=12, fontweight="medium")
        ax2.plot(xs, compare_ys, color=color2, linewidth=2.5, label=self._compare_label)
        ax2.tick_params(axis="y", labelcolor=color2)

        ax1.set_title(f"Dual Returns ({self.env_id})", fontsize=14, fontweight="bold", pad=10)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=11, framealpha=0.9)

        fig.tight_layout()
        fig.savefig(self.output_dir / "dual_returns.png", bbox_inches="tight")
        plt.close(fig)

    def _on_training_end(self) -> None:
        if self.final_eval_episodes > 0:
            if self._dual_reward_mode:
                train_ret, compare_ret = self._run_dual_eval()
                (self.output_dir / "final_eval.json").write_text(
                    json.dumps({
                        "env_id": self.env_id,
                        "timesteps": int(self.num_timesteps),
                        "n_eval_episodes": self.final_eval_episodes,
                        "train_return": float(train_ret),
                        "compare_return": float(compare_ret),
                    }, indent=2)
                )
            else:
                mean, std = self._run_eval(self.final_eval_episodes, self.final_deterministic_eval, "final_eval")
                last_ts = self._eval_history[-1]["timesteps"] if self._eval_history else -1
                if float(self.num_timesteps) != last_ts:
                    self._record_eval(mean, std)
                (self.output_dir / "final_eval.json").write_text(
                    json.dumps({
                        "env_id": self.env_id,
                        "timesteps": int(self.num_timesteps),
                        "n_eval_episodes": self.final_eval_episodes,
                        "deterministic": self.final_deterministic_eval,
                        "mean_reward": float(mean),
                        "std_reward": float(std),
                    }, indent=2)
                )
        self._plot()
