"""Callbacks for offline BC training (JAX)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from easyil.envs import VecEnvProtocol
from easyil.eval import EvalResult, run_eval
from easyil.loggers import Logger

if TYPE_CHECKING:
    from easyil.algos import BCModule


class OfflineTrainCallback:
    """Callback for offline training: logging, evaluation, checkpointing (JAX)."""

    def __init__(
        self,
        *,
        logger: Logger,
        output_dir: Path,
        eval_env: VecEnvProtocol,
        train_cfg: DictConfig,
        algo_cfg: DictConfig,
        env_cfg: DictConfig,
        obs_norm_stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        self._logger = logger
        self.output_dir = Path(output_dir)
        self.eval_env = eval_env
        self.env_id = str(env_cfg.id)
        self.algo_name = str(algo_cfg.name)
        self.obs_norm_stats = obs_norm_stats

        self.obs_horizon = algo_cfg.obs_horizon
        self.action_horizon = algo_cfg.action_horizon
        self.exec_horizon = algo_cfg.get("exec_horizon", algo_cfg.action_horizon)

        self.log_freq = train_cfg.get("log_freq", 50)
        self.eval_freq = train_cfg.get("eval_freq", 1000)
        self.n_eval_episodes = train_cfg.get("n_eval_episodes", 10)
        self.save_best = train_cfg.get("save_best", True)
        self.checkpoint_freq = train_cfg.get("checkpoint_freq", 0)
        self.plot_enabled = train_cfg.get("plot_enabled", True)
        self.plot_every_eval = train_cfg.get("plot_every_eval", True)
        self.final_eval_episodes = train_cfg.get("final_eval_episodes", 10)

        self.best_mean_reward = -np.inf
        self._eval_history: List[dict] = self._load_eval_history()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _eval_csv_path(self) -> Path:
        return self.output_dir / "eval_curve.csv"

    def _load_eval_history(self) -> List[dict]:
        if not self._eval_csv_path.exists():
            return []
        with self._eval_csv_path.open() as f:
            return sorted(
                [
                    {
                        "update": int(r["update"]),
                        "mean_return": float(r["mean_return"]),
                        "std_return": float(r["std_return"]),
                    }
                    for r in csv.DictReader(f)
                ],
                key=lambda d: d["update"],
            )

    def _log(self, metrics: Dict[str, Any], step: int) -> None:
        self._logger.log(metrics, step=step)

    def log_train(self, update: int, loss: float, lr: float) -> None:
        if self.log_freq <= 0 or update % self.log_freq != 0:
            return
        self._log({"train/loss": loss, "train/lr": lr}, step=update)

    def save_checkpoint(self, update: int, module: "BCModule") -> None:
        if self.checkpoint_freq <= 0 or update % self.checkpoint_freq != 0:
            return
        module.save(str(self.output_dir / f"checkpoint_{update:08d}.pkl"))

    def _record_eval(self, update: int, result: EvalResult) -> None:
        row = {
            "update": update,
            "mean_return": result.mean_return,
            "std_return": result.std_return,
            "env_steps": result.env_steps,
            "wall_time_sec": result.wall_time_sec,
        }
        self._eval_history.append(row)
        write_header = not self._eval_csv_path.exists()
        with self._eval_csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["update", "mean_return", "std_return", "env_steps", "wall_time_sec"])
            if write_header:
                w.writeheader()
            w.writerow(row)

    def _plot(self) -> None:
        if not self.plot_enabled or not self._eval_history:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = np.array([r["update"] for r in self._eval_history])
        ys = np.array([r["mean_return"] for r in self._eval_history])
        stds = np.array([r["std_return"] for r in self._eval_history])

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        color = "#2E86AB"
        ax.plot(xs, ys, color=color, linewidth=2.5, label=self.algo_name, zorder=3)
        ax.fill_between(xs, ys - stds, ys + stds, color=color, alpha=0.2, zorder=2)

        ax.set_xlabel("Updates", fontsize=12, fontweight="medium")
        ax.set_ylabel("Return", fontsize=12, fontweight="medium")
        ax.set_title(self.env_id, fontsize=14, fontweight="bold", pad=10)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)

        ax.grid(True, linestyle="-", alpha=0.3, zorder=1)
        ax.set_axisbelow(True)

        fig.tight_layout()
        fig.savefig(self.output_dir / "learning_curve.png", bbox_inches="tight")
        plt.close(fig)

    def _run_eval(self, module: "BCModule", n_episodes: int, seed: int, rng_key: jnp.ndarray) -> EvalResult:
        return run_eval(
            module=module,
            eval_env=self.eval_env,
            n_episodes=n_episodes,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
            exec_horizon=self.exec_horizon,
            rng_key=rng_key,
            seed=seed,
            obs_norm_stats=self.obs_norm_stats,
        )

    def maybe_eval(
        self,
        update: int,
        module: "BCModule",
        seed: int,
        rng_key: jnp.ndarray,
    ) -> Optional[EvalResult]:
        if self.eval_freq <= 0 or update % self.eval_freq != 0:
            return None

        rng_key, eval_key = jax.random.split(rng_key)
        result = self._run_eval(module, self.n_eval_episodes, seed + 12345, eval_key)
        self._record_eval(update, result)
        self._log({"eval/mean_return": result.mean_return, "eval/std_return": result.std_return}, step=update)

        if self.save_best and result.mean_return > self.best_mean_reward:
            self.best_mean_reward = result.mean_return
            module.save(str(self.output_dir / "best_model.pkl"))
            self._log({"eval/best_mean_return": result.mean_return}, step=update)

        if self.plot_every_eval:
            self._plot()
        return result

    def on_training_end(
        self,
        update: int,
        module: "BCModule",
        seed: int,
        rng_key: jnp.ndarray,
    ) -> None:
        if self.final_eval_episodes > 0:
            rng_key, eval_key = jax.random.split(rng_key)
            result = self._run_eval(module, self.final_eval_episodes, seed + 999, eval_key)
            last_update = self._eval_history[-1]["update"] if self._eval_history else -1
            if update != last_update:
                self._record_eval(update, result)
            (self.output_dir / "final_eval.json").write_text(
                json.dumps(
                    {
                        "env_id": self.env_id,
                        "update": update,
                        "n_eval_episodes": self.final_eval_episodes,
                        "mean_return": result.mean_return,
                        "std_return": result.std_return,
                    },
                    indent=2,
                )
            )
        self._plot()
