"""Callbacks for offline BC training."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from easyil.loggers import Logger


class OfflineCallback:
    """Callback for offline training: logging, evaluation, checkpointing."""

    def __init__(
        self,
        *,
        logger: Logger,
        output_dir: Path,
        eval_env: Any,
        device: torch.device,
        train_cfg: DictConfig,
        algo_cfg: DictConfig,
        env_cfg: DictConfig,
        obs_norm_stats: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self._logger = logger
        self.output_dir = Path(output_dir)
        self.eval_env = eval_env
        self.device = device
        self.env_id = str(env_cfg.id)
        self.algo_name = str(algo_cfg.name)
        self.obs_norm_stats = obs_norm_stats

        self.obs_horizon = algo_cfg.get("obs_horizon", 1)
        self.action_horizon = algo_cfg.get("action_horizon", 1)

        self.log_freq = train_cfg.get("log_freq", 50)
        self.eval_freq = train_cfg.get("eval_freq", 1000)
        self.n_eval_episodes = train_cfg.get("n_eval_episodes", 10)
        self.save_best = train_cfg.get("save_best", True)
        self.checkpoint_freq = train_cfg.get("checkpoint_freq", 0)
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

    def log_train(self, update: int, metrics: Dict[str, float]) -> None:
        """Log training metrics."""
        if self.log_freq <= 0 or update % self.log_freq != 0:
            return
        log_metrics = {f"train/{k}": v for k, v in metrics.items()}
        self._log(log_metrics, step=update)

    def save_checkpoint(self, update: int, algo: Any) -> None:
        """Save checkpoint."""
        if self.checkpoint_freq <= 0 or update % self.checkpoint_freq != 0:
            return
        torch.save(algo.state_dict(), self.output_dir / f"checkpoint_{update:08d}.pt")

    def _run_eval(self, algo: Any, n_episodes: int, seed: int) -> Dict[str, float]:
        """Run evaluation episodes."""
        actor = algo.inference_actor()
        actor.eval()

        obs_mean = self.obs_norm_stats["mean"] if self.obs_norm_stats else None
        obs_std = self.obs_norm_stats["std"] if self.obs_norm_stats else None

        returns = []
        for ep in range(n_episodes):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            obs = np.asarray(obs, dtype=np.float32)

            obs_hist = np.zeros((self.obs_horizon, obs.shape[-1]), dtype=np.float32)
            obs_hist[-1] = obs

            ep_return = 0.0
            done = False

            while not done:
                obs_normalized = obs_hist
                if obs_mean is not None:
                    obs_normalized = (obs_hist - obs_mean) / obs_std

                obs_t = torch.from_numpy(obs_normalized).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = actor.deterministic(obs_t)
                action_np = action.squeeze(0).cpu().numpy()

                next_obs, reward, done, info = self.eval_env.step(action_np)
                if isinstance(next_obs, tuple):
                    next_obs = next_obs[0]
                next_obs = np.asarray(next_obs, dtype=np.float32)

                ep_return += float(reward)

                # Update observation history
                obs_hist[:-1] = obs_hist[1:]
                obs_hist[-1] = next_obs

            returns.append(ep_return)

        actor.train()
        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
        }

    def _record_eval(self, update: int, result: Dict[str, float]) -> None:
        row = {
            "update": update,
            "mean_return": result["mean_return"],
            "std_return": result["std_return"],
        }
        self._eval_history.append(row)
        write_header = not self._eval_csv_path.exists()
        with self._eval_csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["update", "mean_return", "std_return"])
            if write_header:
                w.writeheader()
            w.writerow(row)

    def maybe_eval(self, update: int, algo: Any, seed: int) -> Optional[Dict[str, float]]:
        """Run evaluation if needed."""
        if self.eval_freq <= 0 or update % self.eval_freq != 0:
            return None

        result = self._run_eval(algo, self.n_eval_episodes, seed + 12345)
        self._record_eval(update, result)
        self._log(
            {"eval/mean_return": result["mean_return"], "eval/std_return": result["std_return"]},
            step=update,
        )

        if self.save_best and result["mean_return"] > self.best_mean_reward:
            self.best_mean_reward = result["mean_return"]
            torch.save(algo.inference_actor().state_dict(), self.output_dir / "best_model.pt")
            self._log({"eval/best_mean_return": result["mean_return"]}, step=update)

        return result

    def on_training_end(self, update: int, algo: Any, seed: int) -> None:
        """Called at end of training."""
        if self.final_eval_episodes > 0:
            result = self._run_eval(algo, self.final_eval_episodes, seed + 999)
            last_update = self._eval_history[-1]["update"] if self._eval_history else -1
            if update != last_update:
                self._record_eval(update, result)
            (self.output_dir / "final_eval.json").write_text(
                json.dumps(
                    {
                        "env_id": self.env_id,
                        "update": update,
                        "n_eval_episodes": self.final_eval_episodes,
                        "mean_return": result["mean_return"],
                        "std_return": result["std_return"],
                    },
                    indent=2,
                )
            )
