"""Callbacks for online RL training."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from easyil.loggers import Logger


class OnlineCallback:
    """Callback for online training: logging, evaluation, checkpointing."""

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

        self.log_freq = train_cfg.get("log_freq", 1000)
        self.eval_freq = train_cfg.get("eval_freq", 10000)
        self.n_eval_episodes = train_cfg.get("n_eval_episodes", 10)
        self.save_best = train_cfg.get("save_best", True)
        self.checkpoint_freq = train_cfg.get("checkpoint_freq", 100000)
        self.final_eval_episodes = train_cfg.get("final_eval_episodes", 10)

        self.best_mean_reward = -np.inf
        self._eval_history: List[dict] = self._load_eval_history()
        self._last_logged = 0
        self._last_eval = 0
        self._last_ckpt = 0
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
                        "timesteps": int(r["timesteps"]),
                        "mean_return": float(r["mean_return"]),
                        "std_return": float(r["std_return"]),
                    }
                    for r in csv.DictReader(f)
                ],
                key=lambda d: d["timesteps"],
            )

    def _log(self, metrics: Dict[str, Any], step: int) -> None:
        self._logger.log(metrics, step=step)

    def log_train(self, timesteps: int, metrics: Dict[str, float]) -> None:
        """Log training metrics."""
        if self.log_freq <= 0 or (timesteps - self._last_logged) < self.log_freq:
            return
        self._last_logged = timesteps
        log_metrics = {f"train/{k}": v for k, v in metrics.items()}
        log_metrics["time/total_timesteps"] = timesteps
        self._log(log_metrics, step=timesteps)

    def save_checkpoint(self, timesteps: int, algo: Any) -> None:
        """Save checkpoint."""
        if self.checkpoint_freq <= 0 or (timesteps - self._last_ckpt) < self.checkpoint_freq:
            return
        self._last_ckpt = timesteps
        torch.save(algo.state_dict(), self.output_dir / f"checkpoint_{timesteps}.pt")

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

            ep_return = 0.0
            done = False

            while not done:
                obs_normalized = obs
                if obs_mean is not None:
                    obs_normalized = (obs - obs_mean) / obs_std

                obs_t = torch.from_numpy(obs_normalized).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = actor.deterministic(obs_t)
                action_np = action.squeeze(0).cpu().numpy()

                next_obs, reward, done, info = self.eval_env.step(action_np)
                if isinstance(next_obs, tuple):
                    next_obs = next_obs[0]
                obs = np.asarray(next_obs, dtype=np.float32)
                ep_return += float(reward)

            returns.append(ep_return)

        actor.train()
        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
        }

    def _record_eval(self, timesteps: int, result: Dict[str, float]) -> None:
        row = {
            "timesteps": timesteps,
            "mean_return": result["mean_return"],
            "std_return": result["std_return"],
        }
        self._eval_history.append(row)
        write_header = not self._eval_csv_path.exists()
        with self._eval_csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timesteps", "mean_return", "std_return"])
            if write_header:
                w.writeheader()
            w.writerow(row)

    def maybe_eval(self, timesteps: int, algo: Any, seed: int) -> Optional[Dict[str, float]]:
        """Run evaluation if needed."""
        if self.eval_freq <= 0 or (timesteps - self._last_eval) < self.eval_freq:
            return None
        self._last_eval = timesteps

        result = self._run_eval(algo, self.n_eval_episodes, seed + 12345)
        self._record_eval(timesteps, result)
        self._log(
            {"eval/mean_return": result["mean_return"], "eval/std_return": result["std_return"]},
            step=timesteps,
        )

        if self.save_best and result["mean_return"] > self.best_mean_reward:
            self.best_mean_reward = result["mean_return"]
            torch.save(algo.inference_actor().state_dict(), self.output_dir / "best_model.pt")
            self._log({"eval/best_mean_return": result["mean_return"]}, step=timesteps)

        return result

    def on_training_end(self, timesteps: int, algo: Any, seed: int) -> None:
        """Called at end of training."""
        if self.final_eval_episodes > 0:
            result = self._run_eval(algo, self.final_eval_episodes, seed + 999)
            last_ts = self._eval_history[-1]["timesteps"] if self._eval_history else -1
            if timesteps != last_ts:
                self._record_eval(timesteps, result)
            (self.output_dir / "final_eval.json").write_text(
                json.dumps(
                    {
                        "env_id": self.env_id,
                        "timesteps": timesteps,
                        "n_eval_episodes": self.final_eval_episodes,
                        "mean_return": result["mean_return"],
                        "std_return": result["std_return"],
                    },
                    indent=2,
                )
            )
