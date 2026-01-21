"""Online data collector for environment rollouts."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from easyil.buffers import UnifiedBuffer


class OnlineCollector:
    """
    Collector that gathers experience by interacting with environment.

    Supports both single-step and chunk-based action selection.
    """

    def __init__(
        self,
        env: Any,
        actor: Any,
        device: torch.device,
        source_label: str = "policy",
        obs_norm_stats: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.env = env
        self.actor = actor
        self.device = device
        self.source_label = source_label
        self.obs_norm_stats = obs_norm_stats

        self._obs: Optional[np.ndarray] = None
        self._episode_returns: List[float] = []
        self._current_return: float = 0.0

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        self._obs = np.asarray(obs, dtype=np.float32)
        self._current_return = 0.0
        return self._obs

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation if stats are available."""
        if self.obs_norm_stats is None:
            return obs
        return (obs - self.obs_norm_stats["mean"]) / self.obs_norm_stats["std"]

    @torch.no_grad()
    def collect_steps(
        self,
        buffer: "UnifiedBuffer",
        n_steps: int,
        deterministic: bool = False,
    ) -> Dict[str, float]:
        """
        Collect n_steps of experience and add to buffer.

        Returns:
            Dict with collection statistics.
        """
        if self._obs is None:
            self.reset()

        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        rew_list: List[float] = []
        done_list: List[bool] = []
        next_obs_list: List[np.ndarray] = []

        n_episodes = 0
        total_reward = 0.0

        for _ in range(n_steps):
            obs_normalized = self._normalize_obs(self._obs)
            obs_t = torch.from_numpy(obs_normalized).float().unsqueeze(0).to(self.device)

            if deterministic:
                action = self.actor.deterministic(obs_t)
            else:
                if hasattr(self.actor, "sample"):
                    action, _ = self.actor.sample(obs_t)
                else:
                    action = self.actor.deterministic(obs_t)

            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, done, info = self.env.step(action_np)
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            next_obs = np.asarray(next_obs, dtype=np.float32)

            obs_list.append(self._obs)
            act_list.append(action_np)
            rew_list.append(float(reward))
            done_list.append(bool(done))
            next_obs_list.append(next_obs)

            self._current_return += float(reward)
            total_reward += float(reward)

            if done:
                self._episode_returns.append(self._current_return)
                n_episodes += 1
                self._current_return = 0.0
                self._obs = self.reset()
            else:
                self._obs = next_obs

        # Add to buffer
        data = {
            "obs": np.stack(obs_list),
            "actions": np.stack(act_list),
            "rewards": np.array(rew_list, dtype=np.float32),
            "dones": np.array(done_list, dtype=bool),
            "next_obs": np.stack(next_obs_list),
            "source": np.full(n_steps, self.source_label, dtype=object),
        }
        buffer.extend(data)

        return {
            "n_steps": n_steps,
            "n_episodes": n_episodes,
            "mean_reward": total_reward / n_steps,
        }

    def get_episode_returns(self) -> List[float]:
        """Get all completed episode returns."""
        return self._episode_returns.copy()

    def clear_episode_returns(self) -> None:
        """Clear episode return history."""
        self._episode_returns.clear()
