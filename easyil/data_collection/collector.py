from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.vec_env import VecNormalize
from tqdm import tqdm

from easyil.envs import make_env


@dataclass
class ExpertData:
    """Container for collected expert trajectories."""

    obs: np.ndarray
    actions: np.ndarray
    dones: np.ndarray
    rewards: Optional[np.ndarray] = None
    next_obs: Optional[np.ndarray] = None
    traj_lengths: List[int] = field(default_factory=list)

    @property
    def n_transitions(self) -> int:
        return len(self.obs)

    @property
    def n_trajectories(self) -> int:
        return len(self.traj_lengths)

    def save(self, path: Union[str, Path]) -> None:
        """Save to .npz file in difftune-compatible format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict: Dict[str, np.ndarray] = {
            "obs": self.obs,
            "action": self.actions,
            "done": self.dones,
        }
        if self.rewards is not None:
            save_dict["reward"] = self.rewards
        if self.next_obs is not None:
            save_dict["next_obs"] = self.next_obs

        np.savez(path, **save_dict)
        print(f"Saved {self.n_trajectories} trajectories ({self.n_transitions} transitions) to {path}")


class ExpertCollector:
    """Collect expert trajectories from a trained policy."""

    def __init__(
        self,
        model: Any,
        env_cfg: DictConfig,
        output_dir: Path,
        vecnormalize_path: Optional[Path] = None,
    ):
        """Initialize collector.

        Args:
            model: Trained SB3/SBX model with .predict() method.
            env_cfg: Environment configuration.
            output_dir: Output directory for monitor files.
            vecnormalize_path: Optional path to VecNormalize stats.
        """
        self.model = model
        self.env_cfg = env_cfg
        self.output_dir = Path(output_dir)
        self.vecnormalize_path = vecnormalize_path

    def collect(
        self,
        n_episodes: int,
        deterministic: bool = True,
        seed: int = 0,
        show_progress: bool = True,
    ) -> ExpertData:
        """Collect expert trajectories.

        Args:
            n_episodes: Number of episodes to collect.
            deterministic: Whether to use deterministic actions.
            seed: Random seed for environment.
            show_progress: Whether to show progress bar.

        Returns:
            ExpertData containing collected trajectories.
        """
        env = make_env(self.env_cfg, self.output_dir, seed=seed, training=False)

        if self.vecnormalize_path and self.vecnormalize_path.exists():
            if isinstance(env, VecNormalize):
                env = VecNormalize.load(str(self.vecnormalize_path), env.venv)
                env.training = False
                env.norm_reward = False
                print(f"Loaded VecNormalize from {self.vecnormalize_path}")

        all_obs: List[np.ndarray] = []
        all_actions: List[np.ndarray] = []
        all_rewards: List[np.ndarray] = []
        all_dones: List[np.ndarray] = []
        all_next_obs: List[np.ndarray] = []
        traj_lengths: List[int] = []

        episodes_collected = 0
        pbar = tqdm(total=n_episodes, desc="Collecting episodes", disable=not show_progress)

        obs = env.reset()
        current_traj_len = 0
        ep_obs: List[np.ndarray] = []
        ep_actions: List[np.ndarray] = []
        ep_rewards: List[np.ndarray] = []
        ep_dones: List[np.ndarray] = []
        ep_next_obs: List[np.ndarray] = []

        while episodes_collected < n_episodes:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            next_obs, reward, done, info = env.step(action)

            ep_obs.append(obs[0].copy())
            ep_actions.append(action[0].copy())
            ep_rewards.append(reward[0])
            ep_dones.append(done[0])
            ep_next_obs.append(next_obs[0].copy())
            current_traj_len += 1

            obs = next_obs

            if done[0]:
                all_obs.extend(ep_obs)
                all_actions.extend(ep_actions)
                all_rewards.extend(ep_rewards)
                all_dones.extend(ep_dones)
                all_next_obs.extend(ep_next_obs)
                traj_lengths.append(current_traj_len)

                episodes_collected += 1
                pbar.update(1)
                pbar.set_postfix({"traj_len": current_traj_len, "total_steps": sum(traj_lengths)})

                ep_obs = []
                ep_actions = []
                ep_rewards = []
                ep_dones = []
                ep_next_obs = []
                current_traj_len = 0

        pbar.close()
        env.close()

        return ExpertData(
            obs=np.array(all_obs, dtype=np.float32),
            actions=np.array(all_actions, dtype=np.float32),
            dones=np.array(all_dones, dtype=bool),
            rewards=np.array(all_rewards, dtype=np.float32),
            next_obs=np.array(all_next_obs, dtype=np.float32),
            traj_lengths=traj_lengths,
        )


def load_model_for_collection(
    model_path: Union[str, Path],
    env_cfg: DictConfig,
    output_dir: Path,
    seed: int = 0,
) -> Any:
    """Load a trained model for data collection.

    Args:
        model_path: Path to saved model (without extension).
        env_cfg: Environment configuration.
        output_dir: Output directory.
        seed: Random seed.

    Returns:
        Loaded model.
    """
    from sbx import SAC

    env = make_env(env_cfg, output_dir, seed=seed, training=False)
    model = SAC.load(str(model_path), env=env)
    return model
