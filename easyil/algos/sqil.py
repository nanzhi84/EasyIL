"""Soft Q Imitation Learning (SQIL)."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from easyil.actors import GaussianActor, build_actor
from easyil.critics import QCritic, build_critic


@dataclass
class SQILConfig:
    """Configuration for SQIL."""

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: float | None = None

    # SQIL specific
    expert_reward: float = 1.0
    policy_reward: float = 0.0

    grad_clip: float = 1.0


class SQIL:
    """
    Soft Q Imitation Learning.

    Assigns reward=1 to expert data and reward=0 to policy data,
    then trains with SAC-style off-policy learning.

    Simpler than GAIL (no discriminator training) but effective.
    """

    def __init__(
        self,
        actor: GaussianActor,
        critic: QCritic,
        cfg: SQILConfig,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.device = device

        # Networks
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.critic_target = copy.deepcopy(critic).to(device)

        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # Entropy coefficient
        self.log_alpha = torch.tensor(0.0, requires_grad=cfg.auto_alpha, device=device)
        self.alpha = cfg.alpha
        if cfg.auto_alpha:
            self.target_entropy = cfg.target_entropy or -float(actor.cfg.act_dim)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.actor_lr)

        self._update_count = 0

    @property
    def obs_dim(self) -> int:
        return self.actor.cfg.obs_dim

    @property
    def act_dim(self) -> int:
        return self.actor.cfg.act_dim

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one gradient update step.

        Batch should contain 'source' field to assign SQIL rewards.
        """
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device).float().unsqueeze(-1)
        source = batch["source"]

        # Handle source as either string array or tensor
        if isinstance(source, torch.Tensor):
            source = source.cpu().numpy()

        # SQIL reward assignment
        expert_mask = torch.tensor(source == "expert", device=self.device).float()
        rewards = (
            expert_mask * self.cfg.expert_reward + (1 - expert_mask) * self.cfg.policy_reward
        ).unsqueeze(-1)

        # Update critic
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            target_q = self.critic_target.q_min(next_obs, next_action)
            target_q = rewards + self.cfg.gamma * (1 - dones) * (target_q - self.alpha * next_log_prob.unsqueeze(-1))

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_optimizer.step()

        # Update actor
        new_action, log_prob = self.actor.sample(obs)
        q_new = self.critic.q_min(obs, new_action)
        actor_loss = (self.alpha * log_prob.unsqueeze(-1) - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss = 0.0
        if self.cfg.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft update target
        self._soft_update_target()
        self._update_count += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "expert_ratio": expert_mask.mean().item(),
        }

    @torch.no_grad()
    def _soft_update_target(self) -> None:
        """Soft update target critic."""
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.mul_(1 - self.cfg.tau).add_(p.data, alpha=self.cfg.tau)

    def inference_actor(self) -> GaussianActor:
        """Return actor for inference."""
        return self.actor

    def state_dict(self) -> Dict[str, Any]:
        """Return saveable state."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "update_count": self._update_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state."""
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.log_alpha.data.fill_(state["log_alpha"])
        self._update_count = state.get("update_count", 0)


def _build_sqil(cfg: DictConfig, obs_dim: int, act_dim: int, device: str) -> SQIL:
    """Factory function to build SQIL from config."""
    # Build actor
    actor_cfg = cfg.get("actor", {})
    actor_cfg["type"] = "gaussian"
    actor = build_actor(DictConfig(actor_cfg), obs_dim, act_dim)

    # Build critic
    critic_cfg = cfg.get("critic", {})
    critic_cfg["type"] = "q"
    critic = build_critic(DictConfig(critic_cfg), obs_dim, act_dim)

    sqil_cfg = SQILConfig(
        actor_lr=float(cfg.get("actor_lr", 3e-4)),
        critic_lr=float(cfg.get("critic_lr", 3e-4)),
        gamma=float(cfg.get("gamma", 0.99)),
        tau=float(cfg.get("tau", 0.005)),
        alpha=float(cfg.get("alpha", 0.2)),
        auto_alpha=bool(cfg.get("auto_alpha", True)),
        expert_reward=float(cfg.get("expert_reward", 1.0)),
        policy_reward=float(cfg.get("policy_reward", 0.0)),
        grad_clip=float(cfg.get("grad_clip", 1.0)),
    )

    return SQIL(
        actor=actor,
        critic=critic,
        cfg=sqil_cfg,
        device=torch.device(device),
    )
