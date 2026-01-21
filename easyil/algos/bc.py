"""Behavior Cloning algorithm."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from omegaconf import DictConfig
from torch import nn

from easyil.actors import ChunkActor, build_actor


@dataclass
class BCConfig:
    """Configuration for Behavior Cloning."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.999
    obs_noise: float = 0.0
    grad_clip: float = 1.0


class BC:
    """
    Behavior Cloning algorithm.

    Learns a policy by supervised learning on expert demonstrations.
    Supports action chunk prediction and EMA for stable inference.
    """

    def __init__(
        self,
        actor: ChunkActor,
        cfg: BCConfig,
        device: torch.device,
    ) -> None:
        self.actor = actor.to(device)
        self.cfg = cfg
        self.device = device

        # EMA actor for inference
        self.ema_actor = copy.deepcopy(actor).to(device)
        for p in self.ema_actor.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        self._update_count = 0

    @property
    def obs_dim(self) -> int:
        return self.actor.cfg.obs_dim

    @property
    def act_dim(self) -> int:
        return self.actor.cfg.act_dim

    @property
    def obs_horizon(self) -> int:
        return self.actor.cfg.obs_horizon

    @property
    def action_horizon(self) -> int:
        return self.actor.cfg.action_horizon

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one gradient update step.

        Args:
            batch: Dict with 'obs' (B, obs_horizon, obs_dim) and 'actions' (B, action_horizon, act_dim)

        Returns:
            Dict with training metrics.
        """
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)

        # Optional observation noise
        if self.cfg.obs_noise > 0:
            obs = obs + torch.randn_like(obs) * self.cfg.obs_noise

        # Forward pass with AMP
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            pred = self.actor(obs)
            loss = torch.mean((pred - actions) ** 2)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update EMA
        self._update_ema()
        self._update_count += 1

        return {"loss": loss.item()}

    @torch.no_grad()
    def _update_ema(self) -> None:
        """Update EMA actor weights."""
        for ema_p, p in zip(self.ema_actor.parameters(), self.actor.parameters()):
            ema_p.data.mul_(self.cfg.ema_decay).add_(p.data, alpha=1 - self.cfg.ema_decay)

    def inference_actor(self) -> ChunkActor:
        """Return EMA actor for inference."""
        return self.ema_actor

    def state_dict(self) -> Dict[str, Any]:
        """Return saveable state."""
        return {
            "actor": self.actor.state_dict(),
            "ema_actor": self.ema_actor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self._update_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state."""
        self.actor.load_state_dict(state["actor"])
        self.ema_actor.load_state_dict(state["ema_actor"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._update_count = state.get("update_count", 0)


def _build_bc(cfg: DictConfig, obs_dim: int, act_dim: int, device: str) -> BC:
    """Factory function to build BC from config."""
    actor_cfg = cfg.get("actor", {})
    actor_cfg["type"] = "chunk"
    actor_cfg["obs_horizon"] = cfg.get("obs_horizon", 1)
    actor_cfg["action_horizon"] = cfg.get("action_horizon", 1)

    actor = build_actor(DictConfig(actor_cfg), obs_dim, act_dim)

    bc_cfg = BCConfig(
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        ema_decay=float(cfg.get("ema_decay", 0.999)),
        obs_noise=float(cfg.get("obs_noise", 0.0)),
        grad_clip=float(cfg.get("grad_clip", 1.0)),
    )

    return BC(actor=actor, cfg=bc_cfg, device=torch.device(device))
