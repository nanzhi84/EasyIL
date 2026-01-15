"""Diffusion BC module implementation."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from easyil.core.registry import register_algo
from easyil.algos.diffusion_bc.networks import build_noise_predictor
from easyil.algos.diffusion_bc.policy import DiffusionPolicy, DiffusionPolicyConfig
from easyil.algos.diffusion_bc.schedulers import BaseScheduler, build_scheduler


@dataclass
class DiffusionBCModule:
    """Module for Diffusion-based Behavior Cloning."""

    net: nn.Module
    ema_net: nn.Module
    scheduler: BaseScheduler
    policy: DiffusionPolicy
    obs_dim: int
    act_dim: int
    obs_horizon: int
    action_horizon: int
    ema_decay: float = 0.999
    obs_noise: float = 0.0

    def to(self, device: torch.device) -> "DiffusionBCModule":
        self.net.to(device)
        self.ema_net.to(device)
        self.scheduler.to(device)
        self.policy.to(device)
        return self

    @torch.no_grad()
    def update_ema(self) -> None:
        """Update EMA network with current network weights."""
        for ema_param, param in zip(self.ema_net.parameters(), self.net.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def use_ema_for_inference(self) -> None:
        """Switch policy to use EMA network for inference."""
        self.policy.net = self.ema_net

    def use_train_net_for_inference(self) -> None:
        """Switch policy to use training network for inference."""
        self.policy.net = self.net

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        obs = batch["obs"]  # (B, To, obs_dim)
        x0 = batch["actions"]  # (B, Ta, act_dim)

        B = x0.shape[0]
        device = x0.device

        if self.obs_noise > 0.0:
            obs = obs + torch.randn_like(obs) * self.obs_noise

        t = torch.randint(low=0, high=int(self.scheduler.T), size=(B,), device=device)
        noise = torch.randn_like(x0)
        xt = self.scheduler.q_sample(x0, t, noise)
        pred = self.net(xt, t, obs)
        loss = torch.mean((pred - noise) ** 2)
        return loss


def _infer_dims(eval_env: Any) -> tuple[int, int, float]:
    obs_space = eval_env.observation_space
    act_space = eval_env.action_space
    obs_dim = int(obs_space.shape[0])
    act_dim = int(act_space.shape[0])
    clip = float(getattr(act_space, "high", [1.0])[0])
    return obs_dim, act_dim, clip


@register_algo("diffusion_bc")
def diffusion_bc(cfg: DictConfig, eval_env: Any, output_dir: str) -> DiffusionBCModule:
    obs_dim, act_dim, act_clip = _infer_dims(eval_env)

    obs_horizon = int(cfg.get("obs_horizon", 1))
    action_horizon = int(cfg.get("action_horizon", 16))
    ema_decay = float(cfg.get("ema_decay", 0.999))
    obs_noise = float(cfg.get("obs_noise", 0.0))

    scheduler = build_scheduler(cfg.scheduler)
    net = build_noise_predictor(cfg.model, obs_dim, act_dim, obs_horizon, action_horizon)

    ema_net = copy.deepcopy(net)
    for param in ema_net.parameters():
        param.requires_grad = False

    policy_cfg = DiffusionPolicyConfig(
        action_clip=float(cfg.policy.get("action_clip", act_clip)),
        use_amp=bool(cfg.policy.get("use_amp", True)),
    )
    policy = DiffusionPolicy(net=ema_net, scheduler=scheduler, cfg=policy_cfg)

    return DiffusionBCModule(
        net=net,
        ema_net=ema_net,
        scheduler=scheduler,
        policy=policy,
        obs_dim=obs_dim,
        act_dim=act_dim,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        ema_decay=ema_decay,
        obs_noise=obs_noise,
    )
