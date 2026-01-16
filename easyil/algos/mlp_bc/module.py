"""MLP BC module implementation."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from easyil.algos import register_algo
from easyil.envs import infer_env_dims
from easyil.networks import MLPBackbone
from easyil.algos.mlp_bc.policy import MLPPolicy, MLPPolicyConfig


class MLPActionPredictor(nn.Module):
    """MLP that directly predicts action chunks from observation history."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        obs_horizon: int,
        action_horizon: int,
        hidden: int = 512,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        in_dim = obs_dim * obs_horizon
        out_dim = act_dim * action_horizon

        self.backbone = MLPBackbone(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            depth=depth,
            dropout=dropout,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: Observation tensor of shape (B, obs_horizon, obs_dim).

        Returns:
            Action chunk of shape (B, action_horizon, act_dim).
        """
        B = obs.shape[0]
        x = obs.reshape(B, -1)
        out = self.backbone(x)
        return out.view(B, self.action_horizon, self.act_dim)


@dataclass
class MLPBCModule:
    """Module for MLP-based Behavior Cloning."""

    net: nn.Module
    ema_net: nn.Module
    policy: MLPPolicy
    obs_dim: int
    act_dim: int
    obs_horizon: int
    action_horizon: int
    ema_decay: float = 0.999
    obs_noise: float = 0.0

    def to(self, device: torch.device) -> "MLPBCModule":
        self.net.to(device)
        self.ema_net.to(device)
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
        actions = batch["actions"]  # (B, Ta, act_dim)

        if self.obs_noise > 0.0:
            obs = obs + torch.randn_like(obs) * self.obs_noise

        pred = self.net(obs)
        loss = torch.mean((pred - actions) ** 2)
        return loss


@register_algo("mlp_bc")
def mlp_bc(cfg: DictConfig, eval_env: Any, output_dir: str) -> MLPBCModule:
    obs_dim, act_dim, act_clip = infer_env_dims(eval_env)

    obs_horizon = int(cfg.get("obs_horizon", 1))
    action_horizon = int(cfg.get("action_horizon", 1))
    ema_decay = float(cfg.get("ema_decay", 0.999))
    obs_noise = float(cfg.get("obs_noise", 0.0))

    net = MLPActionPredictor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        hidden=int(cfg.model.get("hidden", 512)),
        depth=int(cfg.model.get("depth", 4)),
        dropout=float(cfg.model.get("dropout", 0.0)),
    )

    ema_net = copy.deepcopy(net)
    for param in ema_net.parameters():
        param.requires_grad = False

    policy_cfg = MLPPolicyConfig(
        action_clip=float(cfg.policy.get("action_clip", act_clip)),
        use_amp=bool(cfg.policy.get("use_amp", True)),
    )
    policy = MLPPolicy(net=ema_net, cfg=policy_cfg)

    return MLPBCModule(
        net=net,
        ema_net=ema_net,
        policy=policy,
        obs_dim=obs_dim,
        act_dim=act_dim,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        ema_decay=ema_decay,
        obs_noise=obs_noise,
    )
