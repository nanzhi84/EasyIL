"""Diffusion-based policy for action chunk generation."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from easyil.algos.diffusion_bc.schedulers import BaseScheduler


@dataclass
class DiffusionPolicyConfig:
    action_clip: float = 1.0
    use_amp: bool = True


class DiffusionPolicy(nn.Module):
    """Policy that generates action chunks via iterative denoising."""

    def __init__(
        self, net: nn.Module, scheduler: BaseScheduler, cfg: DiffusionPolicyConfig
    ) -> None:
        super().__init__()
        self.net = net
        self.scheduler = scheduler
        self.cfg = cfg

    @torch.no_grad()
    def sample_action_chunk(
        self,
        obs: torch.Tensor,
        act_dim: int,
        action_horizon: int,
    ) -> torch.Tensor:
        """Sample an action chunk conditioned on obs history.

        Args:
            obs: Observation tensor of shape (B, obs_horizon, obs_dim).
            act_dim: Action dimension.
            action_horizon: Number of actions to predict.

        Returns:
            Action chunk of shape (B, action_horizon, act_dim).
        """
        device = obs.device
        B = obs.shape[0]

        # Pre-encode obs once for speed
        obs_emb = None
        if hasattr(self.net, "encode_obs"):
            obs_emb = self.net.encode_obs(obs)

        # Start from pure noise
        xt = torch.randn((B, action_horizon, act_dim), device=device)

        timesteps = self.scheduler.infer_timesteps.to(device)
        t_list = timesteps.tolist()

        amp_ctx = torch.autocast(
            device_type=str(device.type),
            dtype=torch.float16,
            enabled=bool(self.cfg.use_amp),
        )

        with amp_ctx:
            for i, t in enumerate(t_list):
                t_int = int(t)
                t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)
                eps = self.net(xt, t_batch, obs, obs_emb=obs_emb)

                t_prev = int(t_list[i + 1]) if i + 1 < len(t_list) else 0
                xt = self.scheduler.step(xt, t_int, t_prev, eps)

            x0 = xt

        if self.cfg.action_clip is not None:
            x0 = torch.clamp(x0, -float(self.cfg.action_clip), float(self.cfg.action_clip))
        return x0

    @torch.no_grad()
    def sample_action_chunk_at_step(
        self,
        obs: torch.Tensor,
        act_dim: int,
        action_horizon: int,
        stop_step: int,
        return_x0: bool = False,
    ) -> torch.Tensor:
        """Sample action chunk at a specific diffusion step for analysis.

        Args:
            obs: Observation tensor of shape (B, obs_horizon, obs_dim).
            act_dim: Action dimension.
            action_horizon: Action chunk length.
            stop_step: Diffusion step to stop at (0 = fully denoised).
            return_x0: If True, return predicted x0; otherwise return noisy xt.

        Returns:
            Action chunk of shape (B, action_horizon, act_dim).
        """
        device = obs.device
        B = obs.shape[0]

        obs_emb = None
        if hasattr(self.net, "encode_obs"):
            obs_emb = self.net.encode_obs(obs)

        xt = torch.randn((B, action_horizon, act_dim), device=device)

        timesteps = self.scheduler.infer_timesteps.to(device)
        t_list = timesteps.tolist()

        amp_ctx = torch.autocast(
            device_type=str(device.type),
            dtype=torch.float16,
            enabled=bool(self.cfg.use_amp),
        )

        x0_pred = xt

        with amp_ctx:
            for i, t in enumerate(t_list):
                t_int = int(t)
                if t_int < stop_step:
                    break

                t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)
                eps = self.net(xt, t_batch, obs, obs_emb=obs_emb)

                x0_pred = self.scheduler.predict_x0_from_eps(xt, t_batch, eps)

                if t_int == stop_step:
                    break

                t_prev = int(t_list[i + 1]) if i + 1 < len(t_list) else 0
                xt = self.scheduler.step(xt, t_int, t_prev, eps)

        out = x0_pred if (return_x0 or stop_step == 0) else xt

        if self.cfg.action_clip is not None:
            out = torch.clamp(out, -float(self.cfg.action_clip), float(self.cfg.action_clip))
        return out
