"""SAC algorithm for expert training."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from collect_expert.rl import register_expert_rl


@register_expert_rl("sac")
def sac(cfg: DictConfig, env: Any, output_dir: str) -> Any:
    """Build SBX SAC algorithm for expert training."""
    from sbx import SAC

    out = Path(output_dir)
    kwargs = dict(OmegaConf.to_container(cfg.kwargs, resolve=True))
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs.setdefault("tensorboard_log", str(out / "tb"))

    return SAC(str(cfg.policy), env, **kwargs)
