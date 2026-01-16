from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from easyil.expert.rl import register_expert_rl
from easyil.utils.cfg import drop_none, to_plain_dict


@register_expert_rl("sac")
def sac(cfg: DictConfig, env: Any, output_dir: str) -> Any:
    """Build SBX SAC algorithm for expert training."""
    from sbx import SAC

    out = Path(output_dir)
    raw_kwargs = to_plain_dict(cfg.kwargs)
    raw_kwargs.setdefault("tensorboard_log", str(out / "tb"))
    kwargs = drop_none(raw_kwargs)

    return SAC(str(cfg.policy), env, **kwargs)
