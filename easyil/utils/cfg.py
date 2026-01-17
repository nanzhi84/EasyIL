from __future__ import annotations

import torch


def pick_device(device: str = "auto") -> torch.device:
    """Pick torch device based on string preference."""
    want = device.lower()
    if want in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
