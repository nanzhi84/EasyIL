"""Configuration utilities (JAX)."""
from __future__ import annotations

import jax


def pick_device(device: str = "auto") -> str:
    """Pick JAX device based on string preference.

    Args:
        device: Device preference ("auto", "gpu", "cpu", "tpu").

    Returns:
        Device string for JAX.
    """
    want = device.lower()

    if want == "auto":
        # JAX automatically uses GPU/TPU if available
        backend = jax.default_backend()
        return backend

    if want in {"gpu", "cuda"}:
        try:
            devices = jax.devices("gpu")
            if devices:
                return "gpu"
        except RuntimeError:
            pass
        return "cpu"

    if want == "tpu":
        try:
            devices = jax.devices("tpu")
            if devices:
                return "tpu"
        except RuntimeError:
            pass
        return "cpu"

    return "cpu"


def get_default_device():
    """Get the default JAX device."""
    return jax.devices()[0]
