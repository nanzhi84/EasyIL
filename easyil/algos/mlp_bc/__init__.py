"""MLP-based Behavior Cloning algorithm."""

from easyil.algos.mlp_bc.module import MLPBCModule, mlp_bc
from easyil.algos.mlp_bc.policy import MLPPolicy, MLPPolicyConfig

__all__ = [
    "MLPBCModule",
    "mlp_bc",
    "MLPPolicy",
    "MLPPolicyConfig",
]
