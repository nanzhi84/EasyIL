#!/usr/bin/env bash
set -e

# Train Diffusion BC (offline learning)
# Usage:
#   ./train_offline.sh                              # default diffusion_bc
#   ./train_offline.sh --config-name=mlp_bc         # use MLP BC
#   ./train_offline.sh env.id=Humanoid-v5           # change environment

python -m easyil.train "$@"
