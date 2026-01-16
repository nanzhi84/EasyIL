#!/usr/bin/env bash
set -e

# Train SAC (online RL)
# Usage:
#   ./train_online.sh                               # default SAC on HalfCheetah
#   ./train_online.sh env.id=Humanoid-v5            # change environment

python -m easyil.train --config-name=sac "$@"
