#!/usr/bin/env bash
set -e

# Train SAC on HalfCheetah (online RL)
# All parameters are configured in easyil/conf/

python -m easyil.train_online "$@"
