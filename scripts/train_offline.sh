#!/usr/bin/env bash
set -e

# Train Diffusion BC on HalfCheetah (offline learning)
# All parameters are configured in easyil/conf/

python -m easyil.train_offline "$@"
