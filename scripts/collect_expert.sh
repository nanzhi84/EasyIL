#!/usr/bin/env bash
set -e

# Collect expert trajectories from trained SAC model
# Usage: ./collect_expert.sh model_path=outputs/xxx/best_model n_episodes=100

python -m easyil.collect "$@"
