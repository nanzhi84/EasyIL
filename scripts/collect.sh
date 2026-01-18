#!/usr/bin/env bash
set -e

# Collect expert trajectories from trained SAC model (standalone module)
# Usage: ./scripts/collect.sh model_path=path/to/model n_episodes=100 env.id=HalfCheetah-v5
#
# Examples:
#   ./scripts/collect.sh model_path=outputs/HalfCheetah-v5/sac/best_model n_episodes=100
#   ./scripts/collect.sh model_path=outputs/Hopper-v5/sac/best_model env.id=Hopper-v5 n_episodes=50

python -m collect_expert "$@"
