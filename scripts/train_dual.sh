#!/usr/bin/env bash
set -e

# Dual reward training experiments
# Runs two parallel experiments:
#   1. Train with custom reward, track groundtruth
#   2. Train with groundtruth, track custom reward
#
# Usage:
#   ./train_dual.sh env.id=HalfCheetah-v5
#   ./train_dual.sh env.id=Humanoid-v5

echo "=== Experiment 1: Custom reward training + Groundtruth tracking ==="
python -m easyil.train --config-name=sac \
    env.reward.model_path=/path/to/reward_model.pth \
    env.compare_reward.model_path=null \
    'hydra.run.dir=outputs/${env.id}/dual/${now:%Y-%m-%d_%H-%M-%S}_custom_train' \
    "$@"

echo ""
echo "=== Experiment 2: Groundtruth training + Custom reward tracking ==="
python -m easyil.train --config-name=sac \
    env.reward.model_path=null \
    env.compare_reward.model_path=/path/to/reward_model.pth \
    'hydra.run.dir=outputs/${env.id}/dual/${now:%Y-%m-%d_%H-%M-%S}_gt_train' \
    "$@"

echo ""
echo "=== Both experiments completed ==="
