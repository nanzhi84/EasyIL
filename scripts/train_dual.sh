#!/usr/bin/env bash
set -e

# Dual reward training experiments
# Runs two parallel experiments:
#   1. Train with custom reward, track groundtruth
#   2. Train with groundtruth, track custom reward
#
# Usage:
#   ./train_dual.sh env=halfcheetah  # uses HalfCheetah reward model
#   ./train_dual.sh env=humanoid     # uses Humanoid reward model

echo "=== Experiment 1: Custom reward training + Groundtruth tracking ==="
python -m easyil.train_online \
    env.reward.enabled=true \
    env.compare_reward.enabled=true \
    env.compare_reward.model_path=null \
    'hydra.run.dir=outputs/${env.id}/dual/${now:%Y-%m-%d_%H-%M-%S}_custom_train' \
    "$@"

echo ""
echo "=== Experiment 2: Groundtruth training + Custom reward tracking ==="
python -m easyil.train_online \
    env.reward.enabled=false \
    env.compare_reward.enabled=true \
    'env.compare_reward.model_path=${env.reward.model_path}' \
    'hydra.run.dir=outputs/${env.id}/dual/${now:%Y-%m-%d_%H-%M-%S}_gt_train' \
    "$@"

echo ""
echo "=== Both experiments completed ==="