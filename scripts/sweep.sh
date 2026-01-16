#!/usr/bin/env bash
set -e

# Hyperparameter sweep for diffusion policy
# Uses Hydra multirun to sweep across parameter combinations
#
# Sweep parameters:
# - learning_rate: [1e-4, 3e-4]
# - batch_size: [32, 64, 128]

echo "Starting hyperparameter sweep..."

python -m easyil.train --multirun \
    train.dataset.num_trajs=1 \
    train.learning_rate=1e-4,3e-4 \
    train.batch_size=32,64,128 \
    algo.scheduler.train_steps=20 \
    algo.model.type=unet \
    algo.action_horizon=4 \
    hydra.sweep.dir=outputs/sweep \
    hydra.sweep.subdir='lr${train.learning_rate}_bs${train.batch_size}_T${algo.scheduler.train_steps}_${algo.model.type}_L${algo.action_horizon}'

echo "Sweep completed! Results saved in outputs/sweep/"
