#!/usr/bin/env bash
set -e

# Hyperparameter sweep for diffusion policy (1 trajectory only)
# Uses Hydra multirun to sweep across parameter combinations

# Sweep parameters:
# - learning_rate: [1e-4, 3e-4, 1e-3]
# - batch_size: [32, 64, 128]
# - scheduler.train_steps (diffusion_timesteps): [20, 50, 100]
# - model.type: [mlp, unet]
# - action_horizon (L): [1, 4, 8]

# Fixed: num_trajs=1 (only 1 trajectory)

echo "Starting hyperparameter sweep with 1 trajectory..."

python -m easyil.train --multirun \
    train.dataset.num_trajs=1 \
    train.learning_rate=1e-4,3e-4 \
    train.batch_size=32,64,128 \
    algo.scheduler.train_steps=20 \
    algo.model.type=unet \
    algo.action_horizon=4 \
    hydra.sweep.dir=outputs/sweep_1traj \
    hydra.sweep.subdir='lr${train.learning_rate}_bs${train.batch_size}_T${algo.scheduler.train_steps}_${algo.model.type}_L${algo.action_horizon}'

echo "Sweep completed! Results saved in outputs/sweep_1traj/"
