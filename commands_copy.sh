#!/bin/bash

# Seed values
seeds=(0 1)


for seed in "${seeds[@]}"; do
    echo "Running with seed $seed and K"
    python train.py --algo ppo --env HumanoidPyBulletEnv-v0 --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1 --seed $((10000 + seed)) --hyperparams advantage_type:normal &
done
