#!/bin/bash

# Seed values
# seeds=(0 1 2 3 4)
seeds=(0)

# K values
# Ks=(7 9 11 13 15 17)
Ks=(7)
# Loop over seed and K values
for K in "${Ks[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Running with seed $seed and K $K"
        python train.py --algo ra2c --env Humanoid-v4 --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1 --seed $((10000 + seed)) --hyperparams advantage_type:normal robust_type:rce robust_beta:0.0 bins:$K &
    done
    wait
done
