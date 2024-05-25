#!/bin/bash

# Loop from seed 10000 to 10010
for seed in {10000..10009}
do
    echo "Running training with seed $seed"
    python train.py --algo ppo --env Ant-v4 --eval-freq 5000 --eval-episodes 10 --n-eval-envs 1 --seed $seed --hyperparams advantage_type:normal noisy_type:none noisy_value:0.0 normalize_advantage:False
done