#!/bin/bash

# Seed values from 10000 to 10029 (reduce the seed number for test)
seeds=($(seq 10000 10029))

# (a2c, ppo) 
algo='ppo'

# ('Ant-v4', 'HumanoidStandup-v4', 'Walker2d-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Swimmer-v4', 'InvertedDoublePendulum-v4', 'BipedalWalker-v3', 'LunarLanderContinuous-v2')
env='Ant-v4'

# 'none', 'normal'
noisy_type='none'
noisy_value=0.0

if [ "$noisy_type" = 'normal' ]; then
    noisy_value=0.05
fi

if [ "${algo}" = 'ppo' ]; then
    normalize_advantage=True
else
    normalize_advantage=False
fi

for seed in "${seeds[@]}"; do
    python train.py --algo ${algo} --env ${env} --eval-freq 5000 --eval-episodes 10 --n-eval-envs 1 --seed ${seed} \
     --hyperparams advantage_type:normal noisy_type:${noisy_type} noisy_value:${noisy_value} normalize_advantage:${normalize_advantage}       
done