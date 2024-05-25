#!/bin/bash

# Seed values from 10000 to 10004 (reduce the seed number for test)
seeds=($(seq 10000 10004))

# (a2c, ppo) 
algo='ppo'

# (Atari games)
env='WizardOfWorNoFrameskip-v4'

# 'none', 'normal'
noisy_type='reward_flip'
noisy_value=0.0


if [ "$noisy_type" = 'reward_flip' ]; then
    noisy_value=0.1
fi

ent_coef=0.01

if [ "${algo}" = 'sppo' ]; then
    normalize_advantage=True
else
    normalize_advantage=False
fi

for seed in "${seeds[@]}"; do
    python train.py --algo ${algo} --env ${env} --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1 --seed ${seed} \
     --hyperparams advantage_type:normal ent_coef:${ent_coef} noisy_type:${noisy_type} noisy_value:${noisy_value} normalize_advantage:${normalize_advantage}
done

