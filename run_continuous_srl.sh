#!/bin/bash

# Seed values from 10000 to 10029 (reduce the seed number for test)
seeds=($(seq 10000 10029))

# (sa2c, sppo) (when alpha=1.0, beta=0.0, it's DA2C or DPPO reported in the paper. Also, when beta > 0.0, then it's DSA2C or DSPPO)
algo='sppo'

# ('Ant-v4', 'HumanoidStandup-v4', 'Walker2d-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Swimmer-v4', 'InvertedDoublePendulum-v4', 'BipedalWalker-v3', 'LunarLanderContinuous-v2')
env='Ant-v4'

robust_alpha=0.5 
robust_beta=20.0 # please refer to Table 5 in Appendix Section

# 'none', 'normal'
noisy_type='none'
noisy_value=0.0

if [ "$noisy_type" = 'normal' ]; then
    noisy_value=0.05
fi

if [ "${algo}" = 'sppo' ]; then
    normalize_advantage=True
else
    normalize_advantage=False
fi

for seed in "${seeds[@]}"; do
    python train.py --algo ${algo} --env ${env} --eval-freq 5000 --eval-episodes 10 --n-eval-envs 1 --seed ${seed} \
     --hyperparams advantage_type:normal robust_type:rce robust_alpha:${robust_alpha} robust_beta:${robust_beta} \
      bins:11 rce_Z:-1 decay_term:0 noisy_type:${noisy_type} noisy_value:${noisy_value} normalize_advantage:${normalize_advantage}
done