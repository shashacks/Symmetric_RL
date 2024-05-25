#!/bin/bash

# Seed values from 10000 to 10004 (reduce the seed number for test)
seeds=($(seq 10000 10004))

# (ra2c, sppo) (when alpha=1.0, beta=0.0, it's DA2C or DPPO reported in the paper. Also, when beta > 0.0, then it's DSA2C or DSPPO)
algo='sppo'

# (Atari games)
env='WizardOfWorNoFrameskip-v4'

robust_alpha=0.5 
robust_beta=1.0 # please refer to Table 4 in Appendix Section

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
     --hyperparams advantage_type:normal robust_type:rce robust_alpha:${robust_alpha} robust_beta:${robust_beta} \
      rce_Z:-1 decay_term:0 noisy_type:${noisy_type} noisy_value:${noisy_value} ent_coef:${ent_coef} normalize_advantage:${normalize_advantage}
done

