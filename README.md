# Symmetric Reinforcement Learning Loss for Robust Learning on Diverse Tasks and Model Scales

Reinforcement learning (RL) training is inherently unstable due to factors such as moving targets and high gradient variance. Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning from AI Feedback (RLAIF) can introduce additional difficulty. Differing preferences can complicate the alignment process, and prediction errors in a trained reward model can become more severe as the LLM generates unseen outputs. To enhance training robustness, RL has adopted techniques from supervised learning, such as ensembles and layer normalization. In this work, we improve the stability of RL training by adapting the reverse cross entropy (RCE) from supervised learning for noisy data to define a symmetric RL loss. We demonstrate performance improvements across various tasks and scales. We conduct experiments in discrete action tasks (Atari games) and continuous action space tasks (MuJoCo benchmark and Box2D) using Symmetric A2C (SA2C) and Symmetric PPO (SPPO), with and without added noise with especially notable performance in SPPO across different hyperparameters. Furthermore, we validate the benefits of the symmetric RL loss when using SPPO for large language models through improved performance in RLHF tasks, such as IMDB positive sentiment sentiment and TL;DR summarization tasks. 

We implement our method based on [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3). This repository is for Atari games, MuJoCo benchmark, and Box2D. Please refer to LLM tasks [here](https://github.com/shashacks/Symmetric_tril). 


### Install using pip
Install the Stable Baselines3 package:
```
conda create -n srl python=3.8.17
conda activate srl
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

```

## Example
Please check each script file for detailed information

To run A2C and PPO for Atari games
```python
./run_discrete_basic.sh
```

To run A2C and PPO for MuJoCo benchmark and Box2d environments
```python
./run_continuous_basic.sh
```

To run SA2C and SPPO for Atari games
```python
./run_discrete_srl.sh
```

To run DSA2C and DSPPO (DA2C and DPPO) for MuJoCo benchmark and Box2d environments
```python
./run_continuous_srl.sh
```

##