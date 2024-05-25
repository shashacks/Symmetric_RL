import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_action_dim

"""
wrapper for discretizing continuous action space
"""
def discretizing_wrapper(env, K):
    """
    # discretize each action dimension to K bins
    """

    env.orig_step_ = env.step
    
    action_low, action_high = env.action_space.low, env.action_space.high
    naction = get_action_dim(env.action_space)
    action_table = np.reshape([np.linspace(action_low[i], action_high[i], K) for i in range(naction)], [naction, K])
    assert action_table.shape == (naction, K)

    def discretizing_step(action):
        # action is a sequence of discrete indices
        action_cont = action_table[np.arange(naction), action]
        obs, rews, dones, infos = env.orig_step_(action_cont)
        return obs, rews, dones, infos

    # change observation space
    env.action_space = spaces.MultiDiscrete([K for _ in range(naction)])
    env.step = discretizing_step

    return env