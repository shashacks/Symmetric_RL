from gymnasium.wrappers import FrameStack
import gymnasium as gym

class AtariFrameStack(FrameStack):
    def __init__(self, env: gym.Env, num_stack: int, lz4_compress: bool = False):
        super().__init__(env, num_stack, lz4_compress)

    def step(self, action):
        obs, reward, terminated, _, info = super().step(action)
        return obs, reward, terminated, info
    
    def reset(self):
        obs, _ = super().reset()
        return obs