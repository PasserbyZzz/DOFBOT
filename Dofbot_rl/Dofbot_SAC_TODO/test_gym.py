import numpy as np
import time
import copy
from scipy.spatial.transform import Rotation as R
import dofbotGymReachEnv
import gymnasium as gym

def make_env(env_id, idx):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        print("observation space:", env.observation_space)
        print("action space:", type(env.action_space))
        return env
    
    return thunk

if __name__ == '__main__':
    # env = gym.make("DofbotEnv-v1", render_mode="rgb_array")
    env = gym.make("DofbotReachEnv-v1", render_mode="human")
    observation, info = env.reset(seed=42)
    print("obs shape:", observation.shape)
    print("observation space:", env.observation_space)
    print("action space:", env.action_space)
    step = 0
    while True:
        action = np.array([0, 0, 0, 0, 0])
        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.5)
        step += 1
        print("step:", step)
        if terminated or truncated:
            print("End Episode")
            observation, info = env.reset(seed=42)
            step = 0
    env.close()