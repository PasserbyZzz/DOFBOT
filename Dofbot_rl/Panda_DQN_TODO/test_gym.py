import numpy as np
import time
import copy
from scipy.spatial.transform import Rotation as R
import panda_env
import gymnasium as gym


if __name__ == '__main__':
    # Create our training environment - a cart with a pole that needs balancing
    env = gym.make("PandaEnv-v1", obs_mode="state", render_mode="human")
    print(env.observation_space)
    print(env.action_space)

    observation, info = env.reset(seed=42)
    for i in range(1000):
        while True:
            try:
                action = int(input("Action(0-6): "))
                if 0 <= action <= 6:
                    break  # 输入有效，退出循环
                else:
                    print("输入的数字不在1到7之间，请重新输入。")
            except ValueError:
                print("输入无效，请输入一个数字。")
        for j in range(10):
            observation, reward, terminated, truncated, info = env.step(action)
            print("reward: ", reward)
            if terminated:
                print("terminated")
                observation, info = env.reset(seed=42)
 
    env.close()