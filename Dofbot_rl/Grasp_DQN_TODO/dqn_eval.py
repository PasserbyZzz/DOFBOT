import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register
import panda_env


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    step=0
    while len(episodic_returns) < eval_episodes:
        print(step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        print("action:", actions)
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs
        step += 1

    return episodic_returns

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, obs_mode="state", render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"{run_name}", name_prefix="20M-front",)
        else:
            env = gym.make(env_id, obs_mode="state", render_mode="human")
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

if __name__ == "__main__":
    from dqn_train import QNetwork
    
    evaluate(
        "runs/dqn_test/dqn_train_999999.cleanrl_model",
        make_env,
        "PandaEnv-v1",
        eval_episodes=1,
        run_name=f"eval/test",
        Model=QNetwork,
        device="cpu",
        capture_video=False,
    )
