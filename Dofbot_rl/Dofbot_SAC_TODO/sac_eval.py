import os
import datetime
import pybullet as p
from typing import Callable

import gymnasium as gym
import torch
import torch.nn as nn
import dofbot_env
import time
import numpy as np

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            print(f"Recording video to {run_name}")
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"results/record", name_prefix=run_name)
        else:
            env = gym.make(env_id,  render_mode="human")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

def evaluate(
    model_path: str,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    exploration_noise: float = 0.1,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    actor = Model[0](envs).to(device)
    model_params = torch.load(model_path, map_location=device)
    actor_params = model_params['actor']
    actor.load_state_dict(actor_params)
    actor.eval()
    # note: qf is not used in this script

    save_dir = "results/record"
    os.makedirs(save_dir, exist_ok=True)
    mp4_path = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4")
    
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, mp4_path)
    print(f"Started recording video to {mp4_path}")

    obs, _ = envs.reset()
    episodic_returns = []
    episodic_termination = []
    success = 0
    
    while len(episodic_termination) < eval_episodes:
        with torch.no_grad():
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, _, terminated, _, infos = envs.step(actions)
        time.sleep(0.1)
        
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_termination)}, episodic_return={info['episode']['r']}, terminated={terminated[0]}")
                episodic_returns.append(info["episode"]["r"])
                episodic_termination.append(terminated[0])
                
                if terminated[0]:
                    success += 1
                    
        obs = next_obs

    if log_id >= 0:
        p.stopStateLogging(log_id)
        print(f"Stopped recording video.")

    success_rate = success / len(episodic_termination)
    print("Success Rate:", success_rate)
    print("Average Return:", np.mean(episodic_returns))
    
    return success_rate, episodic_returns


if __name__ == "__main__":
    from sac_train import Actor, SoftQNetwork

    model_path = "runs/sac_demo_1/1200000_ckpt.pt"
    
    evaluate(
        model_path,
        "DofbotReachEnv-v1",
        eval_episodes=1,
        run_name=f"sac_eval",
        Model=(Actor, SoftQNetwork),
        device="cpu",
        capture_video=False,
    )
