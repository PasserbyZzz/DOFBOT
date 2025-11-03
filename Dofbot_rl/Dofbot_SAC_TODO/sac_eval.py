from typing import Callable

import gymnasium as gym
import torch
import torch.nn as nn
import dofbotGymReachEnv
import time
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
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

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()
            # print("obs:", obs)
            # print("actions:", actions)
            # actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        next_obs, _, _, _, infos = envs.step(actions)
        time.sleep(0.1)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from sac import Actor, SoftQNetwork

    model_path = "runs/dofbotReach4/final_ckpt.pt"
    
    evaluate(
        model_path,
        "DofbotReachEnv-v1",
        eval_episodes=10,
        run_name=f"eval",
        Model=(Actor, SoftQNetwork),
        device="cpu",
        capture_video=False,
    )
