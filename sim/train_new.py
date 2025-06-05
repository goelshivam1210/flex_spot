# sim/train_new.py
import os
import time
import datetime
import yaml
import argparse
import math

import numpy as np
import torch
import gym

from env import PrismaticEnv
from td3 import TD3, ReplayBuffer

from torch.utils.tensorboard import SummaryWriter

'''
Usage example:
python train_new.py --seed 1 --config config.yaml

'''

#
# Runs policy and returns averaged metrics
#
def test_policy(env, agent, num_episodes=20, render=False):
    total_reward = 0
    successes = 0
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        terminated = False
        truncated = False
        ep_steps = 0
        
        # Loop while not (terminated or truncated) and within env's max_steps
        while not (terminated or truncated) and ep_steps < env.max_steps :
            action = agent.select_action(np.array(state)) # TD3 select_action returns flattened already
            
            next_state, reward, terminated, truncated, info = env.step(action) # Env returns 5 values
            state = next_state
            ep_reward += reward
            ep_steps += 1
                    
        # Success condition (state[2:4] is displacement from initial_pos_xy)
        # If initial_pos_xy is [0,0], then state[2:4] is current world_pos_xy
        # This check is correct if env.initial_box_pos_xy is effectively [0,0] for displacement calc.
        current_world_pos_from_state_disp = state[2:4] + env.initial_box_pos_xy # Reconstruct world_pos
        if np.linalg.norm(current_world_pos_from_state_disp - env.goal_pos_world) < env.goal_thresh:
            successes += 1
        
        total_reward += ep_reward
    
    avg_reward = total_reward / num_episodes if num_episodes > 0 else 0
    success_rate = successes / num_episodes if num_episodes > 0 else 0
    return {"avg_reward": avg_reward, "success_rate": success_rate}

def main():
    parser = argparse.ArgumentParser(description="Train a TD3 agent on the Prismatic Environment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    # render_test is for test_env's gui flag
    parser.add_argument("--render_test", action="store_true", help="Render test_env during testing if GUI available") 
    parser.add_argument("--test_episodes", type=int, default=20, help="Number of episodes to test on")
    parser.add_argument("--test_freq", type=int, default=1000, help="Test policy every n episodes")
    args = parser.parse_args()
    
    with open(args.config, "r") as f: config = yaml.safe_load(f)
    env_cfg = config["env"]
    agent_cfg = config["agent"]
    training_cfg = config["training"]

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # env.action_space.seed(seed) # For gym 0.26+

    # Directory setup
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run-seed{seed}-{timestamp}" 
    runs_dir = training_cfg.get("runs_dir", "runs")
    run_dir = os.path.join(runs_dir, run_name)
    models_dir = os.path.join(run_dir, "models")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f: yaml.dump(config, f)
    with open(os.path.join(run_dir, "args.yaml"), "w") as f: yaml.dump(vars(args), f)
    writer = SummaryWriter(logs_dir)

    # Pass all relevant env_cfg parameters
    env_kwargs = {
        "gui": env_cfg.get("gui", False),
        "max_force_per_pusher": env_cfg.get("max_force_per_pusher", 50.0),
        "friction": env_cfg.get("friction", 0.5),
        "angular_damping_factor": env_cfg.get("angular_damping_factor", 0.2),
        "goal_pos": np.array(env_cfg.get("goal_pos", [2.0, 0.0])),
        "goal_thresh": env_cfg.get("goal_thresh", 0.1),
        "max_steps": env_cfg.get("max_steps", 500),
        "pusher_offset_distance": env_cfg.get("pusher_offset_distance", 0.15),
        "randomize_initial_yaw": env_cfg.get("randomize_initial_yaw", True),
        "initial_yaw_range": np.array(env_cfg.get("initial_yaw_range", [-math.pi, math.pi]))
    }

    env = PrismaticEnv(**env_kwargs)
    test_env_kwargs = env_kwargs.copy()
    test_env_kwargs["gui"] = args.render_test
    test_env = PrismaticEnv(**test_env_kwargs)
    
    # Seed environment for gym 0.26+ (optional if gym is older)
    # env.reset(seed=seed)
    # test_env.reset(seed=seed+100) # Use a different seed for test_env if desired

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action_val = float(env.action_space.high[0]) # Assuming high is [1.0, 1.0]

    agent = TD3(lr=agent_cfg.get("lr", 1e-4),
                state_dim=state_dim, action_dim=action_dim, max_action=max_action_val)
    replay_buffer = ReplayBuffer(max_size=int(agent_cfg.get("replay_buffer_max_size", 1e6)))

    episodes = training_cfg.get("episodes", 50000)
    start_timesteps = training_cfg.get("start_timesteps", 10000)
    batch_size = agent_cfg.get("batch_size", 100)
    gamma = agent_cfg.get("gamma", 0.99)
    polyak = agent_cfg.get("polyak", 0.995)
    policy_noise = agent_cfg.get("policy_noise", 0.2)
    noise_clip = agent_cfg.get("noise_clip", 0.5)
    policy_delay = agent_cfg.get("policy_delay", 2)
    n_iter_grad = agent_cfg.get("n_iter", 1)
    exploration_noise_std = agent_cfg.get("exploration_noise", 0.1)


    total_steps = 0
    best_success_rate = -1.0 # Initialize to allow saving on first success

    # training loop
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        episode_terminated = False
        episode_truncated = False
        current_ep_steps = 0

        while not (episode_terminated or episode_truncated):
            total_steps += 1
            current_ep_steps +=1

            if total_steps < start_timesteps:
                action = env.action_space.sample()
            else:
                action_from_policy = agent.select_action(np.array(state))
                noise = np.random.normal(0, max_action_val * exploration_noise_std, size=action_dim)
                action = (action_from_policy + noise).clip(env.action_space.low, env.action_space.high)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # done_for_buffer for TD3 should be 1.0 if terminated (true end), 0.0 if truncated or ongoing
            done_for_buffer = float(terminated) 
            replay_buffer.add((state, action, reward, next_state, done_for_buffer))
            
            state = next_state
            ep_reward += reward

            if replay_buffer.current_size > batch_size:
                agent.update(replay_buffer, n_iter_grad, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
            
            episode_terminated = terminated
            episode_truncated = truncated
            # Safety break if env.max_steps isn't triggering truncation correctly
            if current_ep_steps >= env.max_steps and not (episode_terminated or episode_truncated):
                episode_truncated = True


        print(f"Seed {seed} | Ep {ep:04d} | Reward: {ep_reward:8.2f} | Steps: {current_ep_steps:3d} | TotalSteps: {total_steps}")
        writer.add_scalar("training/Reward_Episode", ep_reward, ep)
        writer.add_scalar("training/Episode_Length", current_ep_steps, ep)


        if ep % args.test_freq == 0 or ep == episodes -1 : # Test also on last episode
            test_stats = test_policy(test_env, agent, num_episodes=args.test_episodes, render=args.render_test)
            print(f"Testing @ Ep {ep:04d} >> AvgReward: {test_stats['avg_reward']:.2f}, SuccessRate: {test_stats['success_rate']:.2%}")
            writer.add_scalar("eval/AvgReward", test_stats['avg_reward'], ep)
            writer.add_scalar("eval/SuccessRate", test_stats['success_rate'], ep)
            
            if test_stats['success_rate'] > best_success_rate:
                best_success_rate = test_stats['success_rate']
                agent.save(models_dir, "best_model")
                print(f"** New best model saved (SuccessRate = {best_success_rate:.2%}) **")

        if ep % training_cfg.get("save_freq", 5000) == 0 or ep == episodes - 1:
            agent.save(models_dir, f"td3_ep{ep}")

    # Final test and evaluation
    final_test_stats = test_policy(test_env, agent, num_episodes=args.test_episodes * 2, render=args.render_test) # More episodes for final
    print(f"\nFinal Policy Evaluation (Seed {seed}):")
    print(f"  Average Reward: {final_test_stats['avg_reward']:.2f}")
    print(f"  Success Rate: {final_test_stats['success_rate']:.2%}")
    writer.add_scalar("final_eval/AvgReward", final_test_stats['avg_reward'], episodes) # Log at last episode step
    writer.add_scalar("final_eval/SuccessRate", final_test_stats['success_rate'], episodes)
    
    agent.save(models_dir, "final_model")
    print(f"Final model saved for seed {seed}.")

    env.close()
    test_env.close()
    writer.close()

if __name__ == "__main__":
    main()