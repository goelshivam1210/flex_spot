import os
import time
import datetime
import yaml
import numpy as np
import torch
import gym
from flex_spot.sim.env_1 import PrismaticEnv
from td3 import TD3, ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# Load configuration parameters from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract configuration sections
env_cfg = config["env"]
agent_cfg = config["agent"]
training_cfg = config["training"]

# Get seeds list (if not provided, default to [0])
seeds = config.get("seeds", [0])

# Number of parallel environments
num_envs = training_cfg.get("num_envs", 8)
episodes = training_cfg.get("episodes", 1000)
start_timesteps = training_cfg.get("start_timesteps", 1000)
batch_size = agent_cfg.get("batch_size", 100)
gamma = agent_cfg.get("gamma", 0.99)
polyak = agent_cfg.get("polyak", 0.995)
policy_noise = agent_cfg.get("policy_noise", 0.2)
noise_clip = agent_cfg.get("noise_clip", 0.5)
policy_delay = agent_cfg.get("policy_delay", 2)
n_iter = agent_cfg.get("n_iter", 1)
exploration_noise = agent_cfg.get("exploration_noise", 0.1)

# Create root directory for runs if it doesn't exist
runs_dir = "Runs"
if not os.path.exists(runs_dir):
    os.makedirs(runs_dir)

for seed in seeds:
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create a timestamped run directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run-{seed}-{timestamp}"
    run_dir = os.path.join(runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    models_dir = os.path.join(run_dir, "models")
    logs_dir = os.path.join(run_dir, "logs")
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize TensorBoard SummaryWriter for logging
    writer = SummaryWriter(logs_dir)
    
    # Factory function to create new environment instances
    def make_env():
        def _init():
            env = PrismaticEnv(gui=False,
                               max_force=env_cfg.get("max_force", 10.0),
                               friction=env_cfg.get("friction", 0.5),
                               goal_pos=np.array(env_cfg.get("goal_pos", [2.0, 0.0])),
                               goal_thresh=env_cfg.get("goal_thresh", 0.1),
                               max_steps=env_cfg.get("max_steps", 200))
            try:
                env.seed(seed)
            except:
                pass
            return env
        return _init

    # Create a vectorized environment with multiple instances
    env_fns = [make_env() for _ in range(num_envs)]
    vec_env = gym.vector.AsyncVectorEnv(env_fns)
    
    state_dim = vec_env.single_observation_space.shape[0]
    action_dim = vec_env.single_action_space.shape[0]
    max_action = agent_cfg.get("max_action", 1.0)
    
    # Initialize TD3 agent and replay buffer
    agent = TD3(lr=agent_cfg.get("lr", 1e-3),
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action)
    replay_buffer = ReplayBuffer(max_size=agent_cfg.get("replay_buffer_max_size", 5e5))
    
    total_steps = 0
    episode_rewards = np.zeros(num_envs)
    episode_counts = np.zeros(num_envs, dtype=int)
    
    # Reset vectorized environment to get initial observations
    states = vec_env.reset()
    
    # Training loop over episodes
    for ep in range(episodes):
        done_flags = np.array([False] * num_envs)
        while not np.all(done_flags):
            total_steps += num_envs
            actions = []
            for s in states:
                if total_steps < start_timesteps:
                    action = vec_env.single_action_space.sample()
                else:
                    action = agent.select_action(np.array(s))
                    if action.ndim > 1:
                        action = action.squeeze()
                    action = action.flatten()
                    noise = np.random.normal(0, exploration_noise, size=action_dim)
                    action = (action + noise).clip(vec_env.single_action_space.low,
                                                   vec_env.single_action_space.high)
                actions.append(action)
            actions = np.array(actions)
            
            next_states, rewards, dones, infos = vec_env.step(actions)
            
            for idx in range(num_envs):
                replay_buffer.add((states[idx], actions[idx], rewards[idx], next_states[idx], float(dones[idx])))
                episode_rewards[idx] += rewards[idx]
                if dones[idx]:
                    writer.add_scalar("Reward/episode", episode_rewards[idx], total_steps)
                    print(f"Seed {seed}, Env {idx}, Episode finished with reward: {episode_rewards[idx]}")
                    episode_counts[idx] += 1
                    episode_rewards[idx] = 0.0
            
            states = next_states
            
            # Update agent if enough samples are collected
            if replay_buffer.size > batch_size:
                agent.update(replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
        
        # Optionally save the model every few episodes
        if ep % training_cfg.get("save_freq", 100) == 0:
            agent.save(models_dir, f"td3_ep{ep}")
    
    vec_env.close()
    writer.close()