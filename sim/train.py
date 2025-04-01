import os
import time
import datetime
import yaml
import numpy as np
import torch
import gym
from env import PrismaticEnv
from td3 import TD3, ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# Load configuration parameters from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract configuration sections
env_cfg = config["env"]
agent_cfg = config["agent"]
training_cfg = config["training"]

# Get list of seeds (default to [0] if not specified)
seeds = config.get("seeds", [2])  # Example: multiple seeds for parallel runs

# Root directory for all runs
runs_dir = "runs"
if not os.path.exists(runs_dir):
    os.makedirs(runs_dir)

# Loop over seeds
for seed in seeds:
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create a timestamped run directory for this seed
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

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(logs_dir)

    # Create the environment (single instance)
    env = PrismaticEnv(
        gui=env_cfg.get("gui", False),
        max_force=env_cfg.get("max_force", 20.0),
        friction=env_cfg.get("friction", 0.5),
        goal_pos=np.array(env_cfg.get("goal_pos", [2.0, 0.0])),
        goal_thresh=env_cfg.get("goal_thresh", 0.1),
        max_steps=env_cfg.get("max_steps", 200)
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = agent_cfg.get("max_action", 1.0)

    # Initialize the TD3 agent and replay buffer
    agent = TD3(lr=agent_cfg.get("lr", 1e-3),
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action)
    replay_buffer = ReplayBuffer(max_size=agent_cfg.get("replay_buffer_max_size", 5e5))

    # Training parameters
    episodes = training_cfg.get("episodes", 1000)
    start_timesteps = training_cfg.get("start_timesteps", 200)
    batch_size = agent_cfg.get("batch_size", 100)
    gamma = agent_cfg.get("gamma", 0.99)
    polyak = agent_cfg.get("polyak", 0.995)
    policy_noise = agent_cfg.get("policy_noise", 0.2)
    noise_clip = agent_cfg.get("noise_clip", 0.5)
    policy_delay = agent_cfg.get("policy_delay", 2)
    n_iter = agent_cfg.get("n_iter", 1)
    exploration_noise = agent_cfg.get("exploration_noise", 0.1)

    total_steps = 0

    for ep in range(episodes):
        state, _= env.reset()
        ep_reward = 0
        done = False

        while not done:
            total_steps += 1

            # Select action: random during initial exploration phase
            if total_steps < start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(np.array(state))
                # Ensure action is a 1D array of correct shape
                if action.ndim > 1:
                    action = action.squeeze(0)
                action = action.flatten()
                # Add exploration noise
                noise = np.random.normal(0, exploration_noise, size=action_dim)
                action = (action + noise).clip(env.action_space.low, env.action_space.high)

            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            ep_reward += reward

            # Update TD3 agent if replay buffer has enough samples
            if replay_buffer.size > batch_size:
                agent.update(replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)

        # Log episode reward
        print(f"Seed {seed} | Episode {ep}: Reward = {ep_reward}")
        writer.add_scalar("Reward/Episode", ep_reward, ep)

        # Optionally save the model periodically
        if ep % training_cfg.get("save_freq", 100) == 0:
            agent.save(models_dir, f"td3_ep{ep}")

    env.close()
    writer.close()