import os
import time
import datetime
import yaml
import argparse

import numpy as np
import torch
import gym

from env import SimplePathFollowingEnv
from td3 import TD3, ReplayBuffer

from torch.utils.tensorboard import SummaryWriter

def test_policy(env, agent, num_episodes=20, render=False):
    """
    Test the policy for a specified number of episodes and return statistics.
    
    Args:
        env: The environment to test in
        agent: The agent/policy to test
        num_episodes: Number of episodes to run
        render: Whether to render the environment during testing
    
    Returns:
        dict: Statistics about the test run (avg_reward, success_rate)
    """
    total_reward = 0
    successes = 0
    total_steps = 0
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        ep_steps = 0
        
        while not done and ep_steps < env.max_steps:
            action = agent.select_action(np.array(state))
            if action.ndim > 1:
                action = action.squeeze(0)
            
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            ep_reward += reward
            ep_steps += 1
            
            if render:
                env.render()
        
        # Check if the episode was successful
        # This assumes the environment provides success info or we can infer it
        # from the final state or reward
        if done and info["progress"] > 0.95 and info["deviation"] < env.goal_thresh:
            successes += 1
        
        total_reward += ep_reward
    
    return {
        "avg_reward": total_reward / num_episodes,
        "success_rate": successes / num_episodes
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a TD3 agent on the Prismatic Environment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--render_test", action="store_true", help="Render environment during testing")
    parser.add_argument("--test_episodes", type=int, default=20, help="Number of episodes to test on")
    parser.add_argument("--test_freq", type=int, default=1000, help="Test policy every n episodes")
    args = parser.parse_args()
    
    # Load configuration parameters from config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration sections
    env_cfg = config["env"]
    agent_cfg = config["agent"]
    training_cfg = config["training"]

    # Set random seeds for reproducibility
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create a timestamped run directory for this seed
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run-{seed}-{timestamp}"
    runs_dir = "runs"
    run_dir = os.path.join(runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    models_dir = os.path.join(run_dir, "models")
    logs_dir = os.path.join(run_dir, "logs")
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Save the configuration and arguments
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    with open(os.path.join(run_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(logs_dir)

    # Create the environment for training
    env = SimplePathFollowingEnv(
        gui=env_cfg.get("gui", False),
        max_force=env_cfg.get("max_force", 100.0),
        max_torque=env_cfg.get("max_torque", 50.0),
        friction=env_cfg.get("friction", 0.5),
        goal_thresh=env_cfg.get("goal_thresh", 0.1),
        max_steps=env_cfg.get("max_steps", 200),
        seed = seed
    )
    
    # Create a separate environment for testing
    test_env = SimplePathFollowingEnv(
        gui=args.render_test,  # Only use GUI if we're rendering test episodes
        max_force=env_cfg.get("max_force", 100.0),
        max_torque=env_cfg.get("max_torque", 50.0),
        friction=env_cfg.get("friction", 0.5),
        goal_thresh=env_cfg.get("goal_thresh", 0.1),
        max_steps=env_cfg.get("max_steps", 200),
        seed = seed
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # Initialize the TD3 agent and replay buffer
    agent = TD3(lr=agent_cfg.get("lr", 1e-3),
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                max_torque = env_cfg.get("max_torque", 50.0))
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
    best_success_rate = 0

    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        ep_steps = 0

        while not done and ep_steps < env.max_steps:
            total_steps += 1
            ep_steps += 1

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
        print(f"Seed {seed} | Episode {ep}: Reward = {ep_reward:.2f}")
        writer.add_scalar("Reward/Episode", ep_reward, ep)

        # Test policy periodically
        if ep % args.test_freq == 0:
            test_stats = test_policy(test_env, agent, num_episodes=args.test_episodes, render=args.render_test)
            print(f"Testing policy - Avg. Reward: {test_stats['avg_reward']:.2f}, Success: {test_stats['success_rate']:.2f}")
            writer.add_scalar("Test/AvgReward", test_stats['avg_reward'], ep)
            writer.add_scalar("Test/SuccessRate", test_stats['success_rate'], ep)
            
            # Save the best model based on success rate
            if test_stats['success_rate'] > best_success_rate:
                best_success_rate = test_stats['success_rate']
                agent.save(models_dir, "best_model")
                print(f"New best model saved with success rate: {best_success_rate:.2f}")

        # Save the model periodically
        if ep % training_cfg.get("save_freq", 100) == 0:
            agent.save(models_dir, f"td3_ep{ep}")

    # Final test and evaluation
    final_test_stats = test_policy(test_env, agent, num_episodes=args.test_episodes*2, render=args.render_test)
    print(f"\nFinal Policy Evaluation:")
    print(f"Average Reward: {final_test_stats['avg_reward']:.2f}")
    print(f"Success Rate: {final_test_stats['success_rate']:.2f}")
    
    # Save final model
    agent.save(models_dir, "final_model")

    env.close()
    test_env.close()
    writer.close()

if __name__ == "__main__":
    main()