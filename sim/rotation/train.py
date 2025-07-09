import os
import time
import datetime
import yaml
import argparse

import numpy as np
import torch
import gymnasium as gym

from env_mujoco import SimplePathFollowingEnv
from td3 import TD3, ReplayBuffer

from torch.utils.tensorboard import SummaryWriter

def random_arc_generalization_test(env, agent, episodes=100):
    successes = 0
    for _ in range(episodes):
        # sample a random arc
        r  = np.random.uniform(1.0, 2.0)
        θ0 = np.random.uniform(-np.pi/2, 0)
        θ1 = np.random.uniform(0, np.pi/2)
        env.test_full_arc = True
        env.arc_radius  = r
        env.arc_start   = θ0
        env.arc_end     = θ1
        env.segment_length = None
        state, _ = env.reset()
        done = False
        while True:
            action = agent.select_action(np.array(state)).squeeze(0)
            state, _, done, truncated, info = env.step(action)
            if done or truncated:
                if done and info["progress"] > 0.95 and info["deviation"] < env.goal_thresh:
                    successes += 1
                break
    return successes / episodes

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
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        ep_steps = 0
        
        while True:
            action = agent.select_action(np.array(state)).squeeze(0)            
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            ep_reward += reward
            ep_steps += 1
            
            if render:
                env.render()
            
            if done or truncated:
                # Check if the episode was successful
                # This assumes the environment provides success info or we can infer it
                # from the final state or reward
                if done and info["progress"] > 0.95 and info["deviation"] < env.goal_thresh:
                    successes += 1
                break
        
        total_reward += ep_reward
    
    return {
        "avg_reward": total_reward / num_episodes,
        "success_rate": successes / num_episodes
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a TD3 agent on the Environment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--render_test", action="store_true", help="Render environment during testing")
    parser.add_argument("--test_episodes", type=int, default=20, help="Number of episodes to test on")
    parser.add_argument("--test_freq", type=int, default=50, help="Test policy every n episodes")
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

    env = SimplePathFollowingEnv(**env_cfg)

    test_env_full_cfg = env_cfg.copy()
    test_env_full_cfg['gui'] = args.render_test
    test_env_full_cfg['max_steps'] = 1000
    test_env_full = SimplePathFollowingEnv(**test_env_full_cfg)

    test_env_short_cfg = env_cfg.copy()
    test_env_short_cfg['gui'] = args.render_test
    test_env_short_cfg['segment_length'] = None
    test_env_short = SimplePathFollowingEnv(**test_env_short_cfg)


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

    # --- Training Loop ---
    total_steps = 0
    best_full_success = 0.0

    for ep in range(episodes):
        # Reset training environment (draws a new short segment automatically)
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        ep_steps = 0

        # Run one episode
        while True:
            total_steps += 1
            ep_steps += 1

            # Action selection: random until warm-up, then policy + noise
            if total_steps < start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state).squeeze(0)
                noise = np.random.normal(0, exploration_noise, size=action_dim)
                action = np.clip(action + noise, env.action_space.low, env.action_space.high)

            # Step and store
            next_state, reward, done, truncated, _ = env.step(action)
            episode_over = done or truncated

            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            ep_reward += reward

            # TD3 update
            if replay_buffer.size > batch_size:
                agent.update(
                    replay_buffer,
                    n_iter,
                    batch_size,
                    gamma,
                    polyak,
                    policy_noise,
                    noise_clip,
                    policy_delay
                )
            
            if episode_over:
                break

        # Log training reward
        print(f"[Ep {ep:4d}] Train Reward: {ep_reward:.2f}")
        writer.add_scalar("Train/EpisodeReward", ep_reward, ep)
        writer.add_scalar("Train/Steps", ep_steps, ep)

        # Periodic evaluation on both short segment & full arc
        if ep % args.test_freq == 0:
            short_stats = test_policy(test_env_short, agent,
                                    num_episodes=args.test_episodes,
                                    render=args.render_test)
            full_stats  = test_policy(test_env_full, agent,
                                    num_episodes=args.test_episodes,
                                    render=args.render_test)

            print(f"  >> Short-seg Success: {short_stats['success_rate']:.2f}, "
                f"Full-arc Success: {full_stats['success_rate']:.2f}")
            writer.add_scalar("Eval/ShortSuccess", short_stats["success_rate"], total_steps)
            writer.add_scalar("Eval/FullSuccess",  full_stats["success_rate"],  total_steps)
            writer.add_scalar("Eval/ShortReward",  short_stats["avg_reward"],   total_steps)
            writer.add_scalar("Eval/FullReward",   full_stats["avg_reward"],    total_steps)

            # Save best model on full-arc success
            if full_stats["success_rate"] > best_full_success:
                best_full_success = full_stats["success_rate"]
                agent.save(models_dir, "best_model")
                print(f" New best full-arc success: {best_full_success:.2f} — model saved")

        # Periodic checkpointing
        if ep % training_cfg.get("save_freq", 2000) == 0:
            agent.save(models_dir, f"checkpoint_ep{ep}")

    # --- Final Evaluation ---
    final_short = test_policy(test_env_short, agent,
                            num_episodes=args.test_episodes * 2,
                            render=args.render_test)
    final_full  = test_policy(test_env_full,  agent,
                            num_episodes=args.test_episodes * 2,
                            render=args.render_test)
    print(f"\nFINAL SHORT-SEG: Reward={final_short['avg_reward']:.2f}, Success={final_short['success_rate']:.2f}")
    print(f"FINAL FULL-ARC:  Reward={final_full['avg_reward']:.2f}, Success={final_full['success_rate']:.2f}")

    # Save final model
    agent.save(models_dir, "final_model")

    # Large-scale generalization test
    gen_rate = random_arc_generalization_test(test_env_full, agent, episodes=100)
    print(f"Random-Arc Generalization (100 trials): Success Rate={gen_rate:.2f}")

    # Clean up
    env.close()
    test_env_short.close()
    test_env_full.close()
    writer.close()

if __name__ == "__main__":
    main()