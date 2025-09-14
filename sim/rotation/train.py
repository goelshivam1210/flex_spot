import contextlib
import io
import os
import time
import datetime
import yaml
import argparse
import random
from collections import deque

import numpy as np
import torch
import gymnasium as gym

from env import SimplePathFollowingEnv
from td3 import TD3, ReplayBuffer
from test_dual_force import DualForceTestEnv

from torch.utils.tensorboard import SummaryWriter

def random_arc_generalization_test(env, agent, rng, episodes=100):
    successes = 0
    for _ in range(episodes):
        # sample a random arc
        r  = rng.uniform(1.0, 2.0)
        θ0 = rng.uniform(-np.pi/2, 0)
        θ1 = rng.uniform(0, np.pi/2)
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
    total_reward = 0.0
    successes = 0
    total_steps_mdp = 0
    total_dev_sum = 0.0
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        ep_steps = 0
        ep_dev_sum = 0.0
        
        while True:
            action = agent.select_action(np.array(state)).squeeze(0)            
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            ep_reward += reward
            ep_steps += 1
            ep_dev_sum += float(info["deviation"])
            
            if render:
                env.render()
            
            if done or truncated:
                if done and info["progress"] > 0.95 and info["deviation"] < env.goal_thresh:
                    successes += 1
                break
        
        total_reward += ep_reward
        total_steps_mdp += ep_steps
        total_dev_sum += ep_dev_sum
    
    return {
        "avg_reward": total_reward / num_episodes,
        "success_rate": successes / num_episodes,
        "avg_steps": total_steps_mdp / num_episodes,
        "avg_deviation_sum": total_dev_sum / num_episodes,
    }

def test_policy_dual_force(env, agent, num_episodes=20, render=False):
    total_reward = 0.0
    successes = 0
    total_steps_mdp = 0
    total_dev_sum = 0.0

    with contextlib.redirect_stdout(io.StringIO()):
        env.set_force_mode("contact")

    for ep in range(num_episodes):
        with contextlib.redirect_stdout(io.StringIO()):
            state, _ = env.reset()

        ep_reward = 0.0
        ep_steps = 0
        ep_dev_sum = 0.0

        while True:
            action = agent.select_action(np.array(state)).squeeze(0)
            with contextlib.redirect_stdout(io.StringIO()):
                next_state, reward, done, truncated, info = env.step(action)

            if isinstance(reward, (tuple, list, np.ndarray)):
                reward = float(np.asarray(reward).flat[0])

            state = next_state
            ep_reward += float(reward)
            ep_steps += 1
            ep_dev_sum += float(info["deviation"])

            if render:
                env.render()
            if done or truncated:
                if done and info["progress"] > 0.95 and info["deviation"] < env.goal_thresh:
                    successes += 1
                break
        total_reward += ep_reward
        total_steps_mdp += ep_steps
        total_dev_sum += ep_dev_sum

    return {
        "avg_reward": total_reward / num_episodes,
        "success_rate": successes / num_episodes,
        "avg_steps": total_steps_mdp / num_episodes,
        "avg_deviation_sum": total_dev_sum / num_episodes,
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

    eval_stage1_threshold = float(training_cfg.get("stage1_threshold", 0.85))
    eval_stage1_patience  = int(training_cfg.get("stage1_patience", 5))
    eval_stage1_interval  = int(training_cfg.get("stage1_interval", args.test_freq))

    eval_stage2_avg_threshold = float(training_cfg.get("stage2_avg_threshold", 0.90))
    eval_stage2_window = int(training_cfg.get("stage2_avg_window", 10))
    eval_stage2_interval = int(training_cfg.get("stage2_interval", 5))

    curriculum_cfg = training_cfg.get("curriculum", None)

    # Set random seeds for reproducibility
    seed = args.seed
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    rng_replay = np.random.default_rng(seed + 10)
    rng_exploration = np.random.default_rng(seed + 20)
    rng_generalization_test = np.random.default_rng(seed + 30)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_rng = torch.Generator(device=dev).manual_seed(seed + 40)

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

    start_time = time.perf_counter()
    last_mark_time = start_time
    milestone = 10_000
    next_mark = milestone

    env = SimplePathFollowingEnv(**env_cfg)

    train_mode = training_cfg.get("train_segment_mode", "short").lower()
    if train_mode == "short":
        env.segment_length = env_cfg.get("short_segment_length", 0.3)
    elif train_mode == "full":
        env.segment_length = None
    else:
        raise ValueError(f"Unknown train_segment_mode: {train_mode}")

    env.reset(seed=seed)

    test_env_full_cfg = env_cfg.copy()
    test_env_full_cfg['gui'] = args.render_test
    test_env_full_cfg['max_steps'] = 1000
    test_env_full_cfg['segment_length'] = None
    test_env_full = SimplePathFollowingEnv(**test_env_full_cfg)
    test_env_full.reset(seed=seed + 1)

    test_env_short_cfg = env_cfg.copy()
    test_env_short_cfg['gui'] = args.render_test
    test_env_short_cfg['segment_length'] = env_cfg.get('short_segment_length', 0.3)
    test_env_short = SimplePathFollowingEnv(**test_env_short_cfg)
    test_env_short.reset(seed=seed + 2)

    test_env_dual_cfg = env_cfg.copy()
    test_env_dual_cfg['gui'] = args.render_test
    test_env_dual_cfg['max_steps'] = 1000
    test_env_dual_cfg['segment_length'] = None
    test_env_dual_full = DualForceTestEnv(**test_env_dual_cfg)
    test_env_dual_full.reset(seed=seed + 3)

    for i, e in enumerate([env, test_env_full, test_env_short, test_env_dual_full]):
        e.action_space.seed(seed + 100 + i)
        e.observation_space.seed(seed + 200 + i)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # Initialize the TD3 agent and replay buffer
    agent = TD3(lr=agent_cfg.get("lr", 1e-3),
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                max_torque = env_cfg.get("max_torque", 50.0),
                torch_rng=torch_rng)
    replay_buffer = ReplayBuffer(max_size=agent_cfg.get("replay_buffer_max_size", 5e5), rng=rng_replay)

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
    eval_phase = 1
    eval_interval = eval_stage1_interval
    next_eval_ep = 0

    stage1_streak   = 0
    full_succ_hist  = deque(maxlen=eval_stage2_window)
    dual_succ_hist  = deque(maxlen=eval_stage2_window)

    for ep in range(episodes):
        if curriculum_cfg is not None:
            for tier in curriculum_cfg:
                if ep <= tier["until"]:
                    env.segment_length = tier["segment"] if tier["segment"] is not None else None
                    break

        # Reset training environment (draws a new short segment automatically)
        with contextlib.redirect_stdout(io.StringIO()):
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
                noise = rng_exploration.normal(0, exploration_noise, size=action_dim)
                action = np.clip(action + noise, env.action_space.low, env.action_space.high)

            # Step and store
            with contextlib.redirect_stdout(io.StringIO()):
                next_state, reward, done, truncated, info = env.step(action)
            # writer.add_scalar("Train/StepReward", reward, total_steps)

            if total_steps >= next_mark:
                now = time.perf_counter()
                elapsed_total = now - start_time
                elapsed_window = now - last_mark_time
                writer.add_scalar("Time/ElapsedSec", elapsed_total, total_steps)
                writer.add_scalar("Time/StepsPerSecWindow", milestone / max(elapsed_window, 1e-9), total_steps)
                writer.add_scalar("Time/StepsPerSecCumulative", total_steps / max(elapsed_total, 1e-9), total_steps)
                writer.add_scalar("Time/UnixTime", time.time(), total_steps)

                print(f"[Time] {total_steps} steps — {elapsed_total:.1f}s total, "
                    f"{elapsed_window:.1f}s last {milestone}, "
                    f"{milestone/elapsed_window:.1f} steps/s window")

                last_mark_time = now
                next_mark += milestone

            episode_over = done or truncated

            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            ep_reward += reward

            lateral_err = abs(state[0])
            longitudinal_err = abs(state[1])
            orientation_err = abs(state[2])
            progress =  state[3]
            deviation =  state[4]
            speed_along_path =  state[5]

            # writer.add_scalar("Train/LateralError", lateral_err, total_steps)
            # writer.add_scalar("Train/LongitudinalError", longitudinal_err, total_steps)
            # writer.add_scalar("Train/OrientationError", orientation_err, total_steps)
            # writer.add_scalar("Train/Progress", progress, total_steps)
            # writer.add_scalar("Train/Deviation", deviation, total_steps)
            # writer.add_scalar("Train/SpeedAlongPath", speed_along_path, total_steps)

            # action magnitudes
            force_mag = np.linalg.norm(action[:2]) * env.max_force
            torque_v = action[2] * env.max_torque
            # writer.add_scalar("Action/ForceMagnitude", force_mag, total_steps)
            # writer.add_scalar("Action/Torque", torque_v, total_steps)

            # for name, val in info["reward_comps"].items():
            #     writer.add_scalar(f"Train/Reward/{name}", val, total_steps)

            # writer.add_scalar(
            #     "Train/TerminalAdjustment",
            #     info["terminal_adjustment"],
            #     total_steps
            # )
            # writer.add_scalar(
            #     "Train/GoalSuccess",
            #     1 if info["terminal_event"] == "success" else 0,
            #     total_steps
            # )
            # writer.add_scalar(
            #     "Train/OrientFail",
            #     1 if info["terminal_event"] == "orient_fail" else 0,
            #     total_steps
            # )

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

        # writer.add_scalar("Train/EpisodeReward", ep_reward, ep)
        # writer.add_scalar("Train/EpisodeDeviation", deviation, ep)
        # writer.add_scalar("Train/EpisodeOrientation", orientation_err, ep)
        # writer.add_scalar("Train/Steps", ep_steps, ep)

        # Periodic evaluation on both short segment & full arc
        if ep >= next_eval_ep:
            short_stats = test_policy(test_env_short, agent,
                                    num_episodes=args.test_episodes,
                                    render=args.render_test)
            full_stats  = test_policy(test_env_full, agent,
                                    num_episodes=args.test_episodes,
                                    render=args.render_test)
            
            dual_stats  = test_policy_dual_force(test_env_dual_full, agent,
                                        num_episodes=args.test_episodes,
                                        render=args.render_test)


            print(f"  >> Short-seg Success: {short_stats['success_rate']:.2f}, "
                f"Full-arc Success: {full_stats['success_rate']:.2f}, "
                f"Dual-arc Success: {dual_stats['success_rate']:.2f}")

            writer.add_scalar("Eval/ShortSuccess", short_stats["success_rate"], total_steps)
            writer.add_scalar("Eval/FullSuccess",  full_stats["success_rate"],  total_steps)
            writer.add_scalar("Eval/DualSuccess",  dual_stats["success_rate"],  total_steps)

            writer.add_scalar("Eval/ShortReward",  short_stats["avg_reward"],   total_steps)
            writer.add_scalar("Eval/FullReward",   full_stats["avg_reward"],    total_steps)
            writer.add_scalar("Eval/DualReward",   dual_stats["avg_reward"],    total_steps)

            writer.add_scalar("Eval/ShortAvgSteps", short_stats["avg_steps"], total_steps)
            writer.add_scalar("Eval/FullAvgSteps",  full_stats["avg_steps"],  total_steps)
            writer.add_scalar("Eval/DualAvgSteps",  dual_stats["avg_steps"],  total_steps)

            writer.add_scalar("Eval/ShortDevSum",   short_stats["avg_deviation_sum"], total_steps)
            writer.add_scalar("Eval/FullDevSum",    full_stats["avg_deviation_sum"],  total_steps)
            writer.add_scalar("Eval/DualDevSum",    dual_stats["avg_deviation_sum"],  total_steps)

            # Save best model on full-arc success
            if full_stats["success_rate"] > best_full_success:
                best_full_success = full_stats["success_rate"]
                agent.save(models_dir, "best_model")
                print(f" New best full-arc success: {best_full_success:.2f} — model saved")
            
            if eval_phase == 1:
                meets = (full_stats["success_rate"] >= eval_stage1_threshold and
                        dual_stats["success_rate"] >= eval_stage1_threshold)
                stage1_streak = stage1_streak + 1 if meets else 0

                print(f" [Stage1] streak={stage1_streak}/{eval_stage1_patience} (threshold={eval_stage1_threshold:.2f})")
                if stage1_streak >= eval_stage1_patience:
                    eval_phase = 2
                    eval_interval = eval_stage2_interval
                    next_eval_ep = ep + eval_interval
                    stage1_streak = 0
                    full_succ_hist.clear()
                    dual_succ_hist.clear()
                    print(f" >>> Switched to Stage 2 eval: "
                        f"every {eval_interval} episodes; "
                        f"avg window={eval_stage2_window}; "
                        f"avg threshold={eval_stage2_avg_threshold:.2f}")
                else:
                    next_eval_ep = ep + eval_interval

            elif eval_phase == 2:
                full_succ_hist.append(full_stats["success_rate"])
                dual_succ_hist.append(dual_stats["success_rate"])

                avg_full = sum(full_succ_hist) / len(full_succ_hist)
                avg_dual = sum(dual_succ_hist) / len(dual_succ_hist)

                print(f" [Stage2] window={len(full_succ_hist)}/{eval_stage2_window} "
                    f"avg_full={avg_full:.2f}, avg_dual={avg_dual:.2f} "
                    f"(threshold={eval_stage2_avg_threshold:.2f})")

                writer.add_scalar("Eval/FullSuccessMA", avg_full, total_steps)
                writer.add_scalar("Eval/DualSuccessMA", avg_dual, total_steps)

                if (len(full_succ_hist) == eval_stage2_window and
                    len(dual_succ_hist) == eval_stage2_window and
                    avg_full >= eval_stage2_avg_threshold and
                    avg_dual >= eval_stage2_avg_threshold):
                    agent.save(models_dir, "converged_model")
                    print(f" Early-stop: rolling {eval_stage2_window}-eval averages "
                        f"≥ {eval_stage2_avg_threshold:.2f} for both Full & Dual. Stopping training.")
                    break
                else:
                    next_eval_ep = ep + eval_interval

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
    final_dual = test_policy_dual_force(test_env_dual_full, agent,
                                    num_episodes=args.test_episodes * 2,
                                    render=args.render_test)

    print(f"\nFINAL SHORT-SEG: Reward={final_short['avg_reward']:.2f}, Success={final_short['success_rate']:.2f}")
    print(f"FINAL FULL-ARC:  Reward={final_full['avg_reward']:.2f}, Success={final_full['success_rate']:.2f}")
    print(f"FINAL DUAL-FORCE (Full Arc): Reward={final_dual['avg_reward']:.2f}, Success={final_dual['success_rate']:.2f}")


    # Save final model
    agent.save(models_dir, "final_model")

    # Large-scale generalization test
    gen_rate = random_arc_generalization_test(test_env_full, agent, rng=rng_generalization_test, episodes=100)
    print(f"Random-Arc Generalization (100 trials): Success Rate={gen_rate:.2f}")

    # Clean up
    env.close()
    test_env_short.close()
    test_env_full.close()
    test_env_dual_full.close()
    writer.close()

if __name__ == "__main__":
    main()