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

from env import SimplePathFollowingEnv
from td3 import TD3, ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def test_policy(env, agent, num_episodes=20, render=False):
    total_reward = 0.0
    successes = 0
    total_steps_mdp = 0
    total_dev_sum = 0.0

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
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
                if info["terminal_event"] == "success":
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


def random_arc_generalization_test(env, agent, rng, episodes=100):
    successes = 0
    for _ in range(episodes):
        r = rng.uniform(1.0, 2.0)
        theta0 = rng.uniform(-np.pi / 2, 0)
        theta1 = rng.uniform(0, np.pi / 2)
        env.test_full_arc = True
        env.arc_radius = r
        env.arc_start = theta0
        env.arc_end = theta1
        env.segment_length = None
        state, _ = env.reset()
        while True:
            action = agent.select_action(np.array(state)).squeeze(0)
            state, _, done, truncated, info = env.step(action)
            if done or truncated:
                if info["terminal_event"] == "success":
                    successes += 1
                break
    return successes / episodes


def prefill_replay_buffer(env, agent, replay_buffer, num_episodes, exploration_noise,
                           action_dim, rng_exploration, seed):
    """
    Roll out the loaded policy (with exploration noise) to pre-fill the replay buffer.
    This gives the resumed training meaningful transitions from the start.
    """
    print(f"  Pre-filling replay buffer with {num_episodes} episodes using loaded policy...")
    env.reset(seed=seed)
    total_transitions = 0

    for ep in range(num_episodes):
        with contextlib.redirect_stdout(io.StringIO()):
            state, _ = env.reset()
        done = False

        while True:
            action = agent.select_action(np.array(state)).squeeze(0)
            noise = rng_exploration.normal(0, exploration_noise, size=action_dim)
            action = np.clip(action + noise, env.action_space.low, env.action_space.high)

            with contextlib.redirect_stdout(io.StringIO()):
                next_state, reward, done, truncated, info = env.step(action)

            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            total_transitions += 1

            if done or truncated:
                break

    print(f"  Pre-fill complete: {total_transitions} transitions added "
          f"(buffer size: {replay_buffer.size}).")


def main():
    parser = argparse.ArgumentParser(description="Resume TD3 training from a checkpoint")

    # --- Resume-specific arguments ---
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to the original run directory (e.g. runs/run-0-2024-01-01_12-00-00)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint name to load (e.g. checkpoint_ep500, best_model, final_model)")
    parser.add_argument("--resume_episode", type=int, required=True,
                        help="Episode number to resume FROM (e.g. 500 if loading checkpoint_ep500)")
    parser.add_argument("--prefill_episodes", type=int, default=200,
                        help="Number of episodes to pre-fill replay buffer with loaded policy")

    # --- Standard arguments (same as train.py) ---
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file. Defaults to config.yaml inside --run_dir")
    parser.add_argument("--render_test", action="store_true", help="Render during testing")
    parser.add_argument("--test_episodes", type=int, default=20, help="Episodes per eval")
    parser.add_argument("--test_freq", type=int, default=50, help="Eval every n episodes")
    parser.add_argument("--total_episodes", type=int, default=None,
                    help="Total episodes to run to. Overrides config value. "
                         "e.g. --total_episodes 4000 runs 3000 more from ep1000")

    args = parser.parse_args()

    # --- Load config: prefer the one saved in the run dir for consistency ---
    config_path = args.config if args.config else os.path.join(args.run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_cfg = config["env"]
    agent_cfg = config["agent"]
    training_cfg = config["training"]

    eval_stage1_threshold = float(training_cfg.get("stage1_threshold", 0.85))
    eval_stage1_patience = int(training_cfg.get("stage1_patience", 5))
    eval_stage1_interval = int(training_cfg.get("stage1_interval", args.test_freq))

    eval_stage2_avg_threshold = float(training_cfg.get("stage2_avg_threshold", 0.90))
    eval_stage2_window = int(training_cfg.get("stage2_avg_window", 10))
    eval_stage2_interval = int(training_cfg.get("stage2_interval", 5))

    curriculum_cfg = training_cfg.get("curriculum", None)

    # --- Reproducibility ---
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

    # --- Append to the original run folder ---
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        raise NotADirectoryError(f"Run directory not found: {run_dir}")

    models_dir = os.path.join(run_dir, "models")
    logs_dir = os.path.join(run_dir, "logs")
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Save resume args alongside original config
    resume_args_path = os.path.join(run_dir, f"resume_args_{args.checkpoint}.yaml")
    with open(resume_args_path, "w") as f:
        yaml.dump(vars(args), f)

    # TensorBoard appends to the same logs dir — steps are continuous so graphs connect
    writer = SummaryWriter(logs_dir)

    # --- Environments ---
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

    for i, e in enumerate([env, test_env_full, test_env_short]):
        e.action_space.seed(seed + 100 + i)
        e.observation_space.seed(seed + 200 + i)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # --- Agent ---
    agent = TD3(
        lr=agent_cfg.get("lr", 1e-3),
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        max_torque=env_cfg.get("max_torque", 50.0),
        torch_rng=torch_rng
    )

    # --- Load checkpoint ---
    checkpoint_path = os.path.join(models_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path + "_actor.pth"):
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}_actor.pth\n"
            f"Make sure --checkpoint matches a saved model name (without extension)."
        )
    agent.load(models_dir, args.checkpoint)
    print(f"Loaded checkpoint: {checkpoint_path}")

    # --- Replay buffer ---
    replay_buffer = ReplayBuffer(
        max_size=agent_cfg.get("replay_buffer_max_size", 5e5),
        rng=rng_replay
    )

    # Pre-fill replay buffer by rolling out the loaded policy
    prefill_env_cfg = env_cfg.copy()
    prefill_env_cfg['gui'] = False
    prefill_env = SimplePathFollowingEnv(**prefill_env_cfg)
    prefill_env.segment_length = env.segment_length
    prefill_env.action_space.seed(seed + 999)
    prefill_env.observation_space.seed(seed + 998)

    prefill_replay_buffer(
        env=prefill_env,
        agent=agent,
        replay_buffer=replay_buffer,
        num_episodes=args.prefill_episodes,
        exploration_noise=agent_cfg.get("exploration_noise", 0.1),
        action_dim=action_dim,
        rng_exploration=rng_exploration,
        seed=seed + 500,
    )
    prefill_env.close()

    # --- Training parameters ---
    # episodes = training_cfg.get("episodes", 1000)
    episodes = args.total_episodes if args.total_episodes is not None else training_cfg.get("episodes", 1000)
    batch_size = agent_cfg.get("batch_size", 100)
    gamma = agent_cfg.get("gamma", 0.99)
    polyak = agent_cfg.get("polyak", 0.995)
    policy_noise = agent_cfg.get("policy_noise", 0.2)
    noise_clip = agent_cfg.get("noise_clip", 0.5)
    policy_delay = agent_cfg.get("policy_delay", 2)
    n_iter = agent_cfg.get("n_iter", 1)
    exploration_noise = agent_cfg.get("exploration_noise", 0.1)

    # Resume from the given episode; total_steps estimated from resume_episode
    start_ep = args.resume_episode
    # Estimate total_steps so TensorBoard x-axis is continuous.
    # We don't know exact step count so we approximate: use average of 100 steps/ep
    # as a safe lower-bound. If your runs logged steps you can replace this.
    avg_steps_per_ep = training_cfg.get("avg_steps_per_ep_estimate", 100)
    total_steps = start_ep * avg_steps_per_ep

    best_full_success = 0.0
    eval_phase = 1
    eval_interval = eval_stage1_interval
    next_eval_ep = start_ep  # evaluate immediately on resume so we have a baseline

    stage1_streak = 0
    full_succ_hist = deque(maxlen=eval_stage2_window)

    start_time = time.perf_counter()
    last_mark_time = start_time
    milestone = 10_000
    next_mark = total_steps + milestone

    print(f"\n=== Resuming from episode {start_ep} / {episodes} ===")
    print(f"    Checkpoint : {args.checkpoint}")
    print(f"    Run dir    : {run_dir}")
    print(f"    Buffer size: {replay_buffer.size}")
    print(f"    Device     : {dev}\n")

    for ep in range(start_ep, episodes):
        # Curriculum override
        if curriculum_cfg is not None:
            for tier in curriculum_cfg:
                if ep <= tier["until"]:
                    env.segment_length = tier["segment"] if tier["segment"] is not None else None
                    break

        with contextlib.redirect_stdout(io.StringIO()):
            state, _ = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        while True:
            total_steps += 1
            ep_steps += 1

            # No random warm-up on resume — buffer is already pre-filled
            action = agent.select_action(state).squeeze(0)
            noise = rng_exploration.normal(0, exploration_noise, size=action_dim)
            action = np.clip(action + noise, env.action_space.low, env.action_space.high)

            with contextlib.redirect_stdout(io.StringIO()):
                next_state, reward, done, truncated, info = env.step(action)

            # Time-based logging
            if total_steps >= next_mark:
                now = time.perf_counter()
                elapsed_total = now - start_time
                elapsed_window = now - last_mark_time
                writer.add_scalar("Time/ElapsedSec", elapsed_total, total_steps)
                writer.add_scalar("Time/StepsPerSecWindow",
                                  milestone / max(elapsed_window, 1e-9), total_steps)
                writer.add_scalar("Time/StepsPerSecCumulative",
                                  total_steps / max(elapsed_total, 1e-9), total_steps)
                writer.add_scalar("Time/UnixTime", time.time(), total_steps)
                print(f"[Time] {total_steps} steps — {elapsed_total:.1f}s total, "
                      f"{elapsed_window:.1f}s last {milestone}, "
                      f"{milestone / elapsed_window:.1f} steps/s window")
                last_mark_time = now
                next_mark += milestone

            episode_over = done or truncated
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            ep_reward += reward

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

        writer.add_scalar("Train/EpisodeReward", ep_reward, total_steps)
        print(f"[Ep {ep:4d}] Train Reward: {ep_reward:.2f}")

        # Periodic evaluation
        if ep >= next_eval_ep:
            short_stats = test_policy(test_env_short, agent,
                                      num_episodes=args.test_episodes,
                                      render=args.render_test)
            full_stats = test_policy(test_env_full, agent,
                                     num_episodes=args.test_episodes,
                                     render=args.render_test)

            print(f"  >> Short-seg Success: {short_stats['success_rate']:.2f}, "
                  f"Full-arc Success: {full_stats['success_rate']:.2f}")

            writer.add_scalar("Eval/ShortSuccess", short_stats["success_rate"], total_steps)
            writer.add_scalar("Eval/FullSuccess", full_stats["success_rate"], total_steps)
            writer.add_scalar("Eval/ShortReward", short_stats["avg_reward"], total_steps)
            writer.add_scalar("Eval/FullReward", full_stats["avg_reward"], total_steps)
            writer.add_scalar("Eval/ShortAvgSteps", short_stats["avg_steps"], total_steps)
            writer.add_scalar("Eval/FullAvgSteps", full_stats["avg_steps"], total_steps)
            writer.add_scalar("Eval/ShortDevSum", short_stats["avg_deviation_sum"], total_steps)
            writer.add_scalar("Eval/FullDevSum", full_stats["avg_deviation_sum"], total_steps)

            if full_stats["success_rate"] > best_full_success:
                best_full_success = full_stats["success_rate"]
                agent.save(models_dir, "best_model")
                print(f" New best full-arc success: {best_full_success:.2f} — model saved")

            if eval_phase == 1:
                meets = full_stats["success_rate"] >= eval_stage1_threshold
                stage1_streak = stage1_streak + 1 if meets else 0

                print(f" [Stage1] streak={stage1_streak}/{eval_stage1_patience} "
                      f"(threshold={eval_stage1_threshold:.2f})")

                if stage1_streak >= eval_stage1_patience:
                    eval_phase = 2
                    eval_interval = eval_stage2_interval
                    next_eval_ep = ep + eval_interval
                    stage1_streak = 0
                    full_succ_hist.clear()
                    print(f" >>> Switched to Stage 2 eval: "
                          f"every {eval_interval} episodes; "
                          f"avg window={eval_stage2_window}; "
                          f"avg threshold={eval_stage2_avg_threshold:.2f}")
                else:
                    next_eval_ep = ep + eval_interval

            elif eval_phase == 2:
                full_succ_hist.append(full_stats["success_rate"])
                avg_full = sum(full_succ_hist) / len(full_succ_hist)

                print(f" [Stage2] window={len(full_succ_hist)}/{eval_stage2_window} "
                      f"avg_full={avg_full:.2f} "
                      f"(threshold={eval_stage2_avg_threshold:.2f})")

                writer.add_scalar("Eval/FullSuccessMA", avg_full, total_steps)

                if (len(full_succ_hist) == eval_stage2_window and
                        avg_full >= eval_stage2_avg_threshold):
                    agent.save(models_dir, "converged_model")
                    print(f" Early-stop: rolling {eval_stage2_window}-eval average "
                          f">= {eval_stage2_avg_threshold:.2f} for Full arc. Stopping training.")
                    break
                else:
                    next_eval_ep = ep + eval_interval

        if ep % training_cfg.get("save_freq", 2000) == 0 or ep == episodes - 1:
            agent.save(models_dir, f"checkpoint_ep{ep}")

    # --- Final Evaluation ---
    final_short = test_policy(test_env_short, agent,
                              num_episodes=args.test_episodes * 2,
                              render=args.render_test)
    final_full = test_policy(test_env_full, agent,
                             num_episodes=args.test_episodes * 2,
                             render=args.render_test)

    print(f"\nFINAL SHORT-SEG: Reward={final_short['avg_reward']:.2f}, "
          f"Success={final_short['success_rate']:.2f}")
    print(f"FINAL FULL-ARC:  Reward={final_full['avg_reward']:.2f}, "
          f"Success={final_full['success_rate']:.2f}")

    agent.save(models_dir, "final_model_resumed")

    gen_rate = random_arc_generalization_test(
        test_env_full, agent, rng=rng_generalization_test, episodes=100
    )
    print(f"Random-Arc Generalization (100 trials): Success Rate={gen_rate:.2f}")

    env.close()
    test_env_short.close()
    test_env_full.close()
    writer.close()


if __name__ == "__main__":
    main()