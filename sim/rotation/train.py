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


def run_eval(env_short, env_full, env_gen, agent, num_episodes, rng_gen):
    """
    Unified evaluation across all three test types.
    Mass and friction are randomized per episode via env.reset() domain randomization.
    Arc direction is flipped 50% of the time via env.reset() for all envs.
    Random-arc trials additionally randomize geometry and direction.
    """
    return {
        "short": _eval_env(env_short, agent, num_episodes),
        "full":  _eval_env(env_full,  agent, num_episodes),
        "gen":   _eval_random_arc(env_gen, agent, num_episodes, rng_gen),
    }


def _eval_env(env, agent, num_episodes):
    """Run fixed-geometry eval (short-seg or full-arc)."""
    env._reverse_path = False  # ensure no bleed from random arc eval
    total_reward  = 0.0
    successes     = 0
    total_steps   = 0
    total_deviation = 0.0
    terminal_counts = {"success": 0, "wandered_off": 0, "orient_fail": 0, "truncated": 0}

    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward    = 0.0
        ep_steps     = 0
        ep_deviation = 0.0

        while True:
            action = agent.select_action(np.array(state)).squeeze(0)
            state, reward, done, truncated, info = env.step(action)
            ep_reward    += reward
            ep_steps     += 1
            ep_deviation += float(info["deviation"])

            if done or truncated:
                event = info["terminal_event"] if info["terminal_event"] else "truncated"
                terminal_counts[event] = terminal_counts.get(event, 0) + 1
                if info["terminal_event"] == "success":
                    successes += 1
                break

        total_reward    += ep_reward
        total_steps     += ep_steps
        total_deviation += ep_deviation / max(ep_steps, 1)

    n = num_episodes
    return {
        "avg_reward":      total_reward / n,
        "success_rate":    successes / n,
        "avg_steps":       total_steps / n,
        "avg_deviation":   total_deviation / n,
        "terminal_counts": terminal_counts,
    }


def _eval_random_arc(env, agent, num_episodes, rng):
    """
    Random-arc generalization eval.
    Each episode samples a new arc geometry and randomly flips direction.
    """
    total_reward    = 0.0
    successes       = 0
    total_steps     = 0
    total_deviation = 0.0
    terminal_counts = {"success": 0, "wandered_off": 0, "orient_fail": 0, "truncated": 0}

    for _ in range(num_episodes):
        # Sample random arc geometry
        r      = rng.uniform(1.0, 2.0)
        theta0 = rng.uniform(-np.pi / 2, 0)
        theta1 = rng.uniform(0, np.pi / 2)

        env.test_full_arc  = True
        env.arc_radius     = r
        env.arc_start      = theta0
        env.arc_end        = theta1
        env.segment_length = None
        # Reverse traversal direction 50% of the time — same geometry, opposite direction
        env._reverse_path  = rng.random() > 0.5

        state, _ = env.reset()
        ep_reward    = 0.0
        ep_steps     = 0
        ep_deviation = 0.0

        while True:
            action = agent.select_action(np.array(state)).squeeze(0)
            state, reward, done, truncated, info = env.step(action)
            ep_reward    += reward
            ep_steps     += 1
            ep_deviation += float(info["deviation"])

            if done or truncated:
                event = info["terminal_event"] if info["terminal_event"] else "truncated"
                terminal_counts[event] = terminal_counts.get(event, 0) + 1
                if info["terminal_event"] == "success":
                    successes += 1
                break

        total_reward    += ep_reward
        total_steps     += ep_steps
        total_deviation += ep_deviation / max(ep_steps, 1)

    n = num_episodes
    return {
        "avg_reward":      total_reward / n,
        "success_rate":    successes / n,
        "avg_steps":       total_steps / n,
        "avg_deviation":   total_deviation / n,
        "terminal_counts": terminal_counts,
    }


def log_eval_to_tb(writer, results, total_steps, prefix="Eval"):
    """Write all eval metrics to TensorBoard under clean namespaces."""
    for name, stats in [("Short", results["short"]),
                        ("Full",  results["full"]),
                        ("Gen",   results["gen"])]:
        tag = f"{prefix}/{name}"
        writer.add_scalar(f"{tag}/SuccessRate",  stats["success_rate"],  total_steps)
        writer.add_scalar(f"{tag}/AvgReward",    stats["avg_reward"],    total_steps)
        writer.add_scalar(f"{tag}/AvgSteps",     stats["avg_steps"],     total_steps)
        writer.add_scalar(f"{tag}/AvgDeviation", stats["avg_deviation"], total_steps)

        tc = stats["terminal_counts"]
        n  = max(sum(tc.values()), 1)
        writer.add_scalar(f"{tag}/Terminal/Success",     tc.get("success",      0) / n, total_steps)
        writer.add_scalar(f"{tag}/Terminal/WanderedOff", tc.get("wandered_off", 0) / n, total_steps)
        writer.add_scalar(f"{tag}/Terminal/OrientFail",  tc.get("orient_fail",  0) / n, total_steps)
        writer.add_scalar(f"{tag}/Terminal/Truncated",   tc.get("truncated",    0) / n, total_steps)

def main():
    parser = argparse.ArgumentParser(description="Train a TD3 agent on the Environment")
    parser.add_argument("--seed",          type=int,  default=0,             help="Random seed")
    parser.add_argument("--config",        type=str,  default="config.yaml", help="Path to config file")
    parser.add_argument("--push-from-edge", action="store_true", dest="push_from_edge",
                        help="Use edge-based heading and apply force at edge")
    parser.add_argument("--render_test",   action="store_true",              help="Render during testing")
    parser.add_argument("--test_episodes", type=int,  default=25,            help="Episodes per eval")
    parser.add_argument("--eval_freq",     type=int,  default=25,            help="Eval every N episodes")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    env_cfg = config["env"].copy()
    env_cfg["push_from_edge"] = args.push_from_edge or env_cfg.get("push_from_edge", False)
    config["env"] = env_cfg  # so saved config includes flag
    agent_cfg    = config["agent"]
    training_cfg = config["training"]
    eval_wide_cfg = training_cfg.get("eval_wide", {})
    eval_wide_enabled = bool(eval_wide_cfg.get("enabled", False))
    eval_wide_mass_range = eval_wide_cfg.get("mass_range", [10.0, 40.0])
    eval_wide_friction_range = eval_wide_cfg.get("friction_range", [0.3, 0.7])


    eval_freq = int(training_cfg.get("eval_freq", args.eval_freq))

    # Set random seeds
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

    rng_replay      = np.random.default_rng(seed + 10)
    rng_exploration = np.random.default_rng(seed + 20)
    rng_gen         = np.random.default_rng(seed + 30)

    dev       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_rng = torch.Generator(device=dev).manual_seed(seed + 40)

    # Create timestamped run directory
    timestamp  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name   = f"run-{seed}-{timestamp}"
    run_dir    = os.path.join("runs", run_name)
    models_dir = os.path.join(run_dir, "models")
    logs_dir   = os.path.join(run_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir,   exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    with open(os.path.join(run_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    writer = SummaryWriter(logs_dir)

    # --- Training env ---
    env = SimplePathFollowingEnv(**env_cfg)
    train_mode = training_cfg.get("train_segment_mode", "short").lower()
    if train_mode == "short":
        env.segment_length = env_cfg.get("short_segment_length", 0.3)
    elif train_mode == "full":
        env.segment_length = None
    else:
        raise ValueError(f"Unknown train_segment_mode: {train_mode}")
    env.reset(seed=seed)

    # --- Short-seg eval env ---
    test_env_short_cfg = env_cfg.copy()
    test_env_short_cfg['gui'] = False
    test_env_short_cfg['segment_length'] = env_cfg.get('short_segment_length', 0.3)
    test_env_short = SimplePathFollowingEnv(**test_env_short_cfg)
    test_env_short.reset(seed=seed + 1)

    # --- Full-arc eval env ---
    test_env_full_cfg = env_cfg.copy()
    test_env_full_cfg['gui'] = False
    test_env_full_cfg['max_steps'] = 1000
    test_env_full_cfg['segment_length'] = None
    test_env_full = SimplePathFollowingEnv(**test_env_full_cfg)
    test_env_full.reset(seed=seed + 2)

    # --- Random-arc generalization eval env ---
    test_env_gen_cfg = env_cfg.copy()
    test_env_gen_cfg['gui'] = False
    test_env_gen_cfg['max_steps'] = 1000
    test_env_gen_cfg['segment_length'] = None
    test_env_gen = SimplePathFollowingEnv(**test_env_gen_cfg)
    test_env_gen.reset(seed=seed + 3)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    if eval_wide_enabled:
        test_env_short_wide_cfg = test_env_short_cfg.copy()
        test_env_short_wide_cfg["mass_range"] = eval_wide_mass_range
        test_env_short_wide_cfg["friction_range"] = eval_wide_friction_range
        test_env_short_wide = SimplePathFollowingEnv(**test_env_short_wide_cfg)
        test_env_short_wide.reset(seed=seed + 11)

        test_env_full_wide_cfg = test_env_full_cfg.copy()
        test_env_full_wide_cfg["mass_range"] = eval_wide_mass_range
        test_env_full_wide_cfg["friction_range"] = eval_wide_friction_range
        test_env_full_wide = SimplePathFollowingEnv(**test_env_full_wide_cfg)
        test_env_full_wide.reset(seed=seed + 12)

        test_env_gen_wide_cfg = test_env_gen_cfg.copy()
        test_env_gen_wide_cfg["mass_range"] = eval_wide_mass_range
        test_env_gen_wide_cfg["friction_range"] = eval_wide_friction_range
        test_env_gen_wide = SimplePathFollowingEnv(**test_env_gen_wide_cfg)
        test_env_gen_wide.reset(seed=seed + 13)



    envs_to_seed = [env, test_env_short, test_env_full, test_env_gen]
    if eval_wide_enabled:
        envs_to_seed += [test_env_short_wide, test_env_full_wide, test_env_gen_wide]

    for i, e in enumerate(envs_to_seed):
        e.action_space.seed(seed + 100 + i)
        e.observation_space.seed(seed + 200 + i)

    agent = TD3(
        lr=agent_cfg.get("lr", 3e-4),
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        max_torque=env_cfg.get("max_torque", 50.0),
        torch_rng=torch_rng
    )
    replay_buffer = ReplayBuffer(
        max_size=agent_cfg.get("replay_buffer_max_size", 5e5),
        rng=rng_replay
    )

    # Training hyperparameters
    episodes          = training_cfg.get("episodes", 3000)
    start_timesteps   = training_cfg.get("start_timesteps", 2000)
    batch_size        = agent_cfg.get("batch_size", 256)
    gamma             = agent_cfg.get("gamma", 0.99)
    polyak            = agent_cfg.get("polyak", 0.99)
    policy_noise      = agent_cfg.get("policy_noise", 0.2)
    noise_clip        = agent_cfg.get("noise_clip", 0.5)
    policy_delay      = agent_cfg.get("policy_delay", 2)
    n_iter            = agent_cfg.get("n_iter", 2)
    exploration_noise = agent_cfg.get("exploration_noise", 0.1)
    save_freq         = training_cfg.get("save_freq", 500)

    total_steps       = 0
    best_full_success = 0.0
    next_eval_ep      = 0

    # Rolling train reward for smoothed TensorBoard signal
    train_reward_hist = deque(maxlen=100)

    # Timing
    start_time     = time.perf_counter()
    last_mark_time = start_time
    milestone      = 10_000
    next_mark      = milestone

    # Composite model saving
    best_composite = 0.0

    for ep in range(episodes):
        with contextlib.redirect_stdout(io.StringIO()):
            state, _ = env.reset()

        ep_reward       = 0.0
        ep_steps        = 0
        ep_reward_comps = {"R_Progress": 0.0, "R_Speed": 0.0,
                           "R_Constraint_Lat": 0.0, "R_Constraint_Ori": 0.0,
                           "R_Eff_Spin": 0.0}
        terminal_event  = None

        while True:
            total_steps += 1
            ep_steps    += 1

            if total_steps < start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state).squeeze(0)
                noise  = rng_exploration.normal(0, exploration_noise, size=action_dim)
                action = np.clip(action + noise, env.action_space.low, env.action_space.high)

            with contextlib.redirect_stdout(io.StringIO()):
                next_state, reward, done, truncated, info = env.step(action)

            for k in ep_reward_comps:
                ep_reward_comps[k] += info["reward_comps"].get(k, 0.0)
            terminal_event = info["terminal_event"]

            # Time milestone logging
            if total_steps >= next_mark:
                now            = time.perf_counter()
                elapsed_total  = now - start_time
                elapsed_window = now - last_mark_time
                writer.add_scalar("Time/ElapsedSec",
                                  elapsed_total, total_steps)
                writer.add_scalar("Time/StepsPerSecWindow",
                                  milestone / max(elapsed_window, 1e-9), total_steps)
                writer.add_scalar("Time/StepsPerSecCumulative",
                                  total_steps / max(elapsed_total, 1e-9), total_steps)
                print(f"[Time] {total_steps} steps — {elapsed_total:.1f}s total, "
                      f"{milestone / elapsed_window:.1f} steps/s")
                last_mark_time = now
                next_mark     += milestone

            episode_over = done or truncated
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state      = next_state
            ep_reward += reward

            if replay_buffer.size > batch_size:
                agent.update(replay_buffer, n_iter, batch_size, gamma,
                             polyak, policy_noise, noise_clip, policy_delay)

            if episode_over:
                break

        # --- Per-episode TensorBoard logging ---
        train_reward_hist.append(ep_reward)
        rolling_reward = sum(train_reward_hist) / len(train_reward_hist)

        writer.add_scalar("Train/EpisodeReward",    ep_reward,      total_steps)
        writer.add_scalar("Train/RollingReward100", rolling_reward, total_steps)
        writer.add_scalar("Train/EpisodeLength",    ep_steps,       total_steps)
        writer.add_scalar("Train/Env/Mass",         info["sampled_mass"],     total_steps)
        writer.add_scalar("Train/Env/Friction",     info["sampled_friction"], total_steps)

        # Reward components as per-step averages
        for k, v in ep_reward_comps.items():
            writer.add_scalar(f"Train/RewardComps/{k}", v / max(ep_steps, 1), total_steps)

        # Terminal event flags
        writer.add_scalar("Train/Terminal/Success",
                          1.0 if terminal_event == "success"      else 0.0, total_steps)
        writer.add_scalar("Train/Terminal/WanderedOff",
                          1.0 if terminal_event == "wandered_off" else 0.0, total_steps)
        writer.add_scalar("Train/Terminal/Truncated",
                          1.0 if (terminal_event is None and truncated) else 0.0, total_steps)

        print(f"[Ep {ep:4d}] Reward: {ep_reward:.2f} | "
              f"Rolling100: {rolling_reward:.2f} | "
              f"Steps: {ep_steps} | "
              f"Terminal: {terminal_event}")

        # --- Periodic unified evaluation ---
        if ep >= next_eval_ep:
            print(f"\n--- Eval at ep {ep} (total_steps={total_steps}) ---")

            # In-distribution eval (uses env.mass_range / env.friction_range)
            eval_results_id = run_eval(
                test_env_short, test_env_full, test_env_gen,
                agent, args.test_episodes, rng_gen
            )
            log_eval_to_tb(writer, eval_results_id, total_steps, prefix="EvalID")

            short_id = eval_results_id["short"]["success_rate"]
            full_id  = eval_results_id["full"]["success_rate"]
            gen_id   = eval_results_id["gen"]["success_rate"]

            print(f"  ID   | Short: {short_id:.2f} | Full: {full_id:.2f} | Gen: {gen_id:.2f}")

            # Wide-range eval (same evals, wider mass/friction)
            if eval_wide_enabled:
                eval_results_wide = run_eval(
                    test_env_short_wide, test_env_full_wide, test_env_gen_wide,
                    agent, args.test_episodes, rng_gen
                )
                log_eval_to_tb(writer, eval_results_wide, total_steps, prefix="EvalWide")

                short_w = eval_results_wide["short"]["success_rate"]
                full_w  = eval_results_wide["full"]["success_rate"]
                gen_w   = eval_results_wide["gen"]["success_rate"]

                print(f"  Wide | Short: {short_w:.2f} | Full: {full_w:.2f} | Gen: {gen_w:.2f}")

            # Keep best-model logic unchanged for now (ID composite),
            # or change later once you decide convergence criteria.
            composite = 0.5 * full_id + 0.5 * gen_id
            if composite > best_composite:
                best_composite = composite
                agent.save(models_dir, "best_model")
                print(f"  New best composite (ID): {best_composite:.2f} (full={full_id:.2f}, gen={gen_id:.2f})")

            next_eval_ep = ep + eval_freq

        # Periodic checkpoint
        if ep % save_freq == 0 and ep > 0:
            agent.save(models_dir, f"checkpoint_ep{ep}")
            print(f"  Checkpoint saved at ep {ep}")

    # --- Final evaluation ---
    print("\n--- Final Evaluation (ID) ---")
    final_results_id = run_eval(
        test_env_short, test_env_full, test_env_gen,
        agent, args.test_episodes * 2, rng_gen
    )
    log_eval_to_tb(writer, final_results_id, total_steps, prefix="EvalID_Final")

    print(f"FINAL ID   Short: Success={final_results_id['short']['success_rate']:.2f}, Reward={final_results_id['short']['avg_reward']:.2f}")
    print(f"FINAL ID   Full:  Success={final_results_id['full']['success_rate']:.2f}, Reward={final_results_id['full']['avg_reward']:.2f}")
    print(f"FINAL ID   Gen:   Success={final_results_id['gen']['success_rate']:.2f}, AvgDev={final_results_id['gen']['avg_deviation']:.3f}m")

    if eval_wide_enabled:
        print("\n--- Final Evaluation (Wide) ---")
        final_results_wide = run_eval(
            test_env_short_wide, test_env_full_wide, test_env_gen_wide,
            agent, args.test_episodes * 2, rng_gen
        )
        log_eval_to_tb(writer, final_results_wide, total_steps, prefix="EvalWide_Final")

        print(f"FINAL Wide Short: Success={final_results_wide['short']['success_rate']:.2f}, Reward={final_results_wide['short']['avg_reward']:.2f}")
        print(f"FINAL Wide Full:  Success={final_results_wide['full']['success_rate']:.2f}, Reward={final_results_wide['full']['avg_reward']:.2f}")
        print(f"FINAL Wide Gen:   Success={final_results_wide['gen']['success_rate']:.2f}, AvgDev={final_results_wide['gen']['avg_deviation']:.3f}m")

    agent.save(models_dir, "final_model")
    print("Final model saved.")

    env.close()
    test_env_short.close()
    test_env_full.close()
    test_env_gen.close()
    if eval_wide_enabled:
        test_env_short_wide.close()
        test_env_full_wide.close()
        test_env_gen_wide.close()

    writer.close()


if __name__ == "__main__":
    main()