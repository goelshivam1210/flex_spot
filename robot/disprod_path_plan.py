import argparse
import sys
import os
import numpy as np
import jax
import time
from omegaconf import OmegaConf

from argparse import Namespace

# Attempt to get DISPROD_PATH from environment variables.
# This path is crucial for the script to find other modules like 'utils' and 'planners'.
# DISPROD_PATH = os.getenv("DISPROD_PATH")
DISPROD_PATH = "/home/shivam/spot/spot_flex_novelty/code/DiSProD"

# If DISPROD_PATH is not set, print a warning and attempt to infer it
# based on the script's location. This might not work in all setups.
if not DISPROD_PATH:
    print("WARNING: DISPROD_PATH environment variable not set.")
    print("Please set it to the root of your DiSProD project (e.g., export DISPROD_PATH=/path/to/DiSProD).")
    print("Attempting to infer DISPROD_PATH based on script location. This may not be accurate.")
    # Assuming this script is placed at the root of the DiSProD project,
    # or one level deeper (e.g., in a 'scripts' folder within DiSProD).
    # Adjust this logic if your file structure is different.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == "scripts": # If placed in a 'scripts' subfolder
        DISPROD_PATH = os.path.dirname(current_dir)
    else: # Assume it's at the root
        DISPROD_PATH = current_dir

    if not os.path.exists(os.path.join(DISPROD_PATH, "utils")) or \
       not os.path.exists(os.path.join(DISPROD_PATH, "planners")):
        print(f"ERROR: Inferred DISPROD_PATH '{DISPROD_PATH}' does not seem to be the correct root.")
        print("Please manually set the DISPROD_PATH environment variable.")
        sys.exit(1)
    else:
        print(f"Inferred DISPROD_PATH: {DISPROD_PATH}")

# Add DISPROD_PATH to sys.path to enable importing modules from the project
if DISPROD_PATH not in sys.path:
    sys.path.append(DISPROD_PATH)

# Import necessary modules from the DiSProD project
from utils.common_utils import setup_environment, set_global_seeds, prepare_config, update_config_with_args, setup_output_dirs
from planners.gym_interface import setup_planner

# Define the specific configuration file path for the continuous Dubins car
DISPROD_CONF_PATH = os.path.join(DISPROD_PATH, "config")
DUBINS_CAR_CONFIG_PATH = os.path.join(DISPROD_PATH, "config", "continuous_dubins_car.yaml")
DUBINS_CAR_ENV_NAME = "continuous_dubins_car"
DUBINS_CAR_ENV_ALIAS = "cdc"


class DummyArgs:
    """
    A dummy class to mimic argparse.Namespace for configuration updates.
    This version is tailored for the continuous Dubins car environment,
    including parameters for custom start and goal positions.
    """
    def __init__(self, seed=42, n_episodes=1, **kwargs):
        # Args listed in run_gym.py
        self.env = "cdc" # Hardcoded for continuous Dubins car
        self.render = "False"  # Disable rendering for waypoint generation
        self.seed = seed
        self.run_name = f"dubins_waypoint_gen_{int(time.time())}" # Unique run name for Dubins
        self.depth = 30
        self.alpha = 0.0
        # self.reward_sparsity = None
        self.n_episodes = n_episodes
        self.alg = "disprod"
        self.obstacles_config_file = "dubins" # Default for Dubins car
        self.map_name = "no-ob-1" # Default map for Dubins car (no obstacles)
        self.headless = "True" # Assume headless operation
        # self.n_samples = None  # Used for the non-disprod planners like CEM or MPPI
        # self.step_size = None  # Use default
        # self.step_size_var = None
        self.taylor_expansion_mode = "no_var"
        # self.n_restarts = None
        # self.n_actions = 1

        # Additional possible args
        self.env_name = DUBINS_CAR_ENV_NAME # Hardcoded environment name
        # self.save_as_gif = "False" # Disable GIF saving
        # self.debug_planner = False # Disable debug logging from planner
        # self.plot_imagined_trajectory = False # Disable trajectory plotting
        # self.log_dir = os.path.join(DISPROD_PATH, "logs", self.run_name) # Dummy log directory
        # self.log_file = "log.txt" # Dummy log file name
        # self.graph_dir = os.path.join(DISPROD_PATH, "graphs", self.run_name) # Dummy graph directory

        # Fixed start position for the Dubins car
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_theta = 0.0 # Initial heading (radians)

        # Default goal position (can be overridden by kwargs)
        self.goal_x = 10.0
        self.goal_y = 10.0


        # Update with any additional keyword arguments provided
        for k, v in kwargs.items():
            setattr(self, k, v)

def generate_waypoints_dubins(goal_x, goal_y, seed=42, n_episodes=1, **kwargs):
    """
    Generates waypoints (states) specifically for the continuous Dubins car environment.
    The car always starts at (0, 0) with 0 heading, and aims for the specified goal.

    This function acts as a generator, yielding the environment's state after each step
    of the simulation. It directly uses the configuration for the continuous Dubins car.

    Args:
        goal_x (float): The x-coordinate of the target goal position.
        goal_y (float): The y-coordinate of the target goal position.
        seed (int, optional): The base seed for the random number generators. Defaults to 42.
        n_episodes (int, optional): The number of episodes to simulate. Defaults to 1.
        **kwargs: Arbitrary keyword arguments that will be used to override or set
                  parameters in the loaded configuration. These correspond to the
                  command-line arguments accepted by run_gym.py (e.g., 'depth', 'alg',
                  'n_samples', 'step_size').

    Yields:
        numpy.ndarray or jax.numpy.ndarray: The state (waypoint) of the environment
                                            at the current step, in the format
                                            (x, y, theta, linear_velocity, angular_velocity).
    """
    # Create a dummy args object tailored for Dubins car, including the specified goal
    args_obj = DummyArgs(seed, n_episodes, goal_x=goal_x, goal_y=goal_y, **kwargs)
    args_dict = vars(args_obj)
    name_space = Namespace(**args_dict)

    # Prepare and update the configuration using the specific Dubins car config
    # The config_file path is passed as the directory containing the config, not the file itself
    cfg = prepare_config(DUBINS_CAR_ENV_NAME, DISPROD_CONF_PATH)
    cfg = update_config_with_args(cfg, name_space, base_path=DISPROD_PATH)

    run_name = cfg["run_name"]
    # if name_space.headless.lower() == "true" and cfg["save_as_gif"]:
    #     setup_virtual_display()

    setup_output_dirs(cfg, run_name, DISPROD_PATH)

    # Initialize the environment and planner once outside the episode loop
    # as these are generally expensive operations.
    env = setup_environment(cfg)
    agent = setup_planner(env, cfg)

    # Generate seeds for each episode
    seeds = list(range(seed, seed + n_episodes))

    for idx in range(n_episodes):
        current_seed = seeds[idx]
        set_global_seeds(seed=current_seed, env=env)
        key = jax.random.PRNGKey(current_seed)

        done = False
        
        # Reset environment. This will load the map's default start/goal.
        env.reset()

        # Explicitly set the environment's start and goal to the desired values.
        # This overrides what was loaded from the config file's map.
        env.env.x = args_obj.start_x
        env.env.y = args_obj.start_y
        env.env.theta = args_obj.start_theta
        env.env.linear_velocity = 0.0 # Ensure initial velocity is zero
        env.env.angular_velocity = 0.0 # Ensure initial angular velocity is zero
        env.env.goal_x = args_obj.goal_x
        env.env.goal_y = args_obj.goal_y

        # Get the initial observation based on the overridden start state
        obs = np.array((env.x, env.y, env.theta, env.linear_velocity, env.angular_velocity))

        # Reset the agent with the actual starting observation
        ac_seq, key = agent.reset(key)

        # Yield the initial observation (waypoint) of the episode
        yield obs

        # Simulation loop for the current episode
        while not done:
            # Choose action using the planner
            action, ac_seq, tau, key = agent.choose_action(obs, ac_seq, key)
            
            # Step the environment with the chosen action
            obs, reward, done, _ = env.step(np.array(action))
            
            # Yield the new observation (waypoint) after the step
            yield obs

        # Close the environment after each episode to release resources
        env.close()


if __name__ == "__main__":
    # --- Example Usage ---
    # Before running, make sure you have set the DISPROD_PATH environment variable
    # to the root directory of your DiSProD project.
    # Example (in your terminal): export DISPROD_PATH="/path/to/your/DiSProD"

    # --- Generate waypoints for Continuous Dubins Car with custom goal ---
    print("\n--- Generating waypoints for Continuous Dubins Car (1 episode, Goal at (5, 5)) ---")
    if os.path.exists(DUBINS_CAR_CONFIG_PATH):
        waypoint_counter = 0
        try:
            start_x = 2.5258572
            start_y = -0.28464526
            # Specify the goal_x and goal_y directly
            for waypoint in generate_waypoints_dubins(
                start_x = start_x,
                start_y = start_y,
                goal_x=start_x + 1.0,  # Example goal position
                goal_y=start_y +0,
                n_episodes=1,
                depth=30,  # Example planner parameter (planning horizon)
                alg="disprod", # Example planner algorithm
                map_name="no-ob-1" # Use a map without obstacles
            ):
                print(f"Dubins Car Waypoint {waypoint_counter}: {waypoint}")
                waypoint_counter += 1
            print(f"Total Dubins Car waypoints generated: {waypoint_counter}")
        except Exception as e:
            print(f"An error occurred during Dubins Car waypoint generation: {e}")
    else:
        print(f"Continuous Dubins Car config file not found at: {DUBINS_CAR_CONFIG_PATH}. Skipping example.")

    # print("\n--- Generating waypoints for Continuous Dubins Car (1 episode, Goal at (-8, -8)) ---")
    # if os.path.exists(DUBINS_CAR_CONFIG_PATH):
    #     waypoint_counter = 0
    #     try:
    #         # Another example with a different goal
    #         for waypoint in generate_waypoints_dubins(
    #             goal_x=-8.0,
    #             goal_y=-8.0,
    #             n_episodes=1,
    #             depth=30,
    #             alg="cem", # Using CEM planner for this example
    #             n_samples=50, # CEM-specific parameter
    #             map_name="no-ob-1"
    #         ):
    #             print(f"Dubins Car Waypoint {waypoint_counter}: {waypoint}")
    #             waypoint_counter += 1
    #         print(f"Total Dubins Car waypoints generated: {waypoint_counter}")
    #     except Exception as e:
    #         print(f"An error occurred during Dubins Car waypoint generation: {e}")
    # else:
    #     print(f"Continuous Dubins Car config file not found at: {DUBINS_CAR_CONFIG_PATH}. Skipping example.")

