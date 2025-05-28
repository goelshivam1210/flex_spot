# visualize_policy.py

import os
import time
import argparse
import subprocess
import numpy as np
import torch
import pybullet as p

from env import SimplePathFollowingEnv
from td3 import TD3

def check_ffmpeg_installed():
    """Check if ffmpeg is installed and accessible"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trained policy")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model")
    parser.add_argument("--model_name", type=str, default="best_model", help="Model name to load")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--record", action="store_true", help="Record video using PyBullet's recorder")
    parser.add_argument("--record_fps", type=int, default=30, help="Target FPS for recording")
    parser.add_argument("--playback_speed", type=float, default=0.3, help="Playback speed multiplier (slower is better for recording)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check ffmpeg installation if recording
    if args.record and not check_ffmpeg_installed():
        print("ERROR: ffmpeg is not installed or not in PATH. PyBullet recording requires ffmpeg.")
        print("Please install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  Conda: conda install -c conda-forge ffmpeg")
        return
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Prepare recording directory if needed
    if args.record:
        output_dir = os.path.join(args.model_dir, "videos")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_path = os.path.join(output_dir, f"{args.model_name}_{timestamp}.mp4")
        print(f"Video will be saved to: {video_path}")

    # Create environment with GUI
    env = SimplePathFollowingEnv(
        gui=True,
        max_force=20.0,
        friction=0.5,
        linear_damping=0.9,
        angular_damping=0.9,
        goal_thresh=0.2,
        max_steps=500,
        goal_reward=100
    )
    
    # Initialize TD3 agent with the same parameters
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = TD3(
        lr=0.001,
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action
    )
    
    # Load the model
    print(f"Loading model from {args.model_dir}/{args.model_name}")
    agent.load(args.model_dir, args.model_name)
    
    # Setup camera for visualization - give a better view
    p.resetDebugVisualizerCamera(
        cameraDistance=4.0,
        cameraYaw=30,  # Different angle
        cameraPitch=-40,
        cameraTargetPosition=[0, 0, 0]
    )

    # Slow down simulation for better viewing if recording
    if args.record:
        # Add delay before starting recording to make sure GUI is fully loaded
        print("Waiting for GUI to initialize...")
        time.sleep(2)
    
    # Start video recording if requested
    log_id = None
    if args.record:
        print(f"Starting video recording to {video_path}")
        try:
            log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)
            print(f"Recording started with log ID: {log_id}")
            
            # Add a small delay to ensure recording starts properly
            time.sleep(0.5)
        except Exception as e:
            print(f"ERROR starting PyBullet recording: {e}")
            return
    
    # Run episodes
    for episode in range(args.episodes):
        print(f"Episode {episode+1}/{args.episodes}")
        state, _ = env.reset()
        
        # Short delay between episodes
        time.sleep(0.5)
        
        # For tracking success
        done = False
        truncated = False
        ep_reward = 0
        ep_steps = 0
        ep_deviation = 0
        
        # Run the episode
        while not (done or truncated):
            # Sleep to slow down simulation for better visualization
            # This is crucial for recording - must ensure simulation doesn't run too fast
            time.sleep(1.0 / (args.record_fps * args.playback_speed))
            
            # Select action
            action = agent.select_action(np.array(state))
            if action.ndim > 1:
                action = action.squeeze(0)
                
            # Apply action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update stats
            state = next_state
            ep_reward += reward
            ep_steps += 1
            ep_deviation += info["deviation"]
        
        # End of episode stats
        avg_deviation = ep_deviation / ep_steps
        success = done and info["progress"] > 0.95 and info["deviation"] < env.goal_thresh
        
        print(f"  Reward: {ep_reward:.2f}")
        print(f"  Steps: {ep_steps}")
        print(f"  Avg Deviation: {avg_deviation:.4f}")
        print(f"  Success: {success}")
        
        # Add delay after episode to ensure the results are visible in recording
        if args.record:
            # Show success message in GUI
            if success:
                p.addUserDebugText("Goal reached successfully!", [0, 0, 1], textColorRGB=[0, 1, 0], textSize=1.5)
            else:
                p.addUserDebugText("Episode ended", [0, 0, 1], textColorRGB=[1, 0, 0], textSize=1.5)
            
            # Pause to show the final state
            time.sleep(2.0)
            
            # Clear text
            p.removeAllUserDebugItems()
    
    # Make sure to capture the final state
    if args.record:
        print("Finalizing recording...")
        time.sleep(1.0)
    
    # Stop recording if it was enabled
    if args.record and log_id is not None:
        print(f"Stopping video recording with log ID: {log_id}")
        p.stopStateLogging(log_id)
        
        # Wait for the video file to be finalized
        time.sleep(1.0)
        
        # Verify the video file was created
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path) / (1024*1024)
            print(f"Video recording completed and saved to {video_path}")
            print(f"File size: {file_size:.2f} MB")
            
            if file_size < 0.1:
                print("WARNING: Video file is very small, recording may not have worked properly.")
        else:
            print(f"WARNING: Video file not found at expected location: {video_path}")
    
    # Small delay before closing to ensure recording completes
    time.sleep(1.0)
    env.close()
    print("Visualization complete!")

if __name__ == "__main__":
    main()