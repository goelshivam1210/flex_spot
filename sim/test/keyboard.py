# keyboard_control.py

import time
import numpy as np
import pybullet as p
from env import SimplePathFollowingEnv

def keyboard_control():
    """Run the environment with keyboard control"""
    # Create environment with GUI
    env = SimplePathFollowingEnv(
        gui=True,
        max_force=20.0,
        friction=0.5,
        linear_damping=0.9,
        angular_damping=0.9,
        goal_thresh=0.2,
        max_steps=500
    )
    
    # Reset environment
    state, _ = env.reset()
    
    # Initial action: no force
    action = np.array([0.0, 0.0, 0.5])  # Default force magnitude 0.5
    force_magnitude = 0.5  # Store separately
    
    print("\nKeyboard Control Instructions:")
    print("-------------------------------")
    print("Arrow keys: Apply force (up/down/left/right)")
    print("W/S: Increase/decrease force magnitude")
    print("R: Reset environment")
    print("Q: Exit")
    print("-------------------------------")
    
    running = True
    
    while running:
        # Poll for keyboard events
        keys = p.getKeyboardEvents()
        
        # By default, no force direction
        action[0] = 0.0
        action[1] = 0.0
        
        # Process key events
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            running = False
            print("Exiting...")
        
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            state, _ = env.reset()
            print("Environment reset")
        
        if ord('w') in keys and keys[ord('w')] & p.KEY_WAS_TRIGGERED:
            force_magnitude = min(1.0, force_magnitude + 0.1)
            action[2] = force_magnitude
            print(f"Force magnitude: {force_magnitude:.1f}")
        
        if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
            force_magnitude = max(0.1, force_magnitude - 0.1)
            action[2] = force_magnitude
            print(f"Force magnitude: {force_magnitude:.1f}")
        
        # Movement with arrow keys
        key_pressed = False
        
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            action[0] = -1.0
            key_pressed = True
        
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            action[0] = 1.0
            key_pressed = True
        
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            action[1] = 1.0
            key_pressed = True
        
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            action[1] = -1.0
            key_pressed = True
        
        # Apply action to environment (even if no key pressed, to allow natural physics)
        state, reward, done, truncated, info = env.step(action)
        
        # Print state info periodically
        if env.steps % 10 == 0:
            state_components = {
                    "position_error_x": state[0],
                    "position_error_y": state[1],
                    "orientation_error": state[2],
                    "progress": state[3],
                    "deviation": state[4]
            }
            print(f"State: {state_components}")
            print(f"Progress: {info['progress']:.2f}, Deviation: {info['deviation']:.2f}, Reward: {reward:.2f}")
        
        # Reset if episode is done
        if done or truncated:
            print("Episode ended.")
            state, _ = env.reset()
        
        # Small sleep to control frame rate
        time.sleep(0.02)
    
    env.close()

if __name__ == "__main__":
    keyboard_control()