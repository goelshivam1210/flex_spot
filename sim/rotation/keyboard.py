# keyboard_control.py
import time
import numpy as np
import pybullet as p
from env import SimplePathFollowingEnv  # Update import name as needed

def keyboard_control():
    """Run the environment with keyboard control for force_x, force_y, torque_z"""
    # Create environment with GUI
    env = SimplePathFollowingEnv(
        gui=True,
        max_force=300.0,      # Updated for hollow plywood box
        max_torque=50.0,      # New torque parameter
        friction=0.4,         # Updated for plywood
        linear_damping=0.05,  # Lower damping
        angular_damping=0.1,  # Lower damping
        goal_thresh=0.2,
        max_steps=500
    )
    
    # Reset environment
    state, _ = env.reset()
    
    # Initial action: [force_x, force_y, torque_z]
    action = np.array([0.0, 0.0, 0.0])
    force_scale = 0.5      # Scale factor for forces
    torque_scale = 0.5     # Scale factor for torque
    
    print("\nKeyboard Control Instructions:")
    print("-------------------------------")
    print("TRANSLATION FORCES:")
    print("  Arrow keys: Apply force (up/down/left/right)")
    print("  W/S: Increase/decrease force magnitude")
    print("")
    print("ROTATION TORQUE:")
    print("  A/D: Rotate left/right (torque)")
    print("  Z/X: Increase/decrease torque magnitude")
    print("")
    print("GENERAL:")
    print("  R: Reset environment")
    print("  Q: Exit")
    print("  SPACE: Stop all forces/torques")
    print("-------------------------------")
    print(f"Initial force scale: {force_scale:.1f}")
    print(f"Initial torque scale: {torque_scale:.1f}")
    
    running = True
    while running:
        # Poll for keyboard events
        keys = p.getKeyboardEvents()
        
        # By default, decay forces/torques gradually (for smoother control)
        action[0] *= 0.8  # force_x decay
        action[1] *= 0.8  # force_y decay  
        action[2] *= 0.8  # torque_z decay
        
        # Process key events
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            running = False
            print("Exiting...")
        
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            state, _ = env.reset()
            action = np.array([0.0, 0.0, 0.0])
            print("Environment reset")
        
        if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
            action = np.array([0.0, 0.0, 0.0])
            print("All forces/torques stopped")
        
        # Force magnitude control
        if ord('w') in keys and keys[ord('w')] & p.KEY_WAS_TRIGGERED:
            force_scale = min(1.0, force_scale + 0.1)
            print(f"Force scale: {force_scale:.1f}")
        
        if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
            force_scale = max(0.1, force_scale - 0.1)
            print(f"Force scale: {force_scale:.1f}")
        
        # Torque magnitude control
        if ord('z') in keys and keys[ord('z')] & p.KEY_WAS_TRIGGERED:
            torque_scale = min(1.0, torque_scale + 0.1)
            print(f"Torque scale: {torque_scale:.1f}")
        
        if ord('x') in keys and keys[ord('x')] & p.KEY_WAS_TRIGGERED:
            torque_scale = max(0.1, torque_scale - 0.1)
            print(f"Torque scale: {torque_scale:.1f}")
        
        # Translation forces with arrow keys
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            action[0] = -force_scale  # Force left
        
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            action[0] = force_scale   # Force right
        
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            action[1] = force_scale   # Force forward
        
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            action[1] = -force_scale  # Force backward
        
        # Rotation torque with A/D keys
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            action[2] = torque_scale  # Rotate counter-clockwise
        
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            action[2] = -torque_scale # Rotate clockwise
        
        # Clip actions to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action to environment
        state, reward, done, truncated, info = env.step(action)
        
        # Print state info periodically
        if env.steps % 20 == 0:  # Less frequent to reduce spam
            print(f"\n--- Step {env.steps} ---")
            print(f"State components:")
            print(f"  Lateral error: {state[0]:.2f}")
            print(f"  Longitudinal error: {state[1]:.2f}")
            print(f"  Orientation error: {state[2]:.2f}")
            print(f"  Progress: {state[3]:.2f}")
            print(f"  Deviation: {state[4]:.2f}")
            print(f"  Speed along path: {state[5]:.2f}")
            print(f"Applied action: [fx:{action[0]:.2f}, fy:{action[1]:.2f}, tz:{action[2]:.2f}]")
            print(f"Reward: {reward:.2f}")
        
        # Show active controls
        if env.steps % 60 == 0:  # Every ~1 second
            active_controls = []
            if abs(action[0]) > 0.1:
                active_controls.append(f"Fx:{action[0]:.1f}")
            if abs(action[1]) > 0.1:
                active_controls.append(f"Fy:{action[1]:.1f}")
            if abs(action[2]) > 0.1:
                active_controls.append(f"Tz:{action[2]:.1f}")
            
            if active_controls:
                print(f"Active: {', '.join(active_controls)}")
        
        # Reset if episode is done
        if done or truncated:
            if done:
                print(f"\n🎉 SUCCESS! Goal reached in {env.steps} steps!")
            else:
                print(f"\n⏰ Episode truncated after {env.steps} steps")
            
            print("Press 'R' to reset or 'Q' to quit")
            
            # Wait for user input
            waiting = True
            while waiting:
                keys = p.getKeyboardEvents()
                if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                    state, _ = env.reset()
                    action = np.array([0.0, 0.0, 0.0])
                    print("Environment reset")
                    waiting = False
                elif ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    running = False
                    waiting = False
                time.sleep(0.02)
        
        # Small sleep to control frame rate
        time.sleep(0.02)
    
    env.close()

if __name__ == "__main__":
    keyboard_control()