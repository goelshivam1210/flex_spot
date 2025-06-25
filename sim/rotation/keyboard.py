# keyboard_control.py
import time
import numpy as np
import mujoco
import mujoco.viewer
import yaml
import glfw
from env_mujoco import SimplePathFollowingEnv  # Update import name as needed

key_states = {
    # Movement keys
    'up': False, 'down': False, 'left': False, 'right': False,
    'turn_left': False, 'turn_right': False,
    # Control keys
    'reset': False, 'stop': False, 'quit': False,
    # Scale adjustment keys
    'force_up': False, 'force_down': False,
    'torque_up': False, 'torque_down': False
}

def key_callback(*args):
    """
    A callback function that MuJoCo's viewer will call on every key press/release.
    It updates the global `key_states` dictionary.
    """

    if len(args) == 1:
        keycode, = args
        action  = glfw.PRESS
        mods    = 0

    elif len(args) == 3:
        keycode, action, mods = args

    elif len(args) == 5:
        _, keycode, _, action, mods = args

    else:
        return

    key_map = {
        glfw.KEY_UP: 'up',
        glfw.KEY_DOWN: 'down',
        glfw.KEY_LEFT: 'left',
        glfw.KEY_RIGHT: 'right',
        glfw.KEY_A: 'turn_left',
        glfw.KEY_D: 'turn_right',
        glfw.KEY_W: 'force_up',
        glfw.KEY_S: 'force_down',
        glfw.KEY_Z: 'torque_up',
        glfw.KEY_X: 'torque_down',
        glfw.KEY_R: 'reset',
        glfw.KEY_SPACE: 'stop',
        glfw.KEY_Q: 'quit',
    }
    
    if keycode in key_map:
        key_name = key_map[keycode]
        is_pressed = (action == glfw.PRESS)
        
        # For continuous actions (holding a key down), store the pressed state.
        if key_name in ['up', 'down', 'left', 'right', 'turn_left', 'turn_right']:
            key_states[key_name] = is_pressed
        # For single-trigger actions, only set to True on press.
        elif is_pressed:
            key_states[key_name] = True
        
        return True
    
    return False

    

def keyboard_control():
    """Run the environment with keyboard control for force_x, force_y, torque_z"""
    # Load configuration to get default parameters, ensuring consistency.
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        env_cfg = config.get('env', {})
    except FileNotFoundError:
        print("Warning: config.yaml not found. Using default environment parameters.")
        env_cfg = {}
    
    env = SimplePathFollowingEnv(model_path='scene.xml', **env_cfg)

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
    
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            loop_start_time = time.time()

            # Poll for keyboard events
            if key_states['quit']:
                break
            if key_states['reset']:
                state, _ = env.reset()
                action.fill(0.0)
                print("\nEnvironment Reset!")
                key_states['reset'] = False
            if key_states['stop']:
                action.fill(0.0)
                print("All forces stopped.")
                key_states['stop'] = False

            # Scale adjustment logic (single press)
            if key_states['force_up']:
                force_scale = min(1.0, force_scale + 0.1)
                print(f"Force scale: {force_scale:.1f}")
                key_states['force_up'] = False
            if key_states['force_down']:
                force_scale = max(0.1, force_scale - 0.1)
                print(f"Force scale: {force_scale:.1f}")
                key_states['force_down'] = False
            if key_states['torque_up']:
                torque_scale = min(1.0, torque_scale + 0.1)
                print(f"Torque scale: {torque_scale:.1f}")
                key_states['torque_up'] = False
            if key_states['torque_down']:
                torque_scale = max(0.1, torque_scale - 0.1)
                print(f"Torque scale: {torque_scale:.1f}")
                key_states['torque_down'] = False
            
            # By default, decay forces/torques gradually (for smoother control)
            action *= 0.8

            if key_states['up']:
                action[0] = force_scale  # Forward
            if key_states['down']:
                action[0] = -force_scale # Backward
            if key_states['right']:
                action[1] = -force_scale # Strafe Right
            if key_states['left']:
                action[1] = force_scale  # Strafe Left
                
            # A/D keys for torque
            if key_states['turn_left']:
                action[2] = torque_scale # Rotate counter-clockwise
            if key_states['turn_right']:
                action[2] = -torque_scale # Rotate clockwise

            # Clip actions to valid range
            action = np.clip(action, -1.0, 1.0)
            
            # Apply action to environment
            state, reward, done, truncated, info = env.step(action)
            
            # Print state info periodically
            if env.steps % 20 == 0:  # Less frequent to reduce spam
                print(f"\n--- Step {env.steps} ---")
                print("State components:")
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
            
            if done or truncated:
                if done: print(f"\n🎉 SUCCESS! Goal reached!")
                else: print(f"\n⏰ Episode truncated.")
                state, _ = env.reset()
                action.fill(0.0)
                
            viewer.sync()
            
            time_until_next_frame = (env.model.opt.timestep * env.sim_steps) - (time.time() - loop_start_time)
            if time_until_next_frame > 0:
                time.sleep(time_until_next_frame)
        
        # Small sleep to control frame rate
        time.sleep(0.02)
    
    env.close()

if __name__ == "__main__":
    keyboard_control()