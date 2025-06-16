# run_and_debug.py

import os
import platform
import numpy as np
import imageio
import mujoco
import yaml
from env_mujoco import SimplePathFollowingEnv

# Setup for headless rendering on Linux servers
if platform.system() == 'Linux':
    os.environ['MUJOCO_GL'] = 'egl'

def run_and_debug_env(config: dict):
    """
    Loads config, runs the environment, and prints debug information.
    """
    try:
        env_config = config.get('env', {})
        training_config = config.get('training', {})
        
        env = SimplePathFollowingEnv(model_path='scene.xml', **env_config)
        env.reset()

        print("\n" + "="*50)
        print("DEBUG: Environment Initialized with Parameters:")
        print(f"  - Max Force: {env.max_force} N")
        print(f"  - Max Torque: {env.max_torque} Nm")

        dof_adr = env.model.joint('box_joint').dofadr[0]
        print(f"  - Linear Damping: {env.model.dof_damping[dof_adr]}")
        print(f"  - Angular Damping: {env.model.dof_damping[dof_adr+3]}")
        
        print(f"  - Box Geom Friction (sliding): {env.model.geom('box_geom').friction[0]}")
        print("="*50 + "\n")

        renderer = mujoco.Renderer(env.model, height=480, width=640)
        cam = mujoco.MjvCamera()
        writer = imageio.get_writer("debug_video.mp4", fps=env.metadata['render_fps'])

        for i in range(training_config.get('episodes', 200)):
            action = np.array([1.0, 0.0, 0.2]) 
            
            applied_force_newtons = action[0] * env.max_force
            applied_torque_nm = action[2] * env.max_torque
            print(f"Step {i+1:03d} | Applied Force: {applied_force_newtons:.1f} N | Applied Torque: {applied_torque_nm:.1f} Nm", end="")

            obs, reward, done, truncated, info = env.step(action)

            box_vel = env.data.qvel[0:3]
            box_acc = env.data.qacc[dof_adr:dof_adr+3]
            
            print(f" | Vel:({box_vel[0]:.3f}, {box_vel[1]:.3f}, {box_vel[2]:.3f}) | Acc:({box_acc[0]:.3f}, {box_acc[1]:.3f}, {box_acc[2]:.3f})")

            cam.lookat = [0.0, 0.0, 0.2]
            cam.distance = 5.0
            cam.azimuth = 90
            cam.elevation = -25
            renderer.update_scene(env.data, camera=cam)
            frame = renderer.render()
            writer.append_data(frame)
            
            if done or truncated:
                break

        writer.close()
        env.close()
        renderer.close()
        print(f"\n Saved video: debug_video.mp4")

    except mujoco.FatalError as e:
        print(f"Caught MuJoCo error: {e}")
        if 'missing an OpenGL platform library' in str(e):
            print("\nThis error often means you are on a headless server and need to install EGL.")
            print("On Debian/Ubuntu, try: sudo apt-get install libegl1-mesa-dev")
            print("On Red Hat/CentOS, try: sudo yum install mesa-libEGL-devel")

if __name__ == "__main__":
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please ensure it's in the same directory.")
        exit()
        
    run_and_debug_env(config)