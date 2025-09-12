"""
two_spot_push.py

Dual robot coordinated push/drag manipulation with Spot robots using path-following policy

The system works in 4 phases:
1. Sequential target collection from both robots
2. Coordinated object grasping 
3. Synchronized path-following policy execution with force decomposition
4. Release and cleanup

Example usage:
    python two_spot_push.py --robot1-hostname 192.168.1.100 --robot2-hostname 192.168.1.101 --experiment large_box_no_handle_2robots

Author: Shivam Goel
Date: September 2025
"""

import argparse
import sys
import time
import numpy as np
import math
import threading
from concurrent.futures import ThreadPoolExecutor

# Import Spot SDK modules
from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
from bosdyn.client.math_helpers import SE3Pose

# Import existing Spot modules
from spot.spot import Spot, SpotPerception
from flex.interactive_perception import InteractivePerception
from flex.policy_manager import PolicyManager

# Dual robot experiment configurations
DUAL_EXPERIMENT_CONFIGS = {
    "large_box_no_handle_2robots_push": {
        "description": "Large box without handle, dual robot push task",
        "grasp_strategy": "dual_edge_grasp",
        "task_type": "push",
        "expected_distance": 1.0,
        "max_force_scale": 0.6,
        "num_robots": 2,
        "box_dimensions": {"width": 1.85, "depth": 0.5, "height": 0.8}
    },
    "large_box_no_handle_2robots_drag": {
        "description": "Large box without handle, dual robot drag task", 
        "grasp_strategy": "dual_edge_grasp",
        "task_type": "drag",
        "expected_distance": 1.0,
        "max_force_scale": 0.6,
        "num_robots": 2,
        "box_dimensions": {"width": 1.85, "depth": 0.5, "height": 0.8}
    }
    # ,
    # "large_box_handle_2robots_push": {
    #     "description": "Large box with handles, dual robot push task",
    #     "grasp_strategy": "dual_handle_grasp",
    #     "task_type": "push", 
    #     "expected_distance": 1.0,
    #     "max_force_scale": 0.5,
    #     "num_robots": 2,
    #     "box_dimensions": {"width": 0.6, "depth": 0.4, "height": 0.3}
    # }
}


class DualRobotCoordinator:
    """Coordinates two Spot robots for synchronized manipulation tasks."""
    
    def __init__(self, config, experiment_config):
        self.config = config
        self.experiment_config = experiment_config
        self.robots = {}
        self.robot_configs = []
        self.interactive_perception = InteractivePerception()
        self.policy_manager = PolicyManager()
        self.policy = None
        
        # Synchronization barriers
        self.grasp_barrier = None
        self.policy_step_barrier = None
        self.release_barrier = None
        
    def user_confirm_step(self, step_description):
        """Ask user to confirm before executing each step."""
        print(f"\n{'='*60}")
        print(f"READY FOR: {step_description}")
        print("Press ENTER to continue, 's' + ENTER to skip, or 'q' + ENTER to quit safely")
        print('='*60)
        
        user_input = input().strip().lower()
        
        if user_input == 'q':
            print("User requested safe shutdown...")
            return False
        elif user_input == 's':
            print("User skipped this step...")
            return 'skip'
        
        return True

    def collect_target_for_robot(self, robot_id, hostname, username=None, password=None):
        """Connect to robot and collect user target selection."""
        print(f"\n=== Setting up target for {robot_id} ===")
        
        try:
            # Create temporary robot connection for image capture
            spot = Spot(id=robot_id, hostname=hostname)
            spot.start()
            
            if username and password:
                # Handle authentication if needed
                pass
                
        except Exception as e:
            print(f'Robot {robot_id}: Failed to connect - {e}')
            return None
        
        try:
            # Take picture for target selection
            print(f'Robot {robot_id}: Taking picture with camera: {self.config.image_source}')
            
            color_img, depth_img = spot.take_picture(
                color_src=self.config.image_source,
                depth_src=self.config.depth_source,
                save_images=True
            )
            
            if color_img is None:
                raise Exception('Failed to capture image')
            
            # Get user target selection
            if robot_id == "Robot1":
                print(f'Robot {robot_id}: Click on the LEFT side of the box to grasp...')
            else:
                print(f'Robot {robot_id}: Click on the RIGHT side of the box to grasp...')
            
            target_pixel = SpotPerception.get_target_from_user(color_img)
            
            if target_pixel is None:
                raise Exception('No target selected')
            
            print(f'Robot {robot_id}: Target selected at pixel: {target_pixel}')
            
            # Store robot configuration
            robot_config = {
                'robot_id': robot_id,
                'hostname': hostname,
                'username': username,
                'password': password,
                'target_pixel': target_pixel,
                'side': 'left' if robot_id == "Robot1" else 'right'
            }
            
            return robot_config
            
        except Exception as e:
            print(f'Robot {robot_id}: Error during target collection - {e}')
            return None
        finally:
            # Clean up temporary connection
            try:
                spot.shutdown()
            except:
                pass

    def execute_coordinated_manipulation(self, robot_config, barriers):
        """Execute coordinated manipulation for a single robot."""
        robot_id = robot_config['robot_id']
        hostname = robot_config['hostname']
        target_pixel = robot_config['target_pixel']
        robot_side = robot_config['side']
        
        grasp_barrier, policy_barrier, release_barrier = barriers
        
        try:
            # Initialize robot
            spot = Spot(id=robot_id, hostname=hostname)
            spot.start()
            
            with LeaseKeepAlive(spot.lease_client, must_acquire=True, return_at_exit=True):
                # Power on and stand
                print(f'{robot_id}: Powering on and standing...')
                spot.power_on()
                spot.stand_up()
                spot.open_gripper()
                
                # Phase 1: Grasp object
                print(f'{robot_id}: Grasping object at target pixel {target_pixel}')
                spot.open_gripper()
                success = spot.grasp_edge(target_pixel, img_src=self.config.image_source)
                
                if not success or not spot.check_grip():
                    raise Exception(f'{robot_id}: Failed to grasp object')
                
                print(f'{robot_id}: Object grasped successfully')
                
                # Wait for both robots to complete grasping
                print(f'{robot_id}: Waiting for other robot to grasp...')
                grasp_barrier.wait()
                print(f'{robot_id}: Both robots have grasped the object!')
                
                # Phase 2: Define path (only Robot1 defines, Robot2 follows)
                if robot_id == "Robot1":
                    path_info = self.define_coordinated_path(spot)
                    self.shared_path_info = path_info
                else:
                    # Robot2 waits for path definition
                    time.sleep(2)
                    path_info = self.shared_path_info
                
                # Phase 3: Execute coordinated policy
                self.execute_dual_robot_policy(spot, robot_config, path_info, policy_barrier)
                
                # Phase 4: Release and cleanup
                print(f'{robot_id}: Waiting to release object...')
                release_barrier.wait()
                
                print(f'{robot_id}: Releasing object and cleaning up...')
                spot.open_gripper()
                time.sleep(1)
                spot.stow_arm()
                
                # Dock robot
                try:
                    spot.dock(dock_id=self.config.dock_id)
                    print(f'{robot_id}: Docked successfully')
                except Exception as e:
                    print(f'{robot_id}: Could not dock: {e}')
                    
                return True
                
        except Exception as e:
            print(f'{robot_id}: Error during execution - {e}')
            return False

    def define_coordinated_path(self, lead_spot):
        """Define path for coordinated manipulation (called by Robot1)."""
        print('Defining coordinated manipulation path...')
        
        # Get current hand position as start point
        robot_state = lead_spot._client._state_client.get_robot_state()
        snapshot = robot_state.kinematic_state.transforms_snapshot
        vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
        
        if vision_T_hand is None:
            raise Exception("Could not get hand pose for path definition")
        
        start_position = np.array([vision_T_hand.x, vision_T_hand.y, vision_T_hand.z])
        current_x, current_y, current_yaw = lead_spot.get_current_pose()
        
        # Define arc parameters for coordinated manipulation
        arc_radius = self.config.target_distance
        
        if self.experiment_config["task_type"] == "push":
            # Push: forward arc curving right
            start_angle = -np.pi/6  # -30 degrees
            end_angle = np.pi/6     # +30 degrees
            arc_center = start_position[:2]
        else:  # drag
            # Drag: backward arc curving left  
            start_angle = np.pi - np.pi/6   # 150 degrees
            end_angle = np.pi + np.pi/6     # 210 degrees
            arc_center = start_position[:2]
        
        # Generate arc path points
        path_points_2d = self.interactive_perception.generate_arc_path(
            center=arc_center,
            radius=arc_radius,
            start_angle=start_angle,
            end_angle=end_angle,
            num_points=max(10, int(self.config.target_distance * 20))
        )
        
        # Add z-coordinate (keep constant height)
        path_points = np.column_stack([path_points_2d, np.full(len(path_points_2d), start_position[2])])
        
        end_position = path_points[-1]
        direction_vector = (end_position - start_position)[:2]
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        print(f'Coordinated path defined: {len(path_points)} points')
        print(f'Arc radius: {arc_radius:.2f}m, Task: {self.experiment_config["task_type"]}')
        
        return {
            'start_position': start_position,
            'end_position': end_position,
            'path_points': path_points,
            'direction_vector': direction_vector
        }

    def execute_dual_robot_policy(self, spot, robot_config, path_info, policy_barrier):
        """Execute path-following policy with dual robot coordination."""
        robot_id = robot_config['robot_id']
        robot_side = robot_config['side']
        
        print(f'{robot_id}: Starting coordinated policy execution')
        
        # Load policy (only once, shared between robots)
        if self.policy is None:
            try:
                self.policy = self.policy_manager.load_path_following_policy(
                    self.config.policy_path, self.config.model_name
                )
                print("Path-following policy loaded successfully")
            except FileNotFoundError as e:
                raise Exception(f"Policy loading failed: {e}")
        
        path_points = path_info['path_points']
        closest_path_idx = 0
        
        for step in range(self.config.max_steps):
            # Get current state
            robot_state = spot._client._state_client.get_robot_state()
            snapshot = robot_state.kinematic_state.transforms_snapshot
            current_hand_pose = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
            current_hand_pos = np.array([current_hand_pose.x, current_hand_pose.y, current_hand_pose.z])
            
            current_x, current_y, current_yaw = spot.get_current_pose()
            
            # Only Robot1 computes policy action and box state
            if robot_id == "Robot1":
                # Estimate box center from grasp
                box_center = self.interactive_perception.estimate_box_center_from_grasp(
                    current_hand_pos, 
                    self.experiment_config["grasp_strategy"], 
                    self.experiment_config["box_dimensions"], 
                    current_yaw
                )
                
                # Construct state vector for policy
                state_vector = self.interactive_perception.construct_path_following_state(
                    box_center, path_points, current_yaw, closest_path_idx
                )
                
                # Get action from policy
                action = self.policy.select_action(state_vector)
                if len(action.shape) > 1:
                    action = action.flatten()
                
                # Scale action to get box-level wrench
                box_wrench = action * self.config.action_scale * self.experiment_config["max_force_scale"]
                
                # Decompose wrench to contact forces for both robots
                contact_forces = self.interactive_perception.decompose_wrench_to_contact_forces(
                    box_wrench, self.experiment_config["box_dimensions"]
                )
                
                # Store for Robot2 to access
                self.shared_contact_forces = contact_forces
                self.shared_box_center = box_center
                
                print(f"Step {step+1}/{self.config.max_steps}")
                print(f"Box wrench: {box_wrench}")
                print(f"Contact forces: {contact_forces}")
                
            # Synchronize policy computation
            policy_barrier.wait()
            
            # Apply robot-specific contact force
            try:
                if robot_side == 'left':
                    robot_force = self.shared_contact_forces[0]  # Left robot force
                else:
                    robot_force = self.shared_contact_forces[1]  # Right robot force
                
                # Apply force via movement
                self.apply_contact_force_via_movement(spot, robot_force, robot_config)
                
                # Update path tracking (Robot1 only)
                if robot_id == "Robot1":
                    closest_path_idx = self.interactive_perception.update_closest_path_index(
                        self.shared_box_center, path_points, closest_path_idx
                    )
                    
                    # Check success
                    if self.check_path_following_success(self.shared_box_center, path_info):
                        print(f"Path following completed successfully in {step+1} steps!")
                        break
                        
            except Exception as e:
                print(f"{robot_id}: Error in step {step+1}: {e}")
                break
        
        print(f'{robot_id}: Policy execution completed')

    def apply_contact_force_via_movement(self, spot, contact_force, robot_config):
        """Apply contact force through robot movement commands."""
        robot_id = robot_config['robot_id']
        
        # Convert contact force to movement
        force_x, force_y, force_z = contact_force[0], contact_force[1], contact_force[2]
        
        if self.experiment_config["task_type"] == "push":
            # For pushing: move robot body to apply force
            print(f'{robot_id}: Applying push force: [{force_x:.3f}, {force_y:.3f}]')
            
            # Convert force to displacement
            dt = 1.5  # Time to apply force
            force_scale = 0.02  # Scale factor for force-to-displacement
            
            dx = force_x * force_scale
            dy = force_y * force_scale
            d_yaw = 0.0  # No rotation for individual robot
            
            vx = abs(dx/dt)
            vy = abs(dy/dt)
            v_yaw = 0.0
            
            spot.push_object(
                dx=dx, dy=dy, d_yaw=d_yaw,
                vx=vx, vy=vy, v_yaw=v_yaw,
                dt=dt
            )
            
        else:  # drag
            # For dragging: move arm to apply force
            print(f'{robot_id}: Applying drag force: [{force_x:.3f}, {force_y:.3f}]')
            
            # Get current hand pose
            robot_state = spot._client._state_client.get_robot_state()
            snapshot = robot_state.kinematic_state.transforms_snapshot
            current_hand_pose = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
            current_hand_pos = np.array([current_hand_pose.x, current_hand_pose.y, current_hand_pose.z])
            
            # Convert force to target position
            force_scale = 0.02
            target_position = current_hand_pos + np.array([force_x, force_y, force_z]) * force_scale
            
            target_pose = SE3Pose(
                x=target_position[0],
                y=target_position[1], 
                z=target_position[2],
                rot=current_hand_pose.rot
            )
            
            arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
                target_pose.to_proto(), VISION_FRAME_NAME, seconds=1.5
            )
            cmd_id = spot._client._command_client.robot_command(arm_cmd)
            block_until_arm_arrives(spot._client._command_client, cmd_id, timeout_sec=2.0)

    def check_path_following_success(self, current_pos, path_info):
        """Check if path following succeeded."""
        end_position = path_info['end_position']
        distance_to_goal = np.linalg.norm(current_pos - end_position)
        
        success = distance_to_goal <= self.config.success_distance
        print(f"Distance to goal: {distance_to_goal:.3f}m (threshold: {self.config.success_distance:.3f}m)")
        
        return success

    def cleanup_robots(self):
        """Emergency cleanup for all robots."""
        print('Performing emergency cleanup for all robots...')
        
        for robot_config in self.robot_configs:
            try:
                robot_id = robot_config['robot_id']
                hostname = robot_config['hostname']
                
                print(f'Cleaning up {robot_id}...')
                spot = Spot(id=robot_id, hostname=hostname)
                spot.start()
                
                spot.open_gripper()
                time.sleep(1)
                spot.stow_arm()
                spot.dock(dock_id=self.config.dock_id)
                
                print(f'{robot_id} cleaned up successfully')
                
            except Exception as e:
                print(f'Could not cleanup {robot_id}: {e}')

    def run_dual_robot_manipulation(self):
        """Main execution function for dual robot manipulation."""
        
        try:
            # Phase 1: Collect targets from both robots
            step_result = self.user_confirm_step("Collect targets from both robots")
            if step_result == False:
                return False
            elif step_result != 'skip':
                print("=== Collecting targets from both robots ===")
                
                # Robot1 target
                robot1_config = self.collect_target_for_robot(
                    'Robot1', self.config.robot1_hostname, 
                    self.config.robot1_username, self.config.robot1_password
                )
                if robot1_config is None:
                    print("Failed to collect target for Robot1")
                    return False
                self.robot_configs.append(robot1_config)
                
                # Robot2 target  
                robot2_config = self.collect_target_for_robot(
                    'Robot2', self.config.robot2_hostname,
                    self.config.robot2_username, self.config.robot2_password
                )
                if robot2_config is None:
                    print("Failed to collect target for Robot2")
                    return False
                self.robot_configs.append(robot2_config)
                
                # Show summary
                print("\n=== Target Summary ===")
                for config in self.robot_configs:
                    print(f"{config['robot_id']}: Target at {config['target_pixel']} ({config['side']} side)")
            
            # Phase 2: Execute coordinated manipulation
            step_result = self.user_confirm_step("Execute coordinated manipulation")
            if step_result == False:
                return self.cleanup_robots()
            elif step_result != 'skip':
                print("\n=== Starting coordinated manipulation ===")
                
                # Create synchronization barriers
                num_robots = 2
                grasp_barrier = threading.Barrier(num_robots)
                policy_barrier = threading.Barrier(num_robots)  
                release_barrier = threading.Barrier(num_robots)
                
                barriers = (grasp_barrier, policy_barrier, release_barrier)
                
                # Create and start robot threads
                with ThreadPoolExecutor(max_workers=num_robots) as executor:
                    futures = []
                    
                    for robot_config in self.robot_configs:
                        future = executor.submit(
                            self.execute_coordinated_manipulation,
                            robot_config,
                            barriers
                        )
                        futures.append(future)
                    
                    # Wait for completion
                    results = []
                    for future in futures:
                        try:
                            result = future.result(timeout=300)  # 5 minute timeout
                            results.append(result)
                        except Exception as e:
                            print(f"Robot execution failed: {e}")
                            results.append(False)
                    
                    # Check results
                    if all(results):
                        print("Dual robot manipulation completed successfully!")
                        return True
                    else:
                        print("Dual robot manipulation completed with issues")
                        return False
            
            return True
            
        except KeyboardInterrupt:
            print('Interrupted by user. Cleaning up...')
            return self.cleanup_robots()
        except Exception as e:
            print(f'Error occurred: {e}. Cleaning up...')
            return self.cleanup_robots()


def main():
    """Command line interface for dual robot push/drag manipulation."""
    parser = argparse.ArgumentParser(description='Dual robot coordinated push/drag manipulation')
    
    # Robot connection arguments
    parser.add_argument('--robot1-hostname', required=True, help='Hostname/IP for first robot')
    parser.add_argument('--robot2-hostname', required=True, help='Hostname/IP for second robot')
    parser.add_argument('--robot1-username', help='Username for first robot')
    parser.add_argument('--robot1-password', help='Password for first robot')
    parser.add_argument('--robot2-username', help='Username for second robot')
    parser.add_argument('--robot2-password', help='Password for second robot')
    
    # Experiment configuration
    parser.add_argument('--experiment', required=True, choices=list(DUAL_EXPERIMENT_CONFIGS.keys()),
                        help='Dual robot experiment configuration to run')
    
    # Camera and sensing
    parser.add_argument('--image-source', default='hand_color_image',
                        help='Camera source for color images')
    parser.add_argument('--depth-source', default='hand_depth_in_hand_color_frame', 
                        help='Camera source for depth images')
    
    # Task parameters
    parser.add_argument('--max-steps', type=int, default=20,
                        help='Maximum policy execution steps')
    parser.add_argument('--action-scale', type=float, default=0.03,
                        help='Scale factor for policy actions')
    parser.add_argument('--target-distance', type=float, default=1.0,
                        help='Target distance to push/drag (meters)')
    parser.add_argument('--success-distance', type=float, default=0.15,
                        help='Success threshold distance from target (meters)')
    
    # Policy and model
    parser.add_argument('--policy-path', default='models/rotation',
                        help='Path to trained path-following policy')
    parser.add_argument('--model-name', default='best_model',
                        help='Name of the model to load')
    
    # System parameters
    parser.add_argument('--dock-id', type=int, default=521,
                        help='Docking station ID')
    
    options = parser.parse_args()
    
    # Validate experiment configuration
    if options.experiment not in DUAL_EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {options.experiment}")
    
    experiment_config = DUAL_EXPERIMENT_CONFIGS[options.experiment]
    print(f"Starting dual robot experiment: {experiment_config['description']}")
    
    # Create coordinator and run
    coordinator = DualRobotCoordinator(options, experiment_config)
    
    try:
        success = coordinator.run_dual_robot_manipulation()
        return success
    except Exception as exc:
        print(f'Dual robot manipulation failed: {exc}')
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)