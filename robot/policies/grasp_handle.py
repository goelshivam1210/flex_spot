"""
Open door with Spot robot using intelligent policy-based manipulation

The system works in 3 phases:
1. Phase 1: Walk to door, grasp handle, perform wiggle analysis
3. Phase 3: Release and cleanup

Example usage:
    python door_open.py --hostname 192.168.1.100 --max-steps 15 --action-scale 0.05

Author: Shivam Goel
Date: July 2025
"""

import argparse
import sys
import time
import numpy as np
import math

# Import Spot SDK modules
from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
from bosdyn.client.math_helpers import SE3Pose

# Import existing Spot modules
from spot.spot import Spot, SpotPerception
from flex.interactive_perception import InteractivePerception
from flex.policy_manager import PolicyManager


def user_confirm_step(step_description):
    """Ask user to confirm before executing each step. Return False if user wants to quit."""
    print(f"\n{'='*60}")
    print(f"READY FOR: {step_description}")
    print("Press ENTER to continue, or 'q' + ENTER to quit safely")
    print('='*60)
    
    user_input = input().strip().lower()
    
    if user_input == 'q':
        print("User requested safe shutdown...")
        return False
    
    return True


def wait_for_user_input(message="Press ENTER to continue, or 'q' + ENTER to quit"):
    """Wait for user input. Return False if user wants to quit."""
    print(f"\n{message}")
    user_input = input().strip().lower()
    
    if user_input == 'q':
        print("User requested to continue...")
        return False
    
    return True


def cleanup_and_exit(spot, dock_id=521):
    """Safely cleanup and exit the program with full shutdown."""
    try:
        print('Performing safe shutdown...')
        
        # Release handle and stow arm
        try:
            print('Releasing handle and stowing arm...')
            spot.open_gripper()
            time.sleep(1)
            spot.stow_arm()
            print('Arm stowed successfully')
        except Exception as e:
            print(f'Could not stow arm: {e}')
        
        # Dock robot
        try:
            print('Attempting to dock robot...')
            spot.dock(dock_id=dock_id)
            print('Robot docked successfully')
        except Exception as e:
            print(f'Could not dock robot: {e}')
        
        print('Robot safely shut down.')
        
    except Exception as e:
        print(f'Error during cleanup: {e}')


def walk_to_door_and_grasp(spot, config):
    """Walk to door location and grasp handle."""
    print('Walking to door and grasping handle')
    
    # Take picture for door/handle selection
    print(f'Taking picture with camera: {config.image_source}')
    
    color_img, depth_img = spot.take_picture(
        color_src=config.image_source,
        depth_src=config.depth_source,
        save_images=True
    )
    
    if color_img is None:
        raise Exception('Failed to capture image')
    
    # Get user to click on handle location
    print('Click on the door handle you want to grasp...')
    target_pixel = SpotPerception.get_target_from_user(color_img)
    
    if target_pixel is None:
        raise Exception('No target selected')
    
    print(f'Handle target selected at pixel: {target_pixel}')
    
    # Use existing grasp method
    spot.open_gripper()
    success = spot.grasp_edge(target_pixel, img_src=config.image_source)
    
    if not success:
        raise Exception('Failed to grasp handle')
    
    print('Handle grasped successfully')
    
    # Wait for user input instead of fixed time
    print('Robot is now holding the handle.')
    if not wait_for_user_input("Press ENTER to release handle and continue, or 'q' + ENTER to quit"):
        return False  # User wants to quit
    
    return target_pixel


def grasp_handle(config):
    """Main function to open door with Spot."""
    
    # Initialize robot
    spot = Spot(id="GraspHandle", hostname=config.hostname)
    spot.start()
    
    with LeaseKeepAlive(spot.lease_client, must_acquire=True, return_at_exit=True):
        try:
            # Initialize
            print('Initializing robot...')
            spot.power_on()
            spot.stand_up()
            spot.open_gripper()
            
            # Walk to door and grasp handle
            if not user_confirm_step("Walk to door and grasp handle"):
                return cleanup_and_exit(spot, config.dock_id)
            
            result = walk_to_door_and_grasp(spot, config)
            if result is False:  # User requested quit while holding handle
                return cleanup_and_exit(spot, config.dock_id)
            
        except KeyboardInterrupt:
            print('Interrupted by user. Cleaning up...')
            return cleanup_and_exit(spot, config.dock_id)
        except Exception as e:
            print(f'Error occurred: {e}. Cleaning up...')
            return cleanup_and_exit(spot, config.dock_id)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Open door with Spot robot')
    parser.add_argument('--hostname', required=True, help='Spot robot hostname or IP')
    parser.add_argument('--image-source', 
                        default='hand_color_image',
                        help='Camera source for color images')
    parser.add_argument('--depth-source',
                        default='hand_depth_in_hand_color_frame', 
                        help='Camera source for depth images')
    parser.add_argument('--dock-id',
                        type=int,
                        default=520,
                        help='Docking station ID')
    
    options = parser.parse_args()
    
    try:
        grasp_handle(options)
        return True
    except Exception as exc:
        print(f'Door opening failed: {exc}')
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)