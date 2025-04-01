import numpy as np
import time
from bosdyn.client.robot import Robot
from bosdyn.client import create_standard_sdk
from bosdyn.client.frame_helpers import BODY_FRAME_NAME
from bosdyn.api.spot import robot_command_pb2
from bosdyn.client.manipulation import ManipulationApiClient
from bosdyn.client.manipulation.commands import arm_cartesian_command

class ForcePolicyController:
    def __init__(self, robot: Robot, f_max=50.0, lambda_scale=0.001):
        self.robot = robot
        self.f_max = f_max
        self.lambda_scale = lambda_scale
        self.arm = robot.ensure_client('arm-command')
        self.robot_state_client = robot.ensure_client('robot-state')
        self.manip_client = ManipulationApiClient.create_from_robot(robot)

    def get_end_effector_pose(self):
        state = self.robot_state_client.get_robot_state()
        return state.manipulator_state_tool_pose  # Pose in body frame

    def force_to_pose_delta(self, force_direction, force_scale):
        """ Convert 2D force direction to 3D pose delta (X, Y, Z) """
        f_norm = force_direction / np.linalg.norm(force_direction)
        f = f_norm * force_scale * self.f_max
        delta_pos = np.array([f[0], f[1], 0.0]) * self.lambda_scale  # planar motion
        return delta_pos

    def apply_force_action(self, force_direction, force_scale):
        delta = self.force_to_pose_delta(force_direction, force_scale)
        current_pose = self.get_end_effector_pose().position
        target_position = [
            current_pose.x + delta[0],
            current_pose.y + delta[1],
            current_pose.z + delta[2],
        ]

        # Send cartesian command to move EEF
        arm_cartesian_command(
            robot=self.robot,
            root_frame_name=BODY_FRAME_NAME,
            pos=target_position,
            ori=None,  # Maintain orientation
            duration=0.2,
            absolute=False
        )

        time.sleep(0.2)  # Let motion complete





        '''
        # Example usage:
        controller = ForcePolicyController(robot)
        force_dir = np.array([1.0, 0.5])   # ← push diagonally
        force_scale = 0.6
        controller.apply_force_action(force_dir, force_scale)
        '''