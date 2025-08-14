"""
Handles Spot's connection

Author: Tim
Date: June 26, 2025
"""

from bosdyn.client import create_standard_sdk, util
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient

class SpotClient:
    """
    Handles Spot's connection
    """
    def __init__(self, id, username, password, hostname):
        self.id = id
        self.username = username
        self.password = password
        self.hostname = hostname

        self._sdk = None
        self._spot = None

        self._command_client = None
        self._image_client = None
        self._lease_client = None
        self._manip_client = None
        self._state_client = None

    @property
    def spot_robot(self):
        return self._spot
    
    def start(self):
        if not self.connect():
            exit(1)

        try:
            self.setup_clients()
        except Exception as e:
            print(f"{self.id}: Failed to set up clients - {e}")

    def connect(self) -> bool:
        """
        Connects to the robot and authenticates.
        
        Returns:
            bool: Connection success.
        """
        try:
            self._sdk = create_standard_sdk(f'Controller_{self.id}')
            self._spot = self._sdk.create_robot(self.hostname)
            self._spot.authenticate(self.username, self.password)
            self._spot.time_sync.wait_for_sync()
            print(f"{self.id}: Connection successful.")
            return True
        except Exception as e:
            print(f"{self.id}: Failed to connect - {e}")
            return False
        
    def setup_clients(self) -> None:
        """
        Initialize lease and manipulation API clients on self.
        """
        self._command_client = self._spot.ensure_client(RobotCommandClient.default_service_name)
        self._image_client = self._spot.ensure_client(ImageClient.default_service_name)
        self._lease_client = self._spot.ensure_client(LeaseClient.default_service_name)
        self._lease_client.take('body')
        self._manip_client = self._spot.ensure_client(ManipulationApiClient.default_service_name)
        self._state_client = self._spot.ensure_client(RobotStateClient.default_service_name)
        print(f"{self.id}: Set up clients")

    def power_on(self, timeout_sec=20):
        """
        Powers on Spot
        """
        self._spot.power_on(timeout_sec=timeout_sec)
        assert self._spot.is_powered_on(), "Power on failed."
        print(f"{self.id}: Powered on")

    def print_behavior_faults(self):
        """
        Retrieve and print current behavior faults from Spot.
        """
        state = self._state_client.get_robot_state()
        faults = state.behavior_fault_state.faults
        if not faults:
            print(f"{self.id}: No behavior faults.")
        else:
            print(f"Robot {self.id}: Current behavior faults:")
            for fault in faults:
                print(f"  - {fault}")
