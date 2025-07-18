# FLEX-SPOT

A robotics framework for Boston Dynamics Spot robot manipulation tasks using force based policies. The system combines hardware abstraction, perception, and task-specific applications for autonomous articulated object manipulation using single and multiple robots. For questions please contact shivam.goel@tufts.edu or timothy.duggan@tufts.edu

## Repository Structure

```
robot/
├── spot/                    # Hardware abstraction layer
│   ├── spot.py             # Main robot control
│   ├── spot_client.py      # SDK interface
│   ├── spot_perception.py  # Computer vision
│   └── spot_camera.py      # Image processing
├── flex/                    # Learning layer
│   ├── alg.py              # TD3 neural networks
│   ├── policy_manager.py   # Policy loading
│   └── interactive_perception.py  # Joint analysis
├── policies/                # Task applications
│   ├── button_push.py      # Button/switch manipulation
│   ├── door_open.py        # Door opening (revolute/prismatic)

├── models/                  # Trained neural network policies
│   ├── prismatic/
│   └── revolute/
└── sim/                     # PyBullet simulation environment
```

## Quick Start

### Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_robot.txt

# Set authentication (required)
export BOSDYN_CLIENT_USERNAME=hrilab
export BOSDYN_CLIENT_PASSWORD=hrilabrulesspotphi
```

### Run E-Stop GUI
```bash
# Start E-Stop interface (required for safety)
python examples/estop_gui.py --ip <ROBOT_IP>
```

### Execute Tasks
```bash
# Run from robot/ directory
cd robot/

# Button pushing
python policies/button_push.py --hostname <ROBOT_IP> --press-force-percentage 0.05

# Door opening (intelligent joint detection)
python policies/door_open.py --hostname <ROBOT_IP> --max-steps 20 --action-scale 0.05
```

## Available Tasks

| Task | Command | Description |
|------|---------|-------------|
| **Button Push** | `python policies/button_push.py` | Force-controlled button/switch pressing |
| **Door Opening** | `python policies/door_open.py` | Door manipulation with joint detection |
<!-- | **Box Pulling** | `python policies/pull_box.py` | Pull manipulation tasks |
| **Box Pushing** | `python policies/push_box.py` | Push manipulation tasks |
| **Multi-Robot** | `python policies/multi_robot_push.py` | Coordinated two-robot manipulation | -->

### Common Parameters
- `--hostname <IP>`: Spot robot IP address
- `--dock-id <ID>`: Docking station ID (default: 521)
- `--max-steps <N>`: Maximum policy execution steps
- `--action-scale <FLOAT>`: Scale factor for policy actions

<!-- ## Key Features

- **Modular Architecture**: Clean separation of hardware, intelligence, and applications
- **User Safety**: Step-by-step confirmations with emergency quit ('q') at any stage
- **Intelligent Manipulation**: Automatic joint type detection for doors (revolute/prismatic)
- **Force Control**: Precise force application for button pressing and manipulation
- **Multi-Robot Support**: Coordinated manipulation with multiple Spot robots -->



## Network Setup

### Option 1: Direct Connection
Connect to Spot's network and use the robot's IP address. We use HrilabMobile-5G

### Option 2: Tablet Control
1. Connect tablet to `spotphi` network (same credentials as above)
2. Use tablet for E-Stop and joystick control
3. Run policies from your development machine

## Notes

- Ensure Spot is estopped, authenticated, and on the same network.
- Trained policies must be accessible in this directory or passed via arguments.

You need to export the following environment variables to authenticate with the Boston Dynamics API:

```bash 
export BOSDYN_CLIENT_USERNAME=hrilab
export BOSDYN_CLIENT_PASSWORD=hrilabrulesspotphi
```
also make sure estop_gui is running on the robot.
The file can be found in the SPOTSDK under `examples/estop_gui.py`.
```bash
python estop_gui.py --ip <ROBOT_IP>
```



## Development

### Adding New Tasks
1. Create new file in `policies/` directory
2. Import required modules:
   ```python
   from spot.spot import Spot, SpotPerception
   from flex.policy_manager import PolicyManager  # If using trained policies
   ```
3. Follow the user confirmation pattern from existing policies
4. Implement cleanup using `spot.open_gripper()`, `spot.stow_arm()`, `spot.dock()`

### Directory Guidelines
- **spot/**: General robot capabilities and hardware interface
- **flex/**: Reusable algorithms and perception
- **policies/**: Task-specific logic with user controls
- **models/**: Trained neural network policies

## Requirements

- Python 3.8+
- Boston Dynamics Spot SDK
- PyTorch (for neural network policies)
- Network connection to Spot robot
- E-Stop GUI running for safety

## License

MIT License

```bash
python policies/door_open.py --hostname 192.168.1.101 --force-joint-type prismatic --max-steps 40 --action-scale 1.0 --success-distance 0.35
```
```bash
python policies/door_open.py --hostname 192.168.1.101 --max-steps 10 --action-scale 0.1
```
```bash
python policies/push_button.py --hostname 192.168.1.100 --approach-distance 0.9 --press-force-percentage 0.2
```
