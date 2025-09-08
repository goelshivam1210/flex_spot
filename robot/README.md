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
│   ├── path_following_td3.py # Path-following TD3 variant
│   ├── policy_manager.py   # Policy loading
│   └── interactive_perception.py  # Joint analysis & path state construction
├── policies/                # Task applications
│   ├── button_push.py      # Button/switch manipulation
│   ├── door_open.py        # Door opening (revolute/prismatic)
│   └── push_drag.py        # Push/drag manipulation with path-following

├── models/                  # Trained neural network policies
│   ├── prismatic/
│   ├── revolute/
│   └── rotation/            # Path-following policies
└── sim/                     # PyBullet and Mujoco simulation environment
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

# Push/drag manipulation with autonomous detection
python policies/push_drag.py --hostname <ROBOT_IP> --experiment small_box_no_handle_1robot --autonomous-detection

# Push/drag with manual target selection
python policies/push_drag.py --hostname <ROBOT_IP> --experiment small_box_handle_1robot --target-distance 1.5
```

## Available Tasks

| Task | Command | Description |
|------|---------|-------------|
| **Button Push** | `python policies/button_push.py` | Force-controlled button/switch pressing |
| **Door Opening** | `python policies/door_open.py` | Door manipulation with joint detection |
| **Push/Drag** | `python policies/push_drag.py` | Box pushing/dragging with path-following policies |

### Push/Drag Task Configurations

The push/drag system supports four experiment types:

| Experiment | Task Type | Description |
|------------|-----------|-------------|
| `small_box_no_handle_1robot` | Push | Single robot pushes small box |
| `large_box_no_handle_2robots` | Push | Coordinated two-robot box pushing |
| `small_box_handle_1robot` | Drag | Single robot drags box by handle |
| `large_box_handle_2robots` | Drag | Coordinated two-robot box dragging |

### Common Parameters
- `--hostname <IP>`: Spot robot IP address
- `--dock-id <ID>`: Docking station ID (default: 521)
- `--max-steps <N>`: Maximum policy execution steps
- `--action-scale <FLOAT>`: Scale factor for policy actions

### Push/Drag Specific Parameters
- `--experiment <TYPE>`: Experiment configuration (required)
- `--autonomous-detection`: Use OWL-v2 + SAM for automatic box detection
- `--robot-side <left|right>`: Robot side for multi-robot coordination
- `--target-distance <FLOAT>`: Distance to push/drag (default: 1.0m)
- `--policy-path <PATH>`: Path to trained path-following models
- `--model-name <NAME>`: Model name to load (default: best_model)

## Example Commands

### Door Opening
```bash
# Prismatic door (sliding)
python policies/door_open.py --hostname 192.168.1.101 --force-joint-type prismatic --max-steps 40 --action-scale 1.0 --success-distance 0.35

# Revolute door (hinged) - auto-detect joint type
python policies/door_open.py --hostname 192.168.1.101 --max-steps 10 --action-scale 0.1
```

### Button Pushing
```bash
python policies/push_button.py --hostname 192.168.1.100 --approach-distance 0.9 --press-force-percentage 0.2
```

### Push/Drag Manipulation
```bash
# Single robot push with autonomous detection
python policies/push_drag.py --hostname 192.168.1.100 --experiment small_box_no_handle_1robot --autonomous-detection --target-distance 1.2

# Multi-robot coordination (left robot)
python policies/push_drag.py --hostname 192.168.1.100 --experiment large_box_no_handle_2robots --autonomous-detection --robot-side left

# Handle dragging with manual selection
python policies/push_drag.py --hostname 192.168.1.101 --experiment small_box_handle_1robot --target-distance 0.8
```

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
- For push/drag tasks, ensure path-following models are in `models/rotation/` directory

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
