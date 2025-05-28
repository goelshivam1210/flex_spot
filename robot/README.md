# FLEX-SPOT

- **sim/**: Simulated training environments for force policy learning using PyBullet.
- **robot/**: Execution interface to deploy trained policies on Spot using its SDK.

---

## Requirements

- Python 3.8+
# Spot Deployment Interface

This directory contains code to run learned force-based policies on the Boston Dynamics Spot robot.

---

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirement_robot.txt
```

2. Run the policy controller:
```bash
python control_robot.py --ip <ROBOT_IP>
```

Replace `<ROBOT_IP>` with Spot's IP (e.g., `192.168.80.3`).

---

## Files

- `control_robot.py`: Runs Spot using the trained policy.
- `requirement_robot.txt`: Python dependencies.

---

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
---

## License

MIT License
