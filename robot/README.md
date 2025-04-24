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


---

## License

MIT License
