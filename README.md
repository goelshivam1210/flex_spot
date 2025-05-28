# FLEX-SPOT

This repository houses the code for the flex impleemntation on the BD SPOT Robot.
Please contact Shivam Goel (shivam.goel@tufts.edu) for any queries.

- **sim/**: Simulated training environments for force policy learning using PyBullet.
- **robot/**: Execution interface to deploy trained policies on Spot using its SDK.

---

## Project Structure

```
flex_spot/
├── sim/             # PyBullet environments and TD3 training
├── robot/           # Spot robot policy execution interface
```

---

## Getting Started

### 1. Train in Simulation

```bash
cd sim
python train_new.py
```

### 2. Deploy on Spot

```bash
cd robot
python control_robot.py --ip <ROBOT_IP>
```

---

## Requirements

- Python 3.8+
- Additional dependencies listed in each directory's `requirements` file

---

## License

MIT License
