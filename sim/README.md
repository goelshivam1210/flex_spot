# Simulation Environment for Force-based Policy Learning

This module trains force-based policies in a PyBullet simulation.

---

## Environments

### Prismatic Object Manipulation

```text
Gym environment for prismatic object manipulation in 2D using PyBullet.

Goal:
    Move a box to a goal location using force applied at its centroid.
    The box behaves like a prismatic joint along a 2D axis.

State:
    [joint_axis (2D), displacement (2D)]

Action:
    [force_direction (2D), force_scale ∈ [0,1]]

Reward:
    Dense reward proportional to movement along the goal direction.
    Bonus for reaching the goal.
```

### Revolute Environment

> *Coming soon *

---

## Training

Use the TD3 algorithm (actor-critic) to learn the policy.

```bash
python train_new.py
```

This uses:
- `td3.py`: Dual-headed actor-critic TD3 implementation
- `runs/`: Directory where policies and logs are saved

---

## Requirements

- Python 3.8+
- PyBullet
- NumPy, Gym, etc. (add pip req if needed)

## Installation and running

- Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Run the training script:

```bash
python train_new.py
```
You will see the training process in the terminal. The trained policy will be saved in the `runs/` directory.
Seed 2 | Episode 5313: Reward = 9.82 ...

- Run the simulation:

```bash
python env.py
```
You will see the sim window pop up -- we have an implementation of the prismatic object manipulation task in 2D. There is a keyboard interface to control the object. The goal is to move the box to a target location using force applied at its centroid.


#### Have fun!
