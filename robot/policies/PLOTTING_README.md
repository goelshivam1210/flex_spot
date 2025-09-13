# Trajectory Plotting in Push/Drag Policy

This document describes the trajectory plotting functionality added to the `push_drag.py` policy.

## Overview

The push/drag policy now automatically tracks the robot's position in the vision frame during policy execution and generates plots comparing the actual robot trajectory with the planned path.

## Features

### 1. Position Tracking
- Tracks robot position (x, y, yaw) in vision frame at each policy step
- Stores trajectory data throughout the entire policy execution
- Captures both planned path and actual robot movement

### 2. Trajectory Plot
- **Blue line**: Planned path from start to end
- **Red line**: Actual robot trajectory
- **Green circle**: Start point
- **Red square**: End point
- **Orange arrows**: Robot orientation at key points
- **Statistics box**: Total distance, path length, final error, and step count

### 3. Position Over Time Plot
- Three subplots showing x, y, and yaw over time steps
- Useful for analyzing robot behavior and control performance

## Usage

### Basic Usage
```bash
python push_drag.py --hostname 192.168.1.100 --experiment small_box_no_handle_push
```

### Disable Plotting
```bash
python push_drag.py --hostname 192.168.1.100 --experiment small_box_no_handle_push --no-plots
```

### Example with Custom Parameters
```bash
python push_drag.py \
    --hostname 192.168.1.100 \
    --experiment small_box_no_handle_push \
    --target-distance 1.5 \
    --max-steps 20 \
    --action-scale 0.1
```

## Output Files

Plots are automatically saved to the `plots/` directory with timestamps:

- `trajectory_{experiment}_{timestamp}.png` - Trajectory comparison plot
- `position_over_time_{experiment}_{timestamp}.png` - Position over time plot

Example filenames:
- `trajectory_small_box_no_handle_push_20241201_143022.png`
- `position_over_time_small_box_no_handle_push_20241201_143022.png`

## Plot Interpretation

### Trajectory Plot
- **Good performance**: Red line closely follows blue line
- **Overshoot**: Red line goes beyond the blue end point
- **Undershoot**: Red line stops short of the blue end point
- **Oscillation**: Red line shows back-and-forth movement
- **Final error**: Distance between final robot position and target

### Position Over Time Plot
- **Smooth curves**: Good control performance
- **Oscillations**: Possible control issues or high action scaling
- **Sudden jumps**: Potential policy or control problems

## Technical Details

### Coordinate System
- All positions are tracked in the **vision frame** (world frame)
- X-axis: Forward direction from robot's initial pose
- Y-axis: Left direction from robot's initial pose
- Yaw: Rotation around Z-axis (positive = counterclockwise)

### Path Generation
- Arc paths are generated in local coordinate system (centered at origin)
- Path points are transformed to vision frame accounting for robot orientation
- Transformation includes both rotation (by robot's yaw) and translation (to robot position)
- This ensures paths are correctly oriented regardless of robot's starting pose

### Data Collection
- Position data is collected at each policy step
- Uses `spot.get_current_pose()` for robot position
- Uses `get_a_tform_b()` for hand position in vision frame

### Plotting Libraries
- **matplotlib**: For creating plots
- **numpy**: For data manipulation and calculations

## Troubleshooting

### Common Issues

1. **No plots generated**: Check if `--no-plots` flag was used
2. **Empty plots**: Ensure policy executed for at least one step
3. **Import errors**: Install matplotlib: `pip install matplotlib`

### Debugging
- Check console output for plot generation messages
- Verify `plots/` directory is created
- Look for error messages in the policy execution logs

## Example Analysis

A typical successful push operation should show:
- Robot trajectory closely following the planned arc path
- Smooth position curves over time
- Final error < 0.2m (configurable via `--success-distance`)
- No excessive oscillations in position plots

## Integration with Existing Code

The plotting functionality is integrated into the existing `execute_path_following_policy()` function:

1. **Data collection**: Added position tracking in the main policy loop
2. **Plot generation**: Added after policy completion
3. **File management**: Automatic timestamping and directory creation
4. **User control**: Optional disable via command line flag

No changes to existing policy logic or robot control code were required.
