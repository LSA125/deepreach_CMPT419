# Goal 2 Runbook (RViz Safety Bubble + Stress Test)

This runbook launches the complete Goal 2 stack:
- DeepReach inference safety filter
- MarkerArray safety bubble visualization (`V(x)=0` approximation)
- Stress-test node (baseline vs filtered)
- RViz2 with preset configuration

## 1) Build

From workspace root:

```bash
cd /workspaces/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select deepreach_ros
source install/setup.bash
```

## 2) Launch (Top-down preset)

```bash
ros2 launch deepreach_ros deepreach_goal2.launch.py rviz_preset:=topdown
```

## 3) Launch (Orbit preset)

```bash
ros2 launch deepreach_ros deepreach_goal2.launch.py rviz_preset:=orbit
```

## 4) Expected Topics

Core safety outputs:
- `/safe_omega` (`std_msgs/msg/Float32`)
- `/safety_value` (`std_msgs/msg/Float32`)
- `/override_active` (`std_msgs/msg/Bool`)

Goal 2 outputs:
- `/safety_bubble_markers` (`visualization_msgs/msg/MarkerArray`)
- `/stress_test_passed` (`std_msgs/msg/Bool`)

Inputs:
- `/state` (`geometry_msgs/msg/Pose2D`)
- `/nominal_omega` (`std_msgs/msg/Float32`)

## 5) Quick Verification

In a second terminal:

```bash
cd /workspaces/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic echo /stress_test_passed --once
```

Expected:
- `data: true`

Optional checks:

```bash
ros2 topic echo /safe_omega --once
ros2 topic echo /safety_value --once
ros2 topic echo /override_active --once
ros2 topic echo /safety_bubble_markers --once
```

## 6) Notes

- RViz may print a runtime-dir permission warning in the dev container; this is non-fatal if displays and topics update normally.
- Stress test node starts with a short launch delay so inference/model initialization can complete before evaluation.
