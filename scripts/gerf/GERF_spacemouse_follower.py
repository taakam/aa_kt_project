"""Drive the SO-101 follower arm with a 3DConnexion SpaceMouse via inverse kinematics.

The SpaceMouse 6-DOF input is integrated into a target end-effector pose;
`RobotKinematics` (placo-based) solves IK each tick and the resulting joint
angles are sent to the arm.

Buttons:
  - Button 0 (left)  : lerp back to the joint pose the arm was in at startup
  - Button 1 (right) : toggle gripper open/close

Prerequisites (one-time):
  pip install placo
  Download the SO-101 URDF + meshes from
    https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
  and place them at ./SO101/ so the URDF lives at ./SO101/so101_new_calib.urdf

Edit the CONFIG block below, then `python GERF_spacemouse_follower.py`.
Ctrl-C to stop.
"""

import time
from pathlib import Path

import numpy as np
import pyspacemouse

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.rotation import Rotation

# ----------------------------- CONFIG -----------------------------
PORT = "COM3"
ROBOT_ID = "my_awesome_follower_arm"

URDF_PATH = "./SO101/so101_new_calib.urdf"
EE_FRAME = "gripper_frame_link"

LOOP_HZ = 30.0

# SpaceMouse → end-effector velocities at full deflection.
LINEAR_GAIN = 0.15     # metres per second
ANGULAR_GAIN = 1.5     # radians per second

# Per-axis sign flips to align the 3DConnexion device frame with the URDF base
# frame. Order: [x, y, z, roll, pitch, yaw]. Tune by jogging each axis solo.
AXIS_SIGNS = [+1.0, +1.0, +1.0, +1.0, +1.0, +1.0]

DEADZONE = 0.10
ALPHA = 0.3            # low-pass filter on the 6-D input (higher = less smoothing)

# Per-motor max position change per send_action call. Safety net against IK
# glitches snapping the arm. Body motors are in degrees, gripper in 0..100.
MAX_RELATIVE_TARGET = 5.0

# Buttons.
HOME_BUTTON_IDX = 0
GRIPPER_BUTTON_IDX = 1
GRIPPER_OPEN = 100.0
GRIPPER_CLOSE = 0.0

# Homing: per-tick step size when lerping joints back toward the start pose.
HOMING_STEP_DEG = 2.0
HOMING_TOL_DEG = 1.0
# ------------------------------------------------------------------


def read_twist(state, signs: np.ndarray) -> np.ndarray:
    if state is None:
        return np.zeros(6)
    raw = np.array(
        [state.x, state.y, state.z, state.roll, state.pitch, state.yaw],
        dtype=float,
    )
    return raw * signs


def main() -> None:
    if not Path(URDF_PATH).exists():
        raise FileNotFoundError(
            f"URDF not found at {URDF_PATH}. Download so101_new_calib.urdf from "
            "https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101"
        )

    robot = SO101Follower(
        SO101FollowerConfig(
            port=PORT,
            id=ROBOT_ID,
            use_degrees=True,  # RobotKinematics expects joints in degrees
            max_relative_target=MAX_RELATIVE_TARGET,
        )
    )
    robot.connect()

    motor_names = list(robot.bus.motors.keys())
    gripper_idx = motor_names.index("gripper")

    print(f"Loading IK solver from {URDF_PATH}...")
    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=EE_FRAME,
        joint_names=motor_names,
    )

    print("Opening SpaceMouse...")
    sm = pyspacemouse.open()
    if sm is None:
        robot.disconnect()
        raise RuntimeError("pyspacemouse.open() returned None — device not found?")

    obs = robot.get_observation()
    home_joints = np.array([obs[f"{m}.pos"] for m in motor_names], dtype=float)
    target_pose = kin.forward_kinematics(home_joints).copy()
    gripper_target = float(home_joints[gripper_idx])

    twist_filter = np.zeros(6)
    signs = np.array(AXIS_SIGNS, dtype=float)
    prev_buttons = [0, 0]
    homing = False
    dt = 1.0 / LOOP_HZ

    print("Running. Ctrl-C to stop.")
    print(f"  Home button: index {HOME_BUTTON_IDX}   Gripper toggle: index {GRIPPER_BUTTON_IDX}")
    try:
        while True:
            t0 = time.perf_counter()

            state = sm.read()
            buttons = list(getattr(state, "buttons", [0, 0])) if state is not None else [0, 0]
            while len(buttons) < 2:
                buttons.append(0)

            home_edge = buttons[HOME_BUTTON_IDX] and not prev_buttons[HOME_BUTTON_IDX]
            gripper_edge = buttons[GRIPPER_BUTTON_IDX] and not prev_buttons[GRIPPER_BUTTON_IDX]
            prev_buttons = buttons

            if home_edge:
                print("Returning to home pose...")
                homing = True

            if gripper_edge:
                gripper_target = GRIPPER_CLOSE if gripper_target > 50.0 else GRIPPER_OPEN

            obs = robot.get_observation()
            current_joints = np.array([obs[f"{m}.pos"] for m in motor_names], dtype=float)

            if homing:
                err = home_joints - current_joints
                step = np.clip(err, -HOMING_STEP_DEG, HOMING_STEP_DEG)
                q_target = current_joints + step
                if np.max(np.abs(err)) < HOMING_TOL_DEG:
                    homing = False
                    target_pose = kin.forward_kinematics(home_joints).copy()
                    gripper_target = float(home_joints[gripper_idx])
                    twist_filter = np.zeros(6)
                    print("Home reached.")
            else:
                raw = read_twist(state, signs)
                twist_filter = (1 - ALPHA) * twist_filter + ALPHA * raw
                twist = np.where(np.abs(twist_filter) < DEADZONE, 0.0, twist_filter)

                target_pose[:3, 3] += twist[:3] * LINEAR_GAIN * dt
                drot = Rotation.from_rotvec(twist[3:] * ANGULAR_GAIN * dt).as_matrix()
                target_pose[:3, :3] = drot @ target_pose[:3, :3]

                q_target = kin.inverse_kinematics(current_joints, target_pose)

            q_target[gripper_idx] = gripper_target

            action = {f"{m}.pos": float(q_target[i]) for i, m in enumerate(motor_names)}
            robot.send_action(action)

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        try:
            sm.close()
        except Exception:
            pass
        robot.disconnect()


if __name__ == "__main__":
    main()
