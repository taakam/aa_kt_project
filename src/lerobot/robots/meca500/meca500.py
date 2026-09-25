import logging
import time
from functools import cached_property
from typing import Any

import mecademicpy.robot as mdr
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..utils import ensure_safe_goal_position
from .config_meca500 import Meca500Config

logger = logging.getLogger(__name__)


class Meca500(Robot):
    config_class = Meca500Config
    name = "meca500"

    def __init__(self, config: Meca500Config):
        super().__init__(config)
        self.config = config
        self.robot = mdr.Robot()
        self.cameras = make_cameras_from_configs(config.cameras)
        self._connected = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        # Define what data your robot provides (key: type/shape)
        # Example: {"joint_1.pos": float, "camera_front": (480, 640, 3)}
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
            "joint_6.pos": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # 6 joint targets plus the latched precision-mode flag (see the SpaceMouse
        # teleop). `precision.state` is not a `.pos` key, so send_action ignores it
        # as a joint target while it is still recorded and predicted by the policy.
        return {**self._motors_ft, "precision.state": float}

    @property
    def is_connected(self) -> bool:
        # Robot-level connectivity should succeed even if one or more optional cameras
        # fail to initialize (e.g., missing wrist camera on this setup).
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Connecting to Meca500 at {self.config.ip_address}...")
        try:
            self.robot.Connect(
                address=self.config.ip_address,
                enable_synchronous_mode=False,
                monitor_mode=self.config.monitor_mode,
            )
            if not self.config.monitor_mode:
                logger.info("Homing robot...")
                self.robot.ActivateAndHome()
                self.robot.WaitHomed(timeout=60)
        except Exception as e:
            # The exception that surfaces here is a generic DisconnectError; the real
            # cause ("Another user is already controlling the robot") lives further down
            # the __cause__/__context__ chain, so walk it to find the actual message.
            chain = []
            cause = e
            seen = set()
            while cause is not None and id(cause) not in seen:
                seen.add(id(cause))
                chain.append(str(cause))
                cause = cause.__cause__ or cause.__context__
            if any("Another user is already controlling the robot" in msg for msg in chain):
                raise DeviceNotConnectedError(
                    f"Meca500 at {self.config.ip_address} is already controlled by another session. "
                    f"Close the other connection (e.g. the web interface or a running script) and try again."
                ) from None
            raise DeviceNotConnectedError(f"Failed to connect to Meca500 at {self.config.ip_address}: {e}")

        failed_cams: list[str] = []
        for cam_name, cam in self.cameras.items():
            try:
                cam.connect()
                logger.info(f"Camera '{cam_name}' connected successfully.")
            except ConnectionError as e:
                logger.warning(
                    f"Failed to connect camera '{cam_name}' (configured as {cam}): {e}. "
                    "This camera will be skipped. Run 'lerobot-find-cameras opencv' to inspect available cameras."
                )
                failed_cams.append(cam_name)

        self._connected = True

        if failed_cams:
            logger.warning(f"Meca500 connected with {len(failed_cams)} unavailable camera(s): {failed_cams}")

        logger.info(f"{self} connected and homed.")

    @property
    def is_calibrated(self) -> bool:
        if not self.is_connected:
            return False
        return True  # If its homed, its calibrated

    def calibrate(self) -> None:
        # For Meca500, calibration is "Homing"
        if self.is_connected:
            logger.info("Homing robot...")
            self.robot.ActivateAndHome()
            self.robot.WaitHomed(timeout=60)

    def configure(self) -> None:
        self.robot.SetBlending(100)

        if self.config.default_joint_vel:
            self.robot.SetJointVel(self.config.default_joint_vel)

        self.robot.SetRealTimeMonitoring("all")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        try:
            joints = self.robot.GetJoints()
        except Exception as e:
            logger.error(f"Failed to read robot state: {e}")
            # Fallback or re-raise depending on strictness required
            raise e

        obs_dict = {
            "joint_1.pos": joints[0],
            "joint_2.pos": joints[1],
            "joint_3.pos": joints[2],
            "joint_4.pos": joints[3],
            "joint_5.pos": joints[4],
            "joint_6.pos": joints[5],
        }

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Read cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()

            if cam.is_connected:
                obs_dict[cam_key] = cam.async_read()
            else:
                logger.warning(
                    f"Camera '{cam_key}' is not connected. Returning a blank frame for observation feature consistency."
                )
                blank_height = int(cam.height or getattr(cam, "capture_height", None) or 1)
                blank_width = int(cam.width or getattr(cam, "capture_width", None) or 1)
                obs_dict[cam_key] = np.zeros((blank_height, blank_width, 3), dtype=np.uint8)

            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.monitor_mode:
            return action

        # Parse actions
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # When the policy predicts precision mode, clamp per-step motion harder so the
        # arm scales down its own movements under the microscope. Falls back to the
        # normal clamp when the gate is inactive or unconfigured.
        precision_active = float(action.get("precision.state", 0.0)) > 0.5
        relative_target = self.config.max_relative_target
        if precision_active and self.config.precision_max_relative_target is not None:
            relative_target = self.config.precision_max_relative_target

        if relative_target is not None:
            present_pos_list = self.robot.GetRtTargetJointPos()
            present_pos = {f"joint_{i + 1}": p for i, p in enumerate(present_pos_list)}
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, relative_target)

        logger.debug(f"Sending goal positions: {goal_pos}")
        self.robot.MoveJoints(
            float(goal_pos["joint_1"]),
            float(goal_pos["joint_2"]),
            float(goal_pos["joint_3"]),
            float(goal_pos["joint_4"]),
            float(goal_pos["joint_5"]),
            float(goal_pos["joint_6"]),
        )
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Disconnecting Meca500...")
        try:
            if self.robot.IsConnected():
                # In monitor_mode the wrapper holds an observation-only socket;
                # control commands (PauseMotion/ClearMotion/DeactivateRobot)
                # would raise and trigger mecademicpy's disconnect_on_exception.
                # The teleop owns the control connection and has already
                # deactivated the arm by the time we get here.
                if not self.config.monitor_mode:
                    self.robot.PauseMotion()
                    self.robot.WaitMotionPaused(timeout=5)
                    self.robot.ClearMotion()
                    self.robot.WaitMotionCleared(timeout=5)
                    self.robot.DeactivateRobot()
                    self.robot.WaitDeactivated(timeout=10)
                self.robot.Disconnect()
                self.robot.WaitDisconnected(timeout=5)
        except Exception as e:
            logger.warning(f"Error during disconnect sequence: {e}")

        for cam in self.cameras.values():
            cam.disconnect()

        self._connected = False
        logger.info(f"{self} disconnected.")
