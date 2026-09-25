import json
import logging
import threading
import time
from typing import Any, Dict

import bota_driver
import mecademicpy.robot as mdr
import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_meca500_bota import Meca500BotaConfig

logger = logging.getLogger(__name__)


class Meca500Bota(Teleoperator):
    config_class = Meca500BotaConfig
    name = "meca500_bota"

    def __init__(self, config: Meca500BotaConfig):
        super().__init__(config)
        self.config = config
        self.sensor: bota_driver.BotaDriver | None = None
        self.robot = mdr.Robot()

        self._running = False
        self._thread = None
        self._connected = False
        # Set while a one-shot homing move is in progress; the guidance loop
        # parks itself so its 500 Hz MoveLinVelTrf stream doesn't fight the
        # MoveJoints command queued from go_home().
        self._homing = threading.Event()

        self.wrench_filter = np.zeros(6)


    @property
    def action_features(self) -> dict[str, type]:
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
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def read_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        try:
            logger.info(f"Connecting to Bota Sensor")
            config_content = self.read_json(self.config.json_path)
            self.sensor = bota_driver.BotaDriver(self.config.json_path)
            # Transition driver from UNCONFIGURED to INACTIVE state
            if not self.sensor.configure():
                raise RuntimeError("Failed to configure driver")
            # Tare the sensor
            if not self.sensor.tare():
                raise RuntimeError("Failed to tare sensor")
            # Transition driver from INACTIVE to ACTIVE state
            if not self.sensor.activate():
                raise RuntimeError("Failed to activate driver")
            
        except Exception as e:
            raise DeviceNotConnectedError(f"Failed to connect to Bota Sensor at {self.config.sensor_port}: {e}")

        try:
            logger.info(f"Connecting to Meca500 (Control) at {self.config.meca_address}...")
            self.robot.Connect(address=self.config.meca_address)
            self.robot.ActivateAndHome()
            self.robot.WaitHomed()
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
                    f"Meca500 at {self.config.meca_address} is already controlled by another session. "
                    f"Close the other connection (e.g. the web interface or a running script) and try again."
                ) from None
            raise DeviceNotConnectedError(f"Failed to connect to Meca500 at {self.config.meca_address}: {e}")

        self._connected = True
        self._running = True
        self._thread = threading.Thread(target=self._guidance_loop, daemon=True)
        self._thread.start()
        logger.info("Hand Guidance Started.")

        logger.info(f"{self} connected.")

    def _guidance_loop(self):
        while self._running:
            if self._homing.is_set():
                time.sleep(0.01)
                continue
            # Read sensor
            frame_data = self.sensor.read_frame()
            if frame_data: # and frame_data.status == bota_driver.Status.VALID:
                wrench = np.array([
                    frame_data.force[0], frame_data.force[1], frame_data.force[2],
                    frame_data.torque[0], frame_data.torque[1], frame_data.torque[2]
                ])
                
                # Filter
                self.wrench_filter = (1 - self.config.alpha) * self.wrench_filter + self.config.alpha * wrench
                
                # Logic from your script (simplified for brevity)
                normF = np.linalg.norm(self.wrench_filter[:3])
                normM = np.linalg.norm(self.wrench_filter[3:])
                
                twist = np.zeros(6)
                
                # Translation
                if normF > self.config.f_threshold_high:
                     twist[:3] = self.config.gain_tr * self.wrench_filter[:3]
                
                # Rotation
                if normM > self.config.m_threshold_high:
                     twist[3:] = self.config.gain_rot * self.wrench_filter[3:]

                # Send Velocity to Robot
                # MoveLinVelTrf expects: x, y, z, wx, wy, wz
                self.robot.MoveLinVelTrf(
                    -twist[0], -twist[1], twist[2], 
                    -twist[3], -twist[4], twist[5]
                )
            
            time.sleep(0.002) # ~500Hz loop

    @property
    def is_calibrated(self) -> bool:
        if not self.is_connected:
            return False
        return True  # If its homed, its calibrated

    def calibrate(self) -> None:
        # For Meca500, calibration is "Homing"
        if self.is_connected:
            logger.info("Homing robot...")
            self.robot.Home()
            self.robot.WaitHomed(timeout=60)

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        joints = self.robot.GetJoints()
        return {
            "joint_1.pos": joints[0],
            "joint_2.pos": joints[1],
            "joint_3.pos": joints[2],
            "joint_4.pos": joints[3],
            "joint_5.pos": joints[4],
            "joint_6.pos": joints[5],
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    def go_home(self) -> None:
        """Drive the arm to ``config.home_joints`` over this teleop's control socket."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if len(self.config.home_joints) != 6:
            raise ValueError(
                f"home_joints must have 6 values, got {len(self.config.home_joints)}: {self.config.home_joints}"
            )

        home = [float(j) for j in self.config.home_joints]
        logger.info(f"Auto-homing Meca500 to {home}")
        self._homing.set()
        try:
            # Let the guidance loop park, then brake any in-flight velocity.
            time.sleep(0.02)
            self.robot.MoveLinVelTrf(0, 0, 0, 0, 0, 0)
            self.robot.MoveJoints(*home)
            self.robot.WaitIdle(timeout=self.config.home_timeout_s)
            logger.info("Home reached.")
        except Exception as e:
            logger.warning(f"Auto-home failed: {e}")
        finally:
            # Reset the low-pass filter so stale demo-time wrench doesn't
            # kick the arm the moment guidance resumes.
            self.wrench_filter = np.zeros(6)
            self._homing.clear()

    def disconnect(self) -> None:
        # 1. Stop the guidance thread first so no more MoveLinVelTrf commands
        #    are queued while we tear the connection down.
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        # 2. Bring the Meca500 to a safe state before dropping the TCP control
        #    link. Calling Disconnect() while the robot is active and has motion
        #    queued triggers the controller's "Control connection dropped"
        #    safety stop, which the user then has to clear from the portal.
        if self.robot.IsConnected():
            try:
                self.robot.PauseMotion()
                self.robot.WaitMotionPaused(timeout=5)
                self.robot.ClearMotion()
                self.robot.WaitMotionCleared(timeout=5)
                self.robot.DeactivateRobot()
                self.robot.WaitDeactivated(timeout=10)
            except Exception as e:
                logger.warning(f"Meca500 graceful stop failed, disconnecting anyway: {e}")
            try:
                self.robot.Disconnect()
                self.robot.WaitDisconnected(timeout=5)
                logger.info("Meca500 disconnected.")
            except Exception as e:
                logger.warning(f"Meca500 Disconnect failed: {e}")

        # 3. Tear down the Bota sensor regardless of Meca500 outcome.
        if self.sensor:
            try:
                self.sensor.deactivate()
                self.sensor.shutdown()
            except Exception as e:
                logger.warning(f"Bota sensor shutdown failed: {e}")
            self.sensor = None

        self._connected = False
