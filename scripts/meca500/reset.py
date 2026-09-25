"""Drive the Meca500 to a fixed home pose, then disconnect.

Workflow: edit HOME_JOINTS below, then `python reset.py`.

Uses the `meca500_home` teleop to source the target pose and `Meca500.send_action`
to execute it. The Meca500 wrapper is opened with `monitor_mode=False` so it
owns activation/homing this time (unlike during teleop/record where the
Bota teleop owns the control connection).
"""

import logging
import time

from lerobot.robots.meca500.config_meca500 import Meca500Config
from lerobot.robots.meca500.meca500 import Meca500
from lerobot.teleoperators.meca500_home.config_meca500_home import Meca500HomeConfig
from lerobot.teleoperators.meca500_home.meca500_home import Meca500Home
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

# ----------------------------- CONFIG -----------------------------
HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 90.0, 0.0]
MOVE_TIMEOUT_S = 30.0  # max wait for the arm to reach HOME_JOINTS
# ------------------------------------------------------------------


def main() -> None:
    init_logging()
    register_third_party_plugins()

    # Robot in control mode (not monitor) so it activates, homes, and accepts MoveJoints.
    # No cameras needed for a reset — drop them to skip OpenCV startup.
    robot = Meca500(Meca500Config(id="meca500", monitor_mode=False, cameras={}))
    teleop = Meca500Home(Meca500HomeConfig(id="meca500_home", home=HOME_JOINTS))

    robot.connect()       # activates + homes to factory zero
    robot.configure()     # sets joint velocity / blending defaults
    teleop.connect()

    try:
        action = teleop.get_action()
        logging.info(f"Driving to home: {action}")
        robot.send_action(action)

        # Block until the motion queue drains; otherwise robot.disconnect() will
        # ClearMotion() and abort the move mid-flight.
        deadline = time.perf_counter() + MOVE_TIMEOUT_S
        robot.robot.WaitIdle(timeout=MOVE_TIMEOUT_S)
        if time.perf_counter() > deadline:
            logging.warning("WaitIdle returned past deadline — move may not have completed.")
        else:
            logging.info("Home pose reached.")
    finally:
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
