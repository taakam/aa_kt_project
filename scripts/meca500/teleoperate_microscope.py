"""SpaceMouse teleop for the Meca500 microscope/pipette rig.

Jog the stripper pipette under the microscope with a 3DConnexion SpaceMouse. Cameras:
microscope (index 0) + overhead (index 1) + wrist (index 3). Drive coarsely (gain_tr/gain_rot),
then press **Enter** to toggle precision mode — the gains drop to gain_tr_fine/
gain_rot_fine for fine positioning (press Enter again to toggle back to coarse).
Press **H** to reset: the arm homes to HOME_JOINTS and the precision toggle clears,
the same MoveJoints reset that record_microscope.py runs between episodes. No
recording here; use it to practice the toggle/reset and tune gains before running
record_microscope.py.

Workflow: edit the CONFIG block below, then `python teleoperate_microscope.py`.
Ctrl-C to stop, edit, up-arrow, run again.

Tip: find good exposure/focus + confirm camera indices first with
scripts/meca500/check_camera.py, then paste the values below.

Equivalent to:
    lerobot-teleoperate --robot.type=meca500_microscope \
        --teleop.type=meca500_spacemouse --teleop.gain_tr=50 --teleop.gain_rot=30 \
        --teleop.gain_tr_fine=5 --teleop.gain_rot_fine=3 --display_data=true
"""

import sys

from lerobot.robots.meca500_microscope.config_meca500_microscope import Meca500MicroscopeConfig
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.teleoperators.meca500_spacemouse.config_meca500_spacemouse import Meca500SpacemouseConfig
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import register_third_party_plugins

# ----------------------------- CONFIG -----------------------------
DISPLAY_DATA = True
FPS = 60

# Coarse movement scale (SpaceMouse at full deflection); the default until Enter toggles.
GAIN_TR = 50.0  # mm/s  translation
GAIN_ROT = 30.0  # deg/s rotation

# Fine (precision) scale, used while the Enter toggle is on under the microscope.
# Drop to 1-2 mm/s for very fine positioning.
GAIN_TR_FINE = 5.0  # mm/s  translation
GAIN_ROT_FINE = 3.0  # deg/s rotation

# Reset ("home") pose. Press **H** during teleop to send the arm back here and
# drop the precision toggle — the same MoveJoints reset record runs between episodes.
HOME_JOINTS = [70, 10, 10, 90, -80, 15]  # deg, joints 1-6
HOME_TIMEOUT_S = 30.0

# Camera tuning (driver-dependent units; tune with check_camera.py first).
# The microscope cam has no software focus (fixed lens ring) — focus it by hand.
OVERHEAD_EXPOSURE, OVERHEAD_FOCUS = -6, 100
WRIST_EXPOSURE, WRIST_FOCUS = -6, 100
MICROSCOPE_EXPOSURE = -6
# ------------------------------------------------------------------


def build_robot_config() -> Meca500MicroscopeConfig:
    """Microscope rig config with the CONFIG-block camera tuning applied.

    autoexposure/autofocus are already set in the defaults, so assigning
    exposure/focus after construction is safe (it skips the __post_init__ check).
    """
    cfg = Meca500MicroscopeConfig(id="meca500_microscope")
    cfg.cameras["overhead_cam"].exposure = OVERHEAD_EXPOSURE
    cfg.cameras["overhead_cam"].focus = OVERHEAD_FOCUS
    cfg.cameras["wrist_cam"].exposure = WRIST_EXPOSURE
    cfg.cameras["wrist_cam"].focus = WRIST_FOCUS
    cfg.cameras["microscope_cam"].exposure = MICROSCOPE_EXPOSURE
    return cfg


def main() -> None:
    register_third_party_plugins()

    cfg = TeleoperateConfig(
        robot=build_robot_config(),  # monitor_mode=True (default) — teleop owns activation
        teleop=Meca500SpacemouseConfig(
            id="meca500_spacemouse",
            gain_tr=GAIN_TR,
            gain_rot=GAIN_ROT,
            gain_tr_fine=GAIN_TR_FINE,
            gain_rot_fine=GAIN_ROT_FINE,
            home_joints=HOME_JOINTS,
            home_timeout_s=HOME_TIMEOUT_S,
        ),
        fps=FPS,
        display_data=DISPLAY_DATA,
    )

    try:
        teleoperate(cfg)
    except DeviceNotConnectedError as e:
        sys.exit(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
