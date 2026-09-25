"""Record a SpaceMouse-teleop dataset on the Meca500 microscope/pipette rig.

Cameras: microscope (index 0) + overhead (index 1) + wrist (index 3). The operator drives
coarsely (gain_tr/gain_rot) until the pipette is under the microscope, then presses
**Enter** to toggle precision mode — the SpaceMouse gains drop to gain_tr_fine/
gain_rot_fine for fine positioning (press Enter again to toggle back to coarse). The
toggle state (`precision.state`) is recorded as an action channel and reset to coarse
on the between-episode auto-home.

Uses the auto-home-between-episodes flow (record_reset.record_with_reset) so the arm
homes after each demo and the precision toggle resets for the next approach.

Workflow: edit the CONFIG block below, then `python record_microscope.py`.
Ctrl-C to stop, edit, up-arrow, run again.

Tip: find good exposure/focus + confirm camera indices first with
scripts/meca500/check_camera.py, then paste the values below.
"""

from record_reset import record_with_reset

from lerobot.robots.meca500_microscope.config_meca500_microscope import Meca500MicroscopeConfig
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig
from lerobot.teleoperators.meca500_spacemouse.config_meca500_spacemouse import Meca500SpacemouseConfig
from lerobot.utils.import_utils import register_third_party_plugins

# ----------------------------- CONFIG -----------------------------
USER = "AdamAxelrod"
NAME = "microscope_pipette"
TASK = "move_pipette_under_microscope"

NUM_EPISODES = 100
FPS = 30
EPISODE_TIME_S = 120
RESET_TIME_S = 60
DISPLAY_DATA = True
PUSH_TO_HUB = True

# Coarse movement scale (SpaceMouse at full deflection); the default until Enter toggles.
GAIN_TR = 50.0  # mm/s  translation
GAIN_ROT = 30.0  # deg/s rotation

# Fine (precision) scale, used while the Enter toggle is on under the microscope.
# Drop to 1-2 mm/s for very fine positioning.
GAIN_TR_FINE = 5.0  # mm/s  translation
GAIN_ROT_FINE = 3.0  # deg/s rotation

# Auto-home target (executed between episodes; also resets the precision toggle).
HOME_JOINTS = [70, 10, 10, 90, -80, 15]
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

    cfg = RecordConfig(
        robot=build_robot_config(),  # monitor_mode=True (default) — teleop owns activation
        dataset=DatasetRecordConfig(
            repo_id=f"{USER}/{NAME}",
            single_task=TASK,
            num_episodes=NUM_EPISODES,
            fps=FPS,
            episode_time_s=EPISODE_TIME_S,
            reset_time_s=RESET_TIME_S,
            push_to_hub=PUSH_TO_HUB,
        ),
        teleop=Meca500SpacemouseConfig(
            id="meca500_spacemouse",
            gain_tr=GAIN_TR,
            gain_rot=GAIN_ROT,
            gain_tr_fine=GAIN_TR_FINE,
            gain_rot_fine=GAIN_ROT_FINE,
            home_joints=HOME_JOINTS,
            home_timeout_s=HOME_TIMEOUT_S,
        ),
        display_data=DISPLAY_DATA,
    )

    record_with_reset(cfg)


if __name__ == "__main__":
    main()
