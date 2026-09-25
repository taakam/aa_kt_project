"""Hand-guidance teleop on the Meca500 with the Bota force sensor or a SpaceMouse.

Workflow: edit the CONFIG block below, then `python teleoperate.py`.
Ctrl-C to stop, edit, up-arrow, run again.

Equivalent to:
    lerobot-teleoperate --robot.type=meca500 --teleop.type=meca500_bota \
        --display_data=true
"""

import sys

from lerobot.robots.meca500.config_meca500 import Meca500Config
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.teleoperators.meca500_bota.config_meca500_bota import Meca500BotaConfig
from lerobot.teleoperators.meca500_spacemouse.config_meca500_spacemouse import Meca500SpacemouseConfig
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import register_third_party_plugins

# ----------------------------- CONFIG -----------------------------
TELEOP = "bota"  # "bota" or "spacemouse"
DISPLAY_DATA = False
FPS = 60
# ------------------------------------------------------------------

TELEOP_CONFIGS = {
    "bota": Meca500BotaConfig,
    "spacemouse": Meca500SpacemouseConfig,
}


def main() -> None:
    register_third_party_plugins()

    if TELEOP not in TELEOP_CONFIGS:
        raise ValueError(f"TELEOP must be one of {list(TELEOP_CONFIGS)}, got {TELEOP!r}")

    cfg = TeleoperateConfig(
        robot=Meca500Config(id="meca500"),  # monitor_mode=True (default) — teleop owns activation
        teleop=TELEOP_CONFIGS[TELEOP](id=f"meca500_{TELEOP}"),
        fps=FPS,
        display_data=DISPLAY_DATA,
    )

    try:
        teleoperate(cfg)
    except DeviceNotConnectedError as e:
        sys.exit(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
