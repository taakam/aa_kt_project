from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig
from lerobot.robots.meca500.config_meca500 import Meca500Config


@RobotConfig.register_subclass("meca500_microscope")
@dataclass
class Meca500MicroscopeConfig(Meca500Config):
    """Meca500 with the microscope/pipette rig: microscope camera (index 0),
    overhead camera (index 1), and wrist camera (index 3).

    Identical Meca500 hardware/behaviour — only the camera set differs. Focus and
    exposure are locked (autofocus/autoexposure off) so the visual statistics stay
    consistent between recording and rollout. `focus`/`exposure` are driver-dependent;
    tune per camera/lighting with scripts/meca500/check_camera.py.

    NOTE (verify on the Windows rig): three USB cameras on one bus can exceed
    bandwidth. The wrist cam therefore uses MJPG at a modest 640x480. USB indices are
    not stable across reboots/replugs on Windows — confirm each index (microscope is 0,
    overhead is 1, wrist is 3) and that all three stream without dropped frames via
    scripts/meca500/check_camera.py before running.
    """

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "overhead_cam": OpenCVCameraConfig(
                index_or_path=1,
                fps=30,
                width=640,
                height=480,
                # DirectShow, not the CAP_ANY default (which picks MSMF on Windows and
                # fails to open these UVC cams). Matches scripts/meca500/check_camera.py.
                backend=Cv2Backends.DSHOW,
                autofocus=False,
                focus=100,
                autoexposure=False,
                exposure=-6,
            ),
            "wrist_cam": OpenCVCameraConfig(
                # Index on the Windows rig — microscope=0, overhead=1, wrist=3.
                # Confirm with check_camera.py.
                index_or_path=3,
                fps=30,
                width=640,
                height=480,
                # MJPG to fit a third stream alongside overhead + microscope on the bus.
                fourcc="MJPG",
                backend=Cv2Backends.DSHOW,
                autofocus=False,
                focus=100,
                autoexposure=False,
                exposure=-6,
            ),
            "microscope_cam": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                # Native resolution of the USB microscope cam — it snaps any other
                # request to 1280x1024, which the strict width/fps check then rejects.
                width=1280,
                height=1024,
                # MJPG keeps 1.3MP @ 30fps within USB bandwidth alongside the overhead cam.
                fourcc="MJPG",
                backend=Cv2Backends.DSHOW,
                # Focus is fixed by the lens ring on this camera — the driver rejects
                # CAP_PROP_FOCUS, so leave focus control untouched (None).
                autofocus=None,
                focus=None,
                autoexposure=False,
                exposure=-6,
            ),
        }
    )
