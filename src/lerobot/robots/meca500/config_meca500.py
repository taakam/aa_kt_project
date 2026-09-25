from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("meca500")
@dataclass
class Meca500Config(RobotConfig):
    ip_address: str = "192.168.0.100"

    monitor_mode: bool = True

    # Standard camera configuration.
    # Focus and exposure are locked (autofocus/autoexposure off) so the visual
    # statistics stay consistent between recording and rollout. `focus`/`exposure`
    # values are driver-dependent — tune per camera/lighting if frames look off.
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "overhead_cam": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=640,
                height=480,
                autofocus=False,
                focus=100,
                autoexposure=False,
                exposure=-6,
            ),
            "wrist_cam": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=640,
                height=480,
                autofocus=False,
                focus=100,
                autoexposure=False,
                exposure=-6,
            ),
        }
    )

    max_relative_target: float | dict[str, float] | None = None

    # Tighter per-step joint-delta clamp applied at deployment when the policy's
    # predicted `precision.state` channel is active (> 0.5). Lets a trained policy
    # "scale down its own movements" under the microscope. None disables the gate
    # (the normal max_relative_target still applies). Tune after the first checkpoint.
    precision_max_relative_target: float | dict[str, float] | None = None

    default_joint_vel: float = 25.0
