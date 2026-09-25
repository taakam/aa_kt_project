from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("meca500_spacemouse")
@dataclass
class Meca500SpacemouseConfig(TeleoperatorConfig):
    # Mecademic control connection
    meca_address: str = "192.168.0.100"

    # Velocity gains. SpaceMouse axes are normalised to [-1, 1].
    # MoveLinVelWrf takes mm/s for translation and deg/s for rotation,
    # expressed in the Meca world (base) frame.
    gain_tr: float = 50.0
    gain_rot: float = 30.0

    # Fine (precision) gains used while the Enter-latched precision mode is active
    # (e.g. once the pipette is under the microscope). Scaled down hard so the
    # operator can finely position the tip. Selected live in the guidance loop.
    gain_tr_fine: float = 5.0
    gain_rot_fine: float = 3.0

    # Deadzones in normalised input units (post sign flip).
    deadzone_tr: float = 0.05
    deadzone_rot: float = 0.05

    # Low-pass filter factor applied to the raw 6D input (0..1, higher = less smoothing).
    alpha: float = 0.3

    # Per-axis sign flips to align 3DConnexion device frame with the Meca world frame.
    # Order: [x, y, z, roll, pitch, yaw]. Tune by jogging each axis individually.
    axis_signs: list[float] = field(default_factory=lambda: [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0])

    # Internal control loop period (seconds). 5 ms ≈ 200 Hz is plenty for a SpaceMouse.
    loop_dt: float = 0.005

    # Auto-home target used by record_reset.py between episodes.
    home_joints: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 90.0, 0.0])
    home_timeout_s: float = 30.0
