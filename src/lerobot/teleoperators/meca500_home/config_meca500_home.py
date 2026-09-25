from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("meca500_home")
@dataclass
class Meca500HomeConfig(TeleoperatorConfig):
    # Joint angles (degrees) the robot should be driven to during the
    # `lerobot-record` reset window when running policy inference without
    # a hand-guidance teleop. Order: joint_1 ... joint_6.
    home: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
