from dataclasses import dataclass, field

from ..config import TeleoperatorConfig
import numpy as np


@TeleoperatorConfig.register_subclass("meca500_bota")
@dataclass
class Meca500BotaConfig(TeleoperatorConfig):
    # Port to connect to the arm
    meca_address: str = "192.168.0.100"

    json_path: str = "bota_sensor_config.json"
    sensor_type: str = "Bota_Binary"
    sensor_port: str = "COM5" # Or /dev/ttyUSB0 if on Mac

    # Hand guidance parameters (Ported from hand_guidance.py)
    gain_tr: int = 10 #10
    gain_rot: int = 50  #50
    f_threshold_high: float = 0.5
    f_threshold_low: float = 0.1
    m_threshold_high: float = 0.05
    m_threshold_low: float = 0.01

    alpha: float = 0.1 #  0.1 Simple low pass filter factor

    # Target pose used by go_home() (invoked between episodes by record_reset.py).
    home_joints: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 90.0, 0.0])
    home_timeout_s: float = 30.0
