import logging

from ..teleoperator import Teleoperator
from .config_meca500_home import Meca500HomeConfig

logger = logging.getLogger(__name__)


class Meca500Home(Teleoperator):
    """Passive teleop that emits a fixed home pose every frame.

    Why: during `lerobot-record` policy inference there is no leader/teleop,
    so the reset phase between episodes has no action source and the arm
    stays wherever the policy left it. Passing this teleop fills that gap —
    the recording phase still runs from the policy (policy takes precedence
    over teleop in `record_loop`), and the reset phase drives the arm back
    to `config.home` via the robot wrapper's `send_action`.
    """

    config_class = Meca500HomeConfig
    name = "meca500_home"

    def __init__(self, config: Meca500HomeConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        if len(self.config.home) != 6:
            raise ValueError(
                f"meca500_home expects 6 joint values, got {len(self.config.home)}: {self.config.home}"
            )

    @property
    def action_features(self) -> dict[str, type]:
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

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True
        logger.info(f"{self} connected (passive home teleop, target={self.config.home}).")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        return {f"joint_{i + 1}.pos": float(v) for i, v in enumerate(self.config.home)}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    def disconnect(self) -> None:
        self._connected = False
