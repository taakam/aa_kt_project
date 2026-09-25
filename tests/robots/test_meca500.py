import numpy as np
from unittest.mock import MagicMock

from lerobot.robots.meca500.meca500 import Meca500
from lerobot.robots.meca500.config_meca500 import Meca500Config


class FakeCamera:
    def __init__(self, name: str, should_fail: bool):
        self.name = name
        self.should_fail = should_fail
        self.is_connected = False
        self.height = 120
        self.width = 160

    def connect(self):
        if self.should_fail:
            raise ConnectionError(f"Fake camera {self.name} unavailable")
        self.is_connected = True

    def async_read(self):
        if not self.is_connected:
            raise RuntimeError(f"Fake camera {self.name} is not connected")
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


def test_meca500_connect_with_partial_camera_failure():
    config = Meca500Config()

    robot = Meca500(config)
    robot.cameras = {
        "good_cam": FakeCamera("good_cam", should_fail=False),
        "bad_cam": FakeCamera("bad_cam", should_fail=True),
    }

    # Avoid requiring a real Meca500 hardware connection in unit tests.
    robot.robot = MagicMock()
    robot.robot.Connect = MagicMock()
    robot.robot.ActivateAndHome = MagicMock()
    robot.robot.WaitHomed = MagicMock()

    robot.connect()

    assert robot.is_connected

    obs = robot.get_observation()
    assert "good_cam" in obs
    assert "bad_cam" in obs
    assert obs["good_cam"].shape == (120, 160, 3)
    assert obs["bad_cam"].shape == (120, 160, 3)

    # Confirm that the invalid camera gives a valid blank frame as fallback
    assert np.count_nonzero(obs["bad_cam"]) == 0
