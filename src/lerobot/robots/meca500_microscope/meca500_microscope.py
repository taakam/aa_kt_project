from lerobot.robots.meca500.meca500 import Meca500

from .config_meca500_microscope import Meca500MicroscopeConfig


class Meca500Microscope(Meca500):
    """Same Meca500 robot, microscope camera rig.

    Behaviour is fully inherited — Meca500 builds its cameras from `config.cameras`,
    so the microscope camera set flows through unchanged. This subclass exists only so
    the dynamic device loader can resolve `Meca500MicroscopeConfig` -> `Meca500Microscope`.
    """

    config_class = Meca500MicroscopeConfig
    name = "meca500_microscope"
