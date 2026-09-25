# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Backends, Cv2Rotation

__all__ = ["OpenCVCameraConfig", "ColorMode", "Cv2Rotation", "Cv2Backends"]


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """Configuration class for OpenCV-based camera devices or video files.

    This class provides configuration options for cameras accessed through OpenCV,
    supporting both physical camera devices and video files. It includes settings
    for resolution, frame rate, color mode, and image rotation.

    Example configurations:
    ```python
    # Basic configurations
    OpenCVCameraConfig(0, 30, 1280, 720)   # 1280x720 @ 30FPS
    OpenCVCameraConfig(/dev/video4, 60, 640, 480)   # 640x480 @ 60FPS

    # Advanced configurations with FOURCC format
    OpenCVCameraConfig(128422271347, 30, 640, 480, rotation=Cv2Rotation.ROTATE_90, fourcc="MJPG")     # With 90° rotation and MJPG format
    OpenCVCameraConfig(0, 30, 1280, 720, fourcc="YUYV")     # With YUYV format
    ```

    Attributes:
        index_or_path: Either an integer representing the camera device index,
                      or a Path object pointing to a video file.
        fps: Requested frames per second for the color stream.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)
        fourcc: FOURCC code for video format (e.g., "MJPG", "YUYV", "I420"). Defaults to None (auto-detect).
        backend: OpenCV backend identifier (https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html). Defaults to ANY.
        autofocus: Whether the camera should auto-focus. None (default) leaves the camera's own
                   default untouched; True/False enables/disables autofocus.
        focus: Manual focus value to apply (driver-dependent units). None (default) leaves it untouched.
               Requires `autofocus=False` (most drivers ignore a manual focus while autofocus is on).
        autoexposure: Whether the camera should auto-expose. None (default) leaves the camera's own
                      default untouched; True/False enables/disables auto-exposure. The backend-specific
                      magic value OpenCV expects is handled internally.
        exposure: Manual exposure value to apply (driver-dependent units). None (default) leaves it
                  untouched. Requires `autoexposure=False` (most drivers ignore a manual exposure
                  while auto-exposure is on).

    Note:
        - Only 3-channel color output (RGB/BGR) is currently supported.
        - FOURCC codes must be 4-character strings (e.g., "MJPG", "YUYV"). Some common FOUCC codes: https://learn.microsoft.com/en-us/windows/win32/medfound/video-fourccs#fourcc-constants
        - Setting FOURCC can help achieve higher frame rates on some cameras.
        - Exposure/focus controls are best-effort: OpenCV's support for them is backend- and
          driver-dependent, and many cameras silently ignore them. Unsupported controls log a
          warning rather than failing the connection. For imitation learning you usually want
          `autofocus=False` and `autoexposure=False` (plus fixed `focus`/`exposure`) so the visual
          statistics stay consistent between recording and rollout.
    """

    index_or_path: int | Path
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    fourcc: str | None = None
    backend: Cv2Backends = Cv2Backends.ANY
    autofocus: bool | None = None
    focus: int | None = None
    autoexposure: bool | None = None
    exposure: float | None = None

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        self.backend = Cv2Backends(self.backend)

        if self.fourcc is not None and (not isinstance(self.fourcc, str) or len(self.fourcc) != 4):
            raise ValueError(
                f"`fourcc` must be a 4-character string (e.g., 'MJPG', 'YUYV'), but '{self.fourcc}' is provided."
            )

        if self.focus is not None and self.autofocus is not False:
            raise ValueError(
                "`focus` requires `autofocus=False`; most drivers ignore a manual focus value "
                "while autofocus is enabled."
            )

        if self.exposure is not None and self.autoexposure is not False:
            raise ValueError(
                "`exposure` requires `autoexposure=False`; most drivers ignore a manual exposure value "
                "while auto-exposure is enabled."
            )
