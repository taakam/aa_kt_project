"""Open ONE camera by its port/index so you can SEE which physical camera it is.

Run this ON THE WINDOWS LAPTOP (not the Mac) — it's the only place the cameras and
their real drivers exist:

    python scripts/meca500/check_camera.py

Set CAMERA_INDEX below to the index you want to inspect, then run it. Each index has
a preset (resolution, focus, exposure, codec) in PRESETS so the microscope cam (0), the
overhead cam (1) and the wrist cam (3) each preview at their real settings; unlisted
indices use DEFAULT. A live window pops up (rendered with matplotlib, because lerobot ships the
headless build of OpenCV which has no GUI). Cover the lens / wave at it to confirm
which camera maps to this port. Press Q (or close the window) to quit.

If the port can't be opened you'll get a clear error and a list of which indices
DID open, so you can find the right one by trial.
"""

import platform

import cv2  # type: ignore
import matplotlib.pyplot as plt

# ======================================================================
# CONFIG — set the camera port/index you want to check.
# ======================================================================
CAMERA_INDEX = 3

# Per-index presets — switch CAMERA_INDEX and the matching resolution/controls apply.
# Keys mirror the meca500 rigs; any index not listed falls back to DEFAULT below.
#   width/height : resolution to request (None = driver default)
#   focus        : manual focus, or None to leave focus untouched (e.g. fixed-lens microscope)
#   exposure     : manual exposure, or None to leave auto-exposure on
#   fourcc       : pixel format, e.g. "MJPG" for high-res USB cams (None = driver default)
PRESETS = {
    0: {"width": 1280, "height": 1024, "focus": None, "exposure": -6, "fourcc": "MJPG"},  # microscope_cam
    1: {"width": 640, "height": 480, "focus": 100, "exposure": -6},                     # overhead_cam
    3: {"width": 640, "height": 480, "focus": 100, "exposure": -6, "fourcc": "MJPG"},   # wrist_cam
}
DEFAULT = {"width": 640, "height": 480, "focus": 100, "exposure": -6, "fourcc": None}
# ======================================================================


def settings_for(index: int) -> dict:
    """Preset for this index, with DEFAULT filling any missing keys."""
    return {**DEFAULT, **PRESETS.get(index, {})}


def default_backend() -> int:
    """DSHOW on Windows (what OpenCVCamera uses), platform default elsewhere."""
    return cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY


def auto_exposure_off_value() -> float:
    """Backend magic value meaning 'manual exposure' — mirrors OpenCVCamera."""
    return 0.25 if platform.system() == "Windows" else 1.0


def freeze_focus_and_exposure(cap: cv2.VideoCapture, settings: dict) -> None:
    """Lock focus/exposure exactly as OpenCVCamera does: disable the auto loop
    first, then apply the manual value (drivers ignore a manual value while the
    corresponding auto loop is still active)."""
    if settings["exposure"] is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure_off_value())
        cap.set(cv2.CAP_PROP_EXPOSURE, float(settings["exposure"]))
    if settings["focus"] is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
        cap.set(cv2.CAP_PROP_FOCUS, float(settings["focus"]))


def scan_ports(max_index: int = 10) -> list[int]:
    """Return the indices that can actually be opened — for the failure hint."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, default_backend())
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


def main() -> None:
    backend = default_backend()
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        print(f"ERROR: could not open camera port {CAMERA_INDEX}.")
        print("Scanning for ports that DO open...")
        ports = scan_ports()
        if ports:
            print(f"  available ports: {ports}")
        else:
            print("  no cameras found on indices 0-9.")
        cap.release()
        return

    settings = settings_for(CAMERA_INDEX)
    # FOURCC first: on some drivers it resets resolution, so apply it before size.
    if settings["fourcc"] is not None:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings["fourcc"]))
    if settings["width"] is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings["width"])
    if settings["height"] is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings["height"])

    freeze_focus_and_exposure(cap, settings)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Opened camera port {CAMERA_INDEX}")
    print(f"  backend    : {cap.getBackendName()}")
    print(f"  resolution : {actual_w}x{actual_h}")
    print("  Press Q (or close the window) to quit.")

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.canvas.manager.set_window_title(f"camera port {CAMERA_INDEX} — press Q to quit")
    ax.axis("off")
    ax.set_title(f"camera port {CAMERA_INDEX}  ({actual_w}x{actual_h}, {cap.getBackendName()})")

    quit_flag = {"stop": False}

    def on_key(event) -> None:
        if event.key in ("q", "escape"):
            quit_flag["stop"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", lambda _evt: quit_flag.update(stop=True))

    im = None
    while not quit_flag["stop"]:
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if im is None:
            im = ax.imshow(rgb)
        else:
            im.set_data(rgb)
        plt.pause(0.001)

    cap.release()
    plt.close(fig)


if __name__ == "__main__":
    main()
