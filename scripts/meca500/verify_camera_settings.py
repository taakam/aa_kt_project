"""Verify — and SEE — that auto-exposure and autofocus are disabled on the Meca500 cameras.

Run this ON THE WINDOWS LAPTOP (not the Mac) — it's the only place the cameras and their
real drivers exist:

    python scripts/meca500/verify_camera_settings.py        # all cameras in Meca500Config
    python scripts/meca500/verify_camera_settings.py 0       # just index 0

A live window pops up per camera (rendered with matplotlib, because lerobot ships the
headless build of OpenCV which has no GUI). You can watch with your own eyes:

  1. EXPOSURE SWEEP  — auto-exposure is disabled, then CAP_PROP_EXPOSURE is stepped from
     dark -> bright. You should SEE the picture get brighter at each step. If it does, the
     auto-exposure loop is genuinely off (an active loop would fight back to a flat
     brightness). The brightness number in the title should climb.

  2. FOCUS SWEEP     — autofocus is disabled, then CAP_PROP_FOCUS is stepped across its
     range. Point the camera at a TEXTURED scene at a normal working distance: you should
     SEE it pull in and out of focus, and the sharpness number rise and fall.

  3. LIVE LOCKED VIEW — the camera is set to the exact values from Meca500Config. Now change
     the lighting yourself (cover the lens, turn a lamp on/off). With exposure locked the
     brightness number FOLLOWS the light instead of being auto-corrected back to a target —
     that is what "locked" looks like. Press Q (or close the window) to continue.

Why a sweep instead of trusting read-back: on Windows, get() is unreliable (DSHOW reports
-1.0 for auto-exposure and 2.0 for autofocus no matter what), and set() returns success=True
even when ignored. Only the picture itself tells the truth.
"""

import platform
import statistics
import sys
import time

import cv2  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

from lerobot.robots.meca500.config_meca500 import Meca500Config

EXPOSURE_SWEEP = [-11.0, -9.0, -7.0, -5.0, -3.0]
FOCUS_SWEEP = [0.0, 50.0, 100.0, 150.0, 200.0, 255.0]
EXPOSURE_PASS_PCT = 25.0
FOCUS_PASS_PCT = 25.0
STEP_SECONDS = 2.0  # how long to linger on each sweep step so you can see it


def auto_exposure_off_value() -> float:
    """Backend magic value meaning 'manual exposure' — mirrors OpenCVCamera."""
    return 0.25 if platform.system() == "Windows" else 1.0


def brightness_metric(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def sharpness_metric(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class Viewer:
    """A single matplotlib window we keep updating frame by frame."""

    def __init__(self, title: str):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.fig.canvas.manager.set_window_title(title)
        self.ax.axis("off")
        self.im = None
        self.quit = False
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", lambda _evt: setattr(self, "quit", True))

    def _on_key(self, event) -> None:
        if event.key in ("q", "escape"):
            self.quit = True

    def show(self, cap: cv2.VideoCapture, metric, seconds: float, header: list[str]) -> tuple[float, bool]:
        """Render the live feed for `seconds`, overlaying header + the live metric.

        Returns (mean metric over the window, quit_requested).
        """
        vals: list[float] = []
        t0 = time.time()
        while time.time() - t0 < seconds and not self.quit:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            m = metric(gray)
            vals.append(m)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.im is None:
                self.im = self.ax.imshow(rgb)
            else:
                self.im.set_data(rgb)
            self.ax.set_title("\n".join([*header, f"current = {m:.1f}"]), fontsize=11)
            plt.pause(0.001)
        return (sum(vals) / max(len(vals), 1), self.quit)

    def close(self) -> None:
        plt.close(self.fig)


def sweep(viewer: Viewer, cap: cv2.VideoCapture, prop: int, values: list[float], metric, label: str,
          title: str) -> float:
    readings = []
    for v in values:
        cap.set(prop, v)
        mean_m, _ = viewer.show(cap, metric, STEP_SECONDS, [title, f"{label} = {v}"])
        readings.append(mean_m)
        print(f"    {label}={v:>7}  ->  {mean_m:8.1f}")
        if viewer.quit:
            break
    spread = max(readings) - min(readings)
    return 100.0 * spread / max(statistics.mean(readings), 1e-6)


def check_camera(name: str, index: int, backend: int, exposure: float, focus: float) -> bool:
    print(f"\n=== {name} (index {index}) ===")
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"  ERROR: could not open index {index}; skipping.")
        return False
    print(f"  backend in use : {cap.getBackendName()}")
    viewer = Viewer(f"{name} (index {index}) — press Q to continue")

    # Warm the stream so the first frames aren't garbage.
    viewer.show(cap, brightness_metric, 1.0, ["warming up..."])

    # --- Exposure: disable AE, then prove manual exposure drives brightness. ---
    print("  exposure sweep (auto-exposure OFF -> picture should brighten each step):")
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure_off_value())
    viewer.show(cap, brightness_metric, 0.4, ["disabling auto-exposure..."])
    exp_pct = sweep(
        viewer, cap, cv2.CAP_PROP_EXPOSURE, EXPOSURE_SWEEP, brightness_metric,
        "exposure", "AUTO-EXPOSURE OFF -- brightness should climb",
    )
    exposure_ok = exp_pct > EXPOSURE_PASS_PCT
    print(f"    => brightness spread {exp_pct:.0f}% of mean -> auto-exposure "
          f"{'LOCKED' if exposure_ok else 'NOT disabled'}")

    # --- Focus: disable AF, then prove manual focus drives sharpness. ---
    print("  focus sweep (autofocus OFF -> picture should pull in/out of focus):")
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
    viewer.show(cap, sharpness_metric, 0.4, ["disabling autofocus..."])
    foc_pct = sweep(
        viewer, cap, cv2.CAP_PROP_FOCUS, FOCUS_SWEEP, sharpness_metric,
        "focus", "AUTOFOCUS OFF -- focus should change (needs a textured scene)",
    )
    focus_ok = foc_pct > FOCUS_PASS_PCT
    print(f"    => sharpness spread {foc_pct:.0f}% of mean -> autofocus "
          f"{'LOCKED' if focus_ok else 'NOT disabled (or scene has no texture)'}")

    # --- Live locked view with the real config values. ---
    print(f"  live locked view: exposure={exposure}, focus={focus}. Change the lighting; "
          f"brightness should follow it. Press Q (or close the window) to continue.")
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure_off_value())
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
    cap.set(cv2.CAP_PROP_FOCUS, focus)
    viewer.quit = False  # reset so the close/Q from earlier stages doesn't skip this
    header = [
        "LOCKED at config values",
        f"exposure={exposure}  focus={focus}",
        "Change the lighting -> brightness should follow (no auto-correct).",
        "Press Q or close the window to continue.",
    ]
    viewer.show(cap, brightness_metric, 600.0, header)

    cap.release()
    viewer.close()
    time.sleep(0.3)
    return exposure_ok and focus_ok


def main() -> None:
    cfg = Meca500Config(id="meca500")
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None

    results = {}
    for name, cam_cfg in cfg.cameras.items():
        index = cam_cfg.index_or_path
        if only is not None and index != only:
            continue
        results[name] = check_camera(
            name,
            int(index),
            int(cam_cfg.backend),
            float(cam_cfg.exposure),
            float(cam_cfg.focus),
        )

    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'LOCKED ✅' if ok else 'NEEDS ATTENTION ❌'}")


if __name__ == "__main__":
    main()
