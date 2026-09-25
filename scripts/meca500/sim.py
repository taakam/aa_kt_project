r"""Launch the Meca500 + desk scene in Isaac Sim without memorising the CLI.

``load_meca500_sim.py`` has to run under the Isaac Sim Python, which lives in a
conda env with a long, machine-specific path. This launcher finds that
interpreter for you and re-runs the real script under it, so you can start it
with whatever ``python`` happens to be on your PATH:

    python scripts/meca500/sim.py           # open the GUI viewer
    python scripts/meca500/sim.py --test    # headless smoke test, exits on its own

Any other flags are passed straight through to ``load_meca500_sim.py``:

    python scripts/meca500/sim.py --jog-gain 0.5
    python scripts/meca500/sim.py --headless --steps 500

Interpreter lookup order:
  1. ``$MECA_SIM_PYTHON`` if set (use this if your env lives somewhere unusual)
  2. the current interpreter, if ``isaaclab`` is already importable
  3. the usual conda env locations for Windows and Linux
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = Path(__file__).resolve().parent / "load_meca500_sim.py"

# The scene's geometry payload; desk.usd references it as ./Environment.usd.
DESK_USD = REPO_ROOT / "mecademic_description" / "desk.usd"
ENV_USD = REPO_ROOT / "mecademic_description" / "Environment.usd"

# Conda env names that have carried the Isaac Sim install on our machines.
ENV_NAMES = ("leisaac_envhub", "isaaclab", "env_isaaclab")


def candidate_pythons() -> list[Path]:
    """Plausible Isaac Sim interpreters, best guess first."""
    homes = [Path.home() / n for n in ("miniconda3", "anaconda3", "miniforge3")]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        homes += [Path(local) / n for n in ("miniconda3", "anaconda3")]

    exe = "python.exe" if os.name == "nt" else "python"
    subdir = "" if os.name == "nt" else "bin"

    out = []
    for home in homes:
        for name in ENV_NAMES:
            out.append(home / "envs" / name / subdir / exe)
    return out


def find_python() -> Path:
    """Locate an interpreter that can run Isaac Lab, or exit with guidance."""
    override = os.environ.get("MECA_SIM_PYTHON")
    if override:
        p = Path(override)
        if not p.is_file():
            sys.exit(f"[sim] MECA_SIM_PYTHON points at a missing file: {p}")
        return p

    # Already inside the Isaac Sim env (e.g. `conda activate isaaclab`)? Use it.
    if importlib.util.find_spec("isaaclab") is not None:
        return Path(sys.executable)

    for p in candidate_pythons():
        if p.is_file():
            return p

    sys.exit(
        "[sim] Could not find an Isaac Sim Python.\n"
        "      Activate the Isaac Lab conda env and re-run, or point at it directly:\n"
        "        MECA_SIM_PYTHON=/path/to/envs/isaaclab/bin/python python scripts/meca500/sim.py"
    )


def preflight() -> None:
    """Check the scene assets are present before paying the ~1 min Kit startup."""
    if not TARGET.is_file():
        sys.exit(f"[sim] Missing the sim script: {TARGET}")
    if not DESK_USD.is_file():
        sys.exit(f"[sim] Missing desk USD: {DESK_USD}")
    if not ENV_USD.is_file():
        # desk.usd payloads this in; without it the desk silently fails to appear.
        sys.exit(
            f"[sim] Missing desk geometry: {ENV_USD}\n"
            "      desk.usd payloads ./Environment.usd. Pull the latest commit,\n"
            "      or re-copy the file next to desk.usd."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the Meca500 + desk scene in Isaac Sim.",
        epilog="Unrecognised flags are forwarded to load_meca500_sim.py.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Headless smoke test: load the scene, step 200 times, exit.",
    )
    args, passthrough = parser.parse_known_args()

    preflight()

    forwarded = list(passthrough)
    if args.test and not forwarded:
        forwarded = ["--headless", "--steps", "200"]

    python = find_python()
    cmd = [str(python), str(TARGET), *forwarded]
    print(f"[sim] using {python}", flush=True)
    print(f"[sim] {' '.join(cmd[1:])}", flush=True)

    # Run from the repo root so the script's relative asset paths resolve.
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
