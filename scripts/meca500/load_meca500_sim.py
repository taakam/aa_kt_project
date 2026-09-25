r"""Load Isaac Sim with the Meca500 arm spawned on a ground plane.

Opens the Isaac Sim viewer, drops in the Meca500 USD and lets you jog its 6
joints with a 3DConnexion SpaceMouse. Use it as a sanity check that the robot
loads as an articulation and as a starting point for sim work (teleop, IK, RL).

SpaceMouse control: the 6 axes (x/y/z + roll/pitch/yaw) map to joints 1..6;
deflecting an axis drives that joint, and any device button snaps back to home.
Joints are driven through the tensor API, which (unlike the GUI's drive-target
slider) works on the GPU pipeline -- so there is no need to run on ``--device
cpu``. Without a SpaceMouse connected the arm just holds its home pose.

Tune ``--jog-gain`` for speed and ``--deadzone`` for drift; flip ``AXIS_SIGNS``
entries if a joint moves the wrong way.

Note: this loads the *arm-only* asset (6 revolute joints, no gripper). The full
``urdf/meca/meca.usd`` export is empty/corrupt (492-byte stub), so we use the
working ``urdf/meca_arm_only/meca_arm_only.usd`` instead.

IMPORTANT: run with the Isaac Sim env, NOT the meca500 hardware .venv:

    & "$env:LOCALAPPDATA\miniconda3\envs\leisaac_envhub\python.exe" scripts/meca500/load_meca500_sim.py

Add ``--headless`` to run without the GUI, ``--steps N`` to stop after N physics
steps (0 = run until the window is closed). Ctrl-C also stops it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# --- launch Isaac Sim first; everything below imports its runtime modules ---
parser = argparse.ArgumentParser(description="Load the Meca500 in Isaac Sim.")
parser.add_argument("--steps", type=int, default=0, help="Stop after N physics steps (0 = run until closed).")
parser.add_argument(
    "--jog-gain", type=float, default=1.0, help="Joint speed (rad/s) at full SpaceMouse axis deflection."
)
parser.add_argument(
    "--deadzone", type=float, default=0.1, help="Ignore SpaceMouse axis values below this magnitude (0-1)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ruff: noqa: E402 — these imports require the simulation app to be running.
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

# Repo root is two levels up from scripts/meca500/.
REPO_ROOT = Path(__file__).resolve().parents[2]
MECA_USD = REPO_ROOT / "mecademic_description" / "urdf" / "meca_arm_only" / "meca_arm_only.usd"
ENV_USD = REPO_ROOT / "mecademic_description" / "desk.usd"

# World position of the arm's base, seated on the desk (read from the GUI). Used
# both to spawn the arm and to aim the default camera, so they stay in sync.
ARM_BASE_POS = (-0.649353, -0.084213, 0.911893)

# hidapi.dll is not bundled in the Isaac Sim env. Reuse the copy already sitting
# in the meca500 hardware .venv so pyspacemouse/easyhid can dlopen it.
HIDAPI_DLL_DIR = REPO_ROOT / ".venv" / "Scripts"

# Maps the 6 SpaceMouse axes (x, y, z, roll, pitch, yaw) onto joints 1..6.
# Flip an entry to -1.0 if a joint jogs the wrong way for your device.
AXIS_SIGNS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def open_spacemouse():
    """Open the 3DConnexion SpaceMouse. Returns the device handle, or None on failure."""
    if HIDAPI_DLL_DIR.is_dir():
        # easyhid finds the DLL via ctypes.util.find_library(), which on Windows
        # only scans PATH (it ignores os.add_dll_directory). Prepend the dir to
        # PATH so the lookup succeeds regardless of how the script was launched;
        # add_dll_directory then covers any transitive DLL dependencies.
        os.environ["PATH"] = str(HIDAPI_DLL_DIR) + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(str(HIDAPI_DLL_DIR))
    try:
        import pyspacemouse
    except ImportError:
        print("[meca500] pyspacemouse not installed; run `pip install pyspacemouse`.", flush=True)
        return None
    try:
        dev = pyspacemouse.open()
    except Exception as e:
        print(f"[meca500] failed to open SpaceMouse: {e}", flush=True)
        return None
    if dev is None:
        print("[meca500] no SpaceMouse found (open() returned None).", flush=True)
    return dev


def design_scene() -> Articulation:
    """Spawn ground, light, and the Meca500 articulation; return the robot."""
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)

    # Spawn the Fusion 360 environment (desk) as a static, collidable prop. A USD
    # with collision but no rigid-body API becomes a static collider -- exactly
    # what scenery should be (the arm collides with it; it never moves).
    if not ENV_USD.is_file():
        raise FileNotFoundError(f"Environment USD not found at {ENV_USD}")
    env_cfg = sim_utils.UsdFileCfg(
        usd_path=ENV_USD.as_posix(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    env_cfg.func("/World/Environment", env_cfg, translation=(0.0, 0.0, 0.0))

    if not MECA_USD.is_file():
        raise FileNotFoundError(f"Meca500 USD not found at {MECA_USD}")

    # The USD's defaultPrim (/meca_500_r3) carries the articulation; IsaacLab
    # auto-detects the ArticulationRootAPI prim beneath prim_path.
    robot_cfg = ArticulationCfg(
        prim_path="/World/Meca500",
        spawn=sim_utils.UsdFileCfg(usd_path=MECA_USD.as_posix()),
        init_state=ArticulationCfg.InitialStateCfg(pos=ARM_BASE_POS),
        # Position-control every joint so the arm holds its pose against gravity.
        actuators={
            "meca_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=400.0,
                damping=40.0,
            ),
        },
    )
    return Articulation(robot_cfg)


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    # Aim the camera at the arm: target a point ~15 cm above its base (roughly the
    # body's center), with the eye offset back/up so the whole arm is in frame.
    ax, ay, az = ARM_BASE_POS
    target = (ax, ay, az + 0.15)
    eye = (ax + 0.6, ay + 0.6, az + 0.4)
    sim.set_camera_view(eye=eye, target=target)

    robot = design_scene()
    sim.reset()

    print(f"[meca500] articulation root: {robot.root_physx_view.prim_paths[0]}", flush=True)
    print(f"[meca500] {robot.num_joints} joints: {robot.joint_names}", flush=True)

    sm = open_spacemouse()
    if sm is not None:
        print(
            "[meca500] SpaceMouse ready: x/y/z + roll/pitch/yaw jog joints 1-6. "
            "Press any SpaceMouse button to snap back to the home pose.",
            flush=True,
        )
    else:
        print("[meca500] no SpaceMouse - holding the home pose.", flush=True)

    # Start at the default (home) joint configuration; target_q is what we drive.
    home_q = robot.data.default_joint_pos.clone()
    target_q = home_q.clone()
    robot.write_joint_state_to_sim(home_q, robot.data.default_joint_vel.clone())

    # Per-joint limits so jogging can't push a joint past its physical range.
    limits = robot.data.joint_pos_limits[0]  # [num_joints, 2]
    q_min, q_max = limits[:, 0], limits[:, 1]
    signs = torch.tensor(AXIS_SIGNS, device=target_q.device)

    # Drive through the tensor API (set_joint_position_target). Unlike the GUI's
    # drive-target path, this works on the GPU pipeline (eENABLE_DIRECT_GPU_API).
    sim_dt = sim.get_physics_dt()
    step = 0
    while simulation_app.is_running():
        if sm is not None:
            state = sm.read()
            if state is not None:
                axes = torch.tensor(
                    [state.x, state.y, state.z, state.roll, state.pitch, state.yaw],
                    device=target_q.device,
                )
                axes = torch.where(axes.abs() < args_cli.deadzone, torch.zeros_like(axes), axes)
                # Integrate the deflection into a per-step joint delta (rad).
                target_q[0] += signs * axes * args_cli.jog_gain * sim_dt
                target_q[0] = torch.clamp(target_q[0], q_min, q_max)
                if any(state.buttons):
                    target_q = home_q.clone()

        robot.set_joint_position_target(target_q)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        step += 1
        if args_cli.steps and step >= args_cli.steps:
            print(f"[meca500] reached {step} steps, exiting.", flush=True)
            break

    if sm is not None:
        try:
            sm.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
    # On Windows, simulation_app.close() can hang indefinitely during CUDA/physics
    # teardown. The run is already finished here, so exit the process immediately
    # (the OS reclaims GPU/memory) instead of waiting on a graceful shutdown.
    os._exit(0)
