# Meca500 Robot Learning

A fork of [🤗 LeRobot](https://github.com/huggingface/lerobot) extended with hardware, teleoperation, and simulation support for a **Mecademic Meca500** 6-axis industrial arm, built for PhD research on learned fine-manipulation under a microscope.

Everything upstream LeRobot provides (datasets, policies, training, rollout) still works unchanged. This README covers **what has been added on top**:

| Area | What was added |
|---|---|
| **Robots** | `meca500`, `meca500_microscope` — LeRobot `Robot` implementations driving a real Meca500 over `mecademicpy` |
| **Teleoperators** | `meca500_spacemouse` (3DConnexion 6-DoF jogging), `meca500_bota` (force-sensor hand guidance), `meca500_home` (fixed-pose driver) |
| **Precision mode** | A latched fine-motion mode recorded as an action channel and reproduced at rollout time |
| **Scripts** | `scripts/meca500/` — edit-a-config-block-and-run wrappers for teleop, recording, checkpoint testing, inference, and camera setup |
| **Simulation** | Meca500 USD assets + an Isaac Lab / Isaac Sim scene with SpaceMouse jogging |

---

## The task

The target task is **positioning a stripper pipette under a microscope** with the Meca500 — a manipulation problem where the useful working scale is far smaller than the arm's approach scale. The rig carries three cameras: an overhead view, a wrist view, and a USB microscope camera looking at the workspace at 1280×1024.

The core idea in this fork is that **coarse approach and fine positioning are different regimes**, and the policy should learn to switch between them rather than being tuned to a single motion scale.

---

## Teleoperation

### SpaceMouse (`meca500_spacemouse`)

The main teleop path. A 3DConnexion SpaceMouse's 6 axes drive `MoveLinVelWrf` in the Meca's world frame — translation in mm/s, rotation in deg/s — through a dedicated ~200 Hz control loop, with per-axis sign flips, deadzones, and a low-pass filter to make the raw HID input usable for demonstration data.

**Precision mode.** Pressing **Enter** latches a fine-gain mode: translation drops from 50 → 5 mm/s and rotation from 30 → 3 deg/s, so the operator can position the pipette tip under the microscope without fighting the gains used for the approach. Pressing Enter again toggles back.

Crucially, the latch state is recorded as an action channel — `precision.state` — alongside the 6 joint targets. The policy therefore learns *when* to be in precision mode, and at rollout time `Meca500.send_action` reads the policy's own predicted `precision.state`: when it crosses 0.5, per-step joint motion is clamped to `precision_max_relative_target` instead of the normal `max_relative_target`. The trained policy scales down its own movements as it approaches the microscope.

See [config_meca500_spacemouse.py](src/lerobot/teleoperators/meca500_spacemouse/config_meca500_spacemouse.py) for the gains and [meca500_spacemouse.py](src/lerobot/teleoperators/meca500_spacemouse/meca500_spacemouse.py) for the control loop.

### Bota force sensor (`meca500_bota`)

Hand-guidance teleop: a Bota force/torque sensor at the flange lets the operator physically push the arm through a demonstration. Forces and moments above threshold are mapped to Cartesian velocity commands with a low-pass filter and hysteresis band. Used for the earlier reach-target datasets.

### Home (`meca500_home`)

A degenerate "teleoperator" that just emits a fixed joint pose. Used as the reset driver during autonomous rollout and by [reset.py](scripts/meca500/reset.py).

---

## Scripts

Everything in [scripts/meca500/](scripts/meca500/) follows the same pattern: a `CONFIG` block at the top of the file, then `python <script>.py`. Ctrl-C, edit, up-arrow, run again. Each docstring lists the equivalent `lerobot-*` CLI invocation.

| Script | Purpose |
|---|---|
| [teleoperate_microscope.py](scripts/meca500/teleoperate_microscope.py) | SpaceMouse teleop on the microscope rig — practice the Enter toggle and **H** reset, tune gains |
| [record_microscope.py](scripts/meca500/record_microscope.py) | Record demonstrations on the microscope rig, with auto-home between episodes |
| [record_reset.py](scripts/meca500/record_reset.py) | Generic record loop that issues a `MoveJoints(HOME_JOINTS)` after each episode (and on re-record) |
| [teleoperate.py](scripts/meca500/teleoperate.py) / [record.py](scripts/meca500/record.py) | Bota hand-guidance equivalents |
| [test_checkpoint_microscope.py](scripts/meca500/test_checkpoint_microscope.py) | Run a local training checkpoint on the rig mid-training, with the precision clamp active |
| [inference.py](scripts/meca500/inference.py) | Run a Hub-hosted trained policy via the episodic rollout strategy |
| [reset.py](scripts/meca500/reset.py) | Drive the arm to a fixed home pose and disconnect |
| [check_camera.py](scripts/meca500/check_camera.py) | Open one camera index and preview it, to identify which physical camera it is |
| [verify_camera_settings.py](scripts/meca500/verify_camera_settings.py) | Sweep exposure and focus to *visually* confirm auto-exposure/autofocus are actually disabled |
| [load_meca500_sim.py](scripts/meca500/load_meca500_sim.py) | Isaac Sim scene (see below) |

Training is driven from PowerShell via [train.ps1](train.ps1), which wraps `lerobot-train` with the run naming, output-dir collision handling, and logging used here.

---

## Cameras

The rig runs three USB cameras on one Windows machine, which is where most of the non-obvious configuration lives:

- **Windows backends.** All cameras open with the DirectShow backend explicitly — the OpenCV `CAP_ANY` default picks MSMF on Windows and fails to open these UVC devices.
- **Locked focus and exposure.** Autofocus and auto-exposure are disabled so visual statistics stay consistent between recording and rollout. On Windows this cannot be verified by read-back (DSHOW reports `-1.0`/`2.0` regardless, and `set()` returns success even when ignored), hence the sweep-based [verify_camera_settings.py](scripts/meca500/verify_camera_settings.py).
- **USB bandwidth.** The microscope camera streams 1280×1024 MJPG at its native resolution (it snaps any other request); the wrist camera also uses MJPG so a third stream fits on the bus.
- **Indices are not stable** across reboots and replugs — confirm them with `check_camera.py` before every session.

See [config_meca500_microscope.py](src/lerobot/robots/meca500_microscope/config_meca500_microscope.py).

---

## Simulation — Isaac Lab / Isaac Sim

[scripts/meca500/load_meca500_sim.py](scripts/meca500/load_meca500_sim.py) brings the Meca500 up in Isaac Sim as an Isaac Lab `Articulation`, on a ground plane with the Fusion 360 desk environment spawned as a static collider, and lets you jog all 6 joints with the same SpaceMouse used on the real arm (any device button snaps back to home).

Notes worth knowing before running it:

- Joints are driven through the **tensor API** (`set_joint_position_target`), not the GUI drive-target slider — so it works on the GPU pipeline and does not need `--device cpu`.
- The arm asset is [mecademic_description/urdf/meca_arm_only/](mecademic_description/urdf/meca_arm_only/) (6 revolute joints, no gripper). The full `urdf/meca/meca.usd` export is an empty 492-byte stub.
- `hidapi.dll` is not bundled with the Isaac Sim environment; the script reuses the copy in the hardware `.venv` and prepends it to `PATH`, because easyhid resolves it via `ctypes.util.find_library()` which ignores `os.add_dll_directory`.
- `simulation_app.close()` can hang indefinitely on Windows during CUDA/physics teardown, so the script ends with `os._exit(0)`.

It runs in a **separate Isaac Sim conda environment**, not the hardware `.venv`:

```powershell
& "$env:LOCALAPPDATA\miniconda3\envs\leisaac_envhub\python.exe" scripts/meca500/load_meca500_sim.py
```

Useful flags: `--headless`, `--steps N`, `--jog-gain`, `--deadzone`. `AXIS_SIGNS` in the script flips a joint's jog direction.

The [mecademic_description/](mecademic_description/) package holds the URDF, meshes, launch files, and the `desk.usd` / `Environment.usd` scene assets.

---

## Setup

Hardware code targets **Windows** — that is the machine wired to the Meca500 and its cameras. Development happens on macOS, but macOS is not a reliable proxy for camera focus/exposure behaviour.

```powershell
uv sync --locked --extra all
```

Then, for the SpaceMouse, drop `hidapi.dll` into `.venv\Scripts\` (pyspacemouse 2.0 needs it on `PATH`), and make sure `pynput` is installed — without it the Enter-latch precision mode is silently disabled and `precision.state` stays pinned at 0.

The arm is reached over Ethernet at `192.168.0.100` by default.

Typical loop:

```powershell
python scripts/meca500/check_camera.py             # confirm camera indices
python scripts/meca500/teleoperate_microscope.py   # tune gains, practise the task
python scripts/meca500/record_microscope.py        # collect demonstrations
. .\train.ps1; Start-Training -Dataset microscope_pipette
python scripts/meca500/test_checkpoint_microscope.py
```

---

## Repository notes

- Contributor and architecture guidance for this fork lives in [AGENTS.md](AGENTS.md); a practical user-facing walkthrough of the upstream LeRobot workflow is in [AGENT_GUIDE.md](AGENT_GUIDE.md).
- The Meca500 additions follow LeRobot's own extension points — `RobotConfig.register_subclass` / `TeleoperatorConfig.register_subclass` — so the standard `lerobot-record`, `lerobot-train`, and `lerobot-rollout` CLIs work against them directly.

---

## Upstream

This repository is a fork of **LeRobot** by Hugging Face. All upstream functionality, documentation, and licensing (Apache 2.0) are retained. See the [upstream repository](https://github.com/huggingface/lerobot) and [documentation](https://huggingface.co/docs/lerobot/index).

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```
