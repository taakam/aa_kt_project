# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## ⚠️ Development vs. Deployment Environment

**Code is written on a macOS laptop, but the machine physically connected to the Meca500 robot (and its cameras) is a Windows laptop. All code must be written to run on that Windows machine.**

This means, especially for anything touching hardware (`robots/`, `cameras/`, `teleoperators/`, `motors/`, the `meca500` subpackage, and `scripts/meca500/`):

- **Target Windows at runtime.** Use `pathlib.Path` (never hardcoded POSIX paths or `/`-separated strings), avoid POSIX-only assumptions (`os.fork`, signal semantics, `/dev/...` device paths, `select` on non-sockets, etc.), and prefer cross-platform APIs.
- **Cameras use the Windows OpenCV backends** (DSHOW/MSMF), not V4L2/AVFoundation. Backend-specific values (e.g. the `CAP_PROP_AUTO_EXPOSURE` magic numbers, camera indices) must match Windows behavior. macOS is **not** a reliable proxy for testing camera focus/exposure.
- **macOS-side checks are best-effort only.** Tests and quick scripts may be run on the Mac for convenience, but hardware behavior is only authoritative on the Windows laptop. Don't conclude a hardware/camera change "works" from macOS results alone.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                                          # All tests
uv run pytest tests/datasets/test_lerobot_dataset.py::test_init_loads_data -svv  # Single test
DEVICE=cuda make test-end-to-end                                               # All E2E tests
pre-commit run --all-files                                                     # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

### Core abstractions

- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level for IL). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir (act, diffusion, pi0, pi05, smolvla, wall_x, sac, …). All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`rewards/`** — Reward model hierarchy parallel to policies. `PreTrainedRewardModel` base in `pretrained.py`; concrete models in `classifier/`, `topreward/`, `robometer/`, `sarm/`. Config in `configs/rewards.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

### Training

- **Imitation learning (IL)**: `scripts/lerobot_train.py` → `TrainPipelineConfig` in `configs/train.py`.
- **Reinforcement learning (RL)**: `rl/train_rl.py` → `TrainRLServerPipelineConfig` (extends `TrainPipelineConfig`, `dataset` optional). Actor/learner are separate processes. Algorithm configs live in `rl/algorithms/configs.py`; SAC is the current implementation.

### Deployment

- **`rollout/`** — Unified deployment engine invoked via `lerobot-rollout`. `RolloutConfig` selects a strategy and inference backend via draccus polymorphism:
  - `base` — autonomous rollout, no recording
  - `sentry` — continuous recording with size-based episode rotation
  - `highlight` — ring-buffer recording, save on keypress
  - `episodic` — mirrors `lerobot-record` (fixed episode count/duration)
  - `dagger` — human-in-the-loop correction collection (keyboard or foot pedal)
  - Inference backends: `sync` (blocking) or `rtc` (async, via gRPC)
- **`async_inference/`** — gRPC-based async policy server (`policy_server.py`) and robot client (`robot_client.py`). Used by the `rtc` inference backend.
- **`transport/`** — Protobuf/gRPC definitions (`services.proto`, generated `_pb2.py` files) shared between async inference and RL actor/learner.

### Utilities

- **`model/`** — Robot kinematics (`kinematics.py`).
- **`transforms/`** — Image transform utilities.
- **`scripts/`** — CLI entry points mapped in `pyproject.toml [project.scripts]`.

## CLI Entry Points

All mapped in `pyproject.toml [project.scripts]`:

| Command | Purpose |
|---|---|
| `lerobot-train` | Imitation learning training |
| `lerobot-eval` | Policy evaluation in sim |
| `lerobot-rollout` | Unified deployment (all strategies) |
| `lerobot-record` | Simple episode recording (legacy; episodic strategy preferred) |
| `lerobot-replay` | Replay a recorded dataset on hardware |
| `lerobot-teleoperate` | Teleoperation without recording |
| `lerobot-calibrate` | Robot/teleop calibration |
| `lerobot-setup-motors` | One-time motor ID + baudrate setup |
| `lerobot-find-port` | Identify USB port for a robot arm |
| `lerobot-find-cameras` | List available cameras |
| `lerobot-info` | Print dataset/policy metadata |
| `lerobot-dataset-viz` | Visualize a dataset |
| `lerobot-edit-dataset` | Edit dataset episodes |

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
- **gRPC generated files**: `*_pb2.py` and `*_pb2_grpc.py` in `transport/` are auto-generated from `services.proto` — do not edit them directly.
- **Runtime target is Windows**: code is authored on macOS but runs on the Windows laptop wired to the Meca500. Write Windows-compatible code (see "Development vs. Deployment Environment" above).
