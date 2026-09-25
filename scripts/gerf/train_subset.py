"""Train a policy on a random subset of episodes from a recorded dataset.

Workflow: edit the CONFIG block below, then `python train_subset.py`. Same as
`train.py` but samples NUM_DEMOS episodes from the dataset (seeded, so the
subset is reproducible and the same indices can be re-used across runs).

Equivalent to:
    lerobot-train --dataset.repo_id=<USER>/<DATASET> \
        --dataset.episodes="[i1,i2,...,iN]" --policy.type=act \
        --output_dir=outputs/train/<DATASET>_<N>demos --policy.device=<auto> ...

Device selection (automatic):
  cuda  — Nvidia GPU (Windows/Linux PC)
  mps   — Apple Silicon GPU (MacBook)
  cpu   — fallback
"""

import os
import random
import sys
from pathlib import Path

import torch

os.environ.setdefault(
    "PYTHONWARNINGS", "ignore::UserWarning:torchvision.io._video_deprecation_warning"
)

from lerobot.configs.default import DatasetConfig, WandBConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.scripts.lerobot_train import train  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

from train_logging import setup_split_logging  # noqa: E402

# ----------------------------- CONFIG -----------------------------
USER = "AdamAxelrod"
DATASET = "space_mouse_puple_dot"

NUM_DEMOS = 100
SEED = 42  # change to draw a different subset; same seed => same episodes

STEPS = 50_000
BATCH_SIZE = 32
NUM_WORKERS = 4
SAVE_FREQ = 5_000

CHUNK_SIZE = 50
N_ACTION_STEPS = CHUNK_SIZE

ENABLE_WANDB = True
PUSH_TO_HUB = True
# ------------------------------------------------------------------


def _select_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _autosuffix(base: Path) -> Path:
    if not base.exists():
        return base
    i = 2
    while True:
        candidate = base.with_name(f"{base.name}_{i}")
        if not candidate.exists():
            return candidate
        i += 1


def _sample_episodes(repo_id: str, n: int, seed: int) -> list[int]:
    meta = LeRobotDatasetMetadata(repo_id)
    total = meta.total_episodes
    if n > total:
        raise ValueError(f"NUM_DEMOS={n} exceeds dataset's {total} episodes ({repo_id}).")
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), n))


def main() -> None:
    setup_split_logging()  # INFO -> train_log.txt, ERROR -> train_error.txt
    register_third_party_plugins()

    repo_id = f"{USER}/{DATASET}"
    run_name = f"{DATASET}_{NUM_DEMOS}demos_seed{SEED}"

    device = _select_device()
    use_amp = device == "cuda"  # AMP only supported on CUDA

    episodes = _sample_episodes(repo_id, NUM_DEMOS, SEED)

    output_dir = _autosuffix(Path("outputs/train") / run_name)
    run_tag = output_dir.name

    # Hub repo matches the local output folder name (== wandb run name) so the
    # Hub model, the outputs/ dir, and the wandb run all share one identifier
    # and runs don't silently overwrite an existing same-name model on the Hub.
    model_repo = f"{USER}/{run_tag}"

    policy = ACTConfig(
        device=device,
        use_amp=use_amp,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        push_to_hub=PUSH_TO_HUB,
        repo_id=model_repo if PUSH_TO_HUB else None,
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=repo_id, episodes=episodes),
        policy=policy,
        output_dir=output_dir,
        job_name=run_tag,
        steps=STEPS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        save_checkpoint=True,
        save_freq=SAVE_FREQ,
        wandb=WandBConfig(enable=ENABLE_WANDB),
    )

    print(f"Device:     {device} (use_amp={use_amp})", flush=True)
    print(f"Dataset:    {repo_id}", flush=True)
    print(f"Subset:     {NUM_DEMOS} episodes (seed={SEED})", flush=True)
    print(f"Episodes:   {episodes}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Run tag:    {run_tag}", flush=True)
    print(f"Hub repo:   {model_repo if PUSH_TO_HUB else '(push disabled)'}", flush=True)
    print(f"Policy:     act (chunk_size={CHUNK_SIZE}, n_action_steps={N_ACTION_STEPS})", flush=True)
    print(f"Batch x WK: {BATCH_SIZE} x {NUM_WORKERS}", flush=True)
    print(f"Steps:      {STEPS} (save every {SAVE_FREQ})", flush=True)
    print("", flush=True)

    sys.argv = [sys.argv[0]]
    train(cfg)


if __name__ == "__main__":
    main()
