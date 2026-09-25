"""Train a policy on a recorded dataset.

Workflow: edit the CONFIG block below, then `python train.py`.
Ctrl-C to stop, edit, up-arrow, run again. Output dir is auto-suffixed so
re-runs do not collide; the corresponding `training_log_<runtag>.txt`
captures stdout+stderr.

Equivalent to:
    lerobot-train --dataset.repo_id=<USER>/<DATASET> --policy.type=act \
        --output_dir=outputs/train/<RUN> --policy.device=cuda \
        --policy.use_amp=true --batch_size=32 --num_workers=4 --steps=50000 \
        --save_freq=5000 --policy.push_to_hub=true \
        --policy.repo_id=<USER>/<DATASET>_model --wandb.enable=true
"""

import os
import sys
from pathlib import Path

# Silence the torchvision video-decoding deprecation warning before lerobot imports.
os.environ.setdefault(
    "PYTHONWARNINGS", "ignore::UserWarning:torchvision.io._video_deprecation_warning"
)

from lerobot.configs.default import DatasetConfig, WandBConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.scripts.lerobot_train import train  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

from train_logging import setup_split_logging  # noqa: E402

# ----------------------------- CONFIG -----------------------------
USER = "AdamAxelrod"
DATASET = "space_mouse_puple_dot"
RUN = DATASET  # output dir name under outputs/train/

STEPS = 50_000
BATCH_SIZE = 32
NUM_WORKERS = 4
SAVE_FREQ = 5_000

CHUNK_SIZE = 50
N_ACTION_STEPS = CHUNK_SIZE  # full-chunk inference (original ACT behavior)

USE_AMP = True
ENABLE_WANDB = True
PUSH_TO_HUB = True
# ------------------------------------------------------------------


def _autosuffix(base: Path) -> Path:
    """Match the .ps1 behavior: outputs/train/red_dot, red_dot_2, red_dot_3, ..."""
    if not base.exists():
        return base
    i = 2
    while True:
        candidate = base.with_name(f"{base.name}_{i}")
        if not candidate.exists():
            return candidate
        i += 1


def main() -> None:
    setup_split_logging()  # INFO -> train_log.txt, ERROR -> train_error.txt
    register_third_party_plugins()

    repo_id = f"{USER}/{DATASET}"
    model_repo = f"{USER}/{DATASET}_model"

    output_dir = _autosuffix(Path("outputs/train") / RUN)
    run_tag = output_dir.name

    policy = ACTConfig(
        device="cuda",
        use_amp=USE_AMP,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        push_to_hub=PUSH_TO_HUB,
        repo_id=model_repo if PUSH_TO_HUB else None,
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=repo_id),
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

    print(f"Dataset:    {repo_id}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Run tag:    {run_tag}", flush=True)
    print(f"Policy:     act (chunk_size={CHUNK_SIZE}, n_action_steps={N_ACTION_STEPS})", flush=True)
    print(f"Batch x WK: {BATCH_SIZE} x {NUM_WORKERS}", flush=True)
    print(f"Steps:      {STEPS} (save every {SAVE_FREQ})", flush=True)
    print("", flush=True)

    # Strip CLI args so @parser.wrap() and TrainPipelineConfig.validate() see a
    # clean argv — otherwise stray flags from the parent shell would be re-parsed.
    sys.argv = [sys.argv[0]]
    train(cfg)


if __name__ == "__main__":
    main()
