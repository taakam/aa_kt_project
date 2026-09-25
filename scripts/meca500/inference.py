"""Run policy inference on the Meca500.

Workflow: edit the CONFIG block below, then `python inference.py`.
Ctrl-C to stop, edit, up-arrow, run again.

Equivalent to:
    lerobot-rollout --strategy.type=episodic \
        --robot.type=meca500 --robot.monitor_mode=False \
        --teleop.type=meca500_home --teleop.home=[0,0,0,0,90,0] \
        --policy.path=<USER>/<RUN>_model \
        --dataset.repo_id=<USER>/rollout_<RUN> \
        --dataset.single_task=<TASK> \
        --dataset.push_to_hub=False
"""

import sys

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots.meca500.config_meca500 import Meca500Config
from lerobot.rollout.configs import EpisodicStrategyConfig, RolloutConfig
from lerobot.scripts.lerobot_rollout import rollout
from lerobot.teleoperators.meca500_home.config_meca500_home import Meca500HomeConfig
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import register_third_party_plugins

# ----------------------------- CONFIG -----------------------------
USER = "AdamAxelrod"
RUN = "space_mouse_puple_dot_100demos"
TASK = "reach_purple_dot"

# ACT inference mode:
#   None             → full-chunk inference (fast, wobbles at chunk boundaries)
#   float in (0, 1)  → temporal ensembling (smooth, every-step inference)
TEMPORAL_ENSEMBLE_COEFF: float | None = None

HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 90.0, 0.0]
NUM_EPISODES = 10
EPISODE_TIME_S = 30
RESET_TIME_S = 10
FPS = 30
DISPLAY_DATA = True
# ------------------------------------------------------------------


def main() -> None:
    register_third_party_plugins()

    policy_path = f"{USER}/{RUN}_model"

    policy_overrides: list[str] = []
    if TEMPORAL_ENSEMBLE_COEFF is not None:
        policy_overrides = [
            f"--temporal_ensemble_coeff={TEMPORAL_ENSEMBLE_COEFF}",
            "--n_action_steps=1",
        ]

    # Strip CLI args before constructing RolloutConfig so __post_init__ doesn't
    # re-parse them and overwrite the policy we're about to load.
    sys.argv = [sys.argv[0]]

    policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=policy_overrides)
    policy.pretrained_path = policy_path

    cfg = RolloutConfig(
        robot=Meca500Config(id="meca500", monitor_mode=False),
        dataset=DatasetRecordConfig(
            repo_id=f"{USER}/rollout_{RUN}",
            single_task=TASK,
            fps=FPS,
            num_episodes=NUM_EPISODES,
            episode_time_s=EPISODE_TIME_S,
            reset_time_s=RESET_TIME_S,
            push_to_hub=False,
        ),
        teleop=Meca500HomeConfig(id="meca500_home", home=HOME_JOINTS),
        policy=policy,
        strategy=EpisodicStrategyConfig(),
        fps=FPS,
        display_data=DISPLAY_DATA,
    )

    try:
        rollout(cfg)
    except DeviceNotConnectedError as e:
        sys.exit(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
