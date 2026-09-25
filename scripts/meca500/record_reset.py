"""Record a dataset and auto-home the Meca500 between episodes.

Same flow as `lerobot-record`, plus a one-shot `MoveJoints(HOME_JOINTS)` over
the selected teleop's control socket after each demo (and on rerecord). Skipped
after the final episode. Edit the CONFIG block below, then `python record_reset.py`.
"""

import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.robots.meca500.config_meca500 import Meca500Config
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.meca500_bota.config_meca500_bota import Meca500BotaConfig
from lerobot.teleoperators.meca500_spacemouse.config_meca500_spacemouse import Meca500SpacemouseConfig
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun

# ============================== CONFIG ==============================
# Teleoperator: "bota" (force-sensor hand-guidance) or "spacemouse" (3DConnexion)
TELEOP = "spacemouse"

# Dataset
USER = "AdamAxelrod"
NAME = "space_mouse_purple_dot"
SINGLE_TASK = "reach_purple_dot"
NUM_EPISODES = 100
EPISODE_TIME_S = 60  # seconds per demo
RESET_TIME_S = 60  # manual environment-reset window after auto-home
FPS = 30
PUSH_TO_HUB = True
PRIVATE = False

# Recording UI
DISPLAY_DATA = True  # rerun viewer
PLAY_SOUNDS = True
RESUME = False  # append to an existing dataset?
CLEAR_EXISTING_CACHE = True  # if not resuming, delete the local cache for REPO_ID before recording

# Auto-home (executed between episodes by this script)
HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 90.0, 0.0]
HOME_TIMEOUT_S = 30.0
# ====================================================================


def build_config() -> RecordConfig:
    robot_cfg = Meca500Config(id="meca500")  # defaults: monitor_mode=True + overhead/wrist cams
    if TELEOP == "bota":
        teleop_cfg = Meca500BotaConfig(
            id="meca500_bota",
            home_joints=HOME_JOINTS,
            home_timeout_s=HOME_TIMEOUT_S,
        )
    elif TELEOP == "spacemouse":
        teleop_cfg = Meca500SpacemouseConfig(
            id="meca500_spacemouse",
            home_joints=HOME_JOINTS,
            home_timeout_s=HOME_TIMEOUT_S,
        )
    else:
        raise ValueError(f"Unknown TELEOP: {TELEOP!r}. Expected 'bota' or 'spacemouse'.")
    dataset_cfg = DatasetRecordConfig(
        repo_id=f"{USER}/{NAME}",
        single_task=SINGLE_TASK,
        num_episodes=NUM_EPISODES,
        episode_time_s=EPISODE_TIME_S,
        reset_time_s=RESET_TIME_S,
        fps=FPS,
        push_to_hub=PUSH_TO_HUB,
        private=PRIVATE,
    )
    return RecordConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        dataset=dataset_cfg,
        display_data=DISPLAY_DATA,
        play_sounds=PLAY_SOUNDS,
        resume=RESUME,
    )


def record_with_reset(cfg: RecordConfig, clear_existing_cache: bool = True) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # Fresh recording: optionally wipe any prior local cache for this repo_id
        # so LeRobotDataset.create() can mkdir() it cleanly. Mirrors the path
        # logic in LeRobotDataset.__init__ (root override else HF_LEROBOT_HOME / repo_id).
        if clear_existing_cache:
            cache_root = Path(cfg.dataset.root) if cfg.dataset.root else HF_LEROBOT_HOME / cfg.dataset.repo_id
            if cache_root.exists():
                logging.warning(f"Removing existing dataset cache at {cache_root}")
                shutil.rmtree(cache_root)

        sanity_check_dataset_name(cfg.dataset.repo_id, None)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    robot.connect()
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

    # The whole session lives inside try/finally so Ctrl-C still tears down the
    # bota driver (otherwise nanobind leaks) and the Meca500 control socket
    # (otherwise the controller faults on "control connection dropped").
    try:
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                )

                # Skip the reset phase (and the auto-home) after the final
                # episode unless the user re-records.
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    # Auto-home over the teleop's control socket before the
                    # manual reset window. Duck-typed so a missing go_home()
                    # is just a warning.
                    if teleop is not None and hasattr(teleop, "go_home"):
                        try:
                            log_say("Homing robot", cfg.play_sounds)
                            teleop.go_home()
                        except Exception as e:
                            logging.warning(f"teleop.go_home() failed, continuing: {e}")
                    else:
                        logging.warning(
                            "Teleop does not implement go_home(); skipping auto-home. "
                            "record_reset.py expects meca500_bota."
                        )

                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_compressed_images=False,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                # Guard against an empty buffer (e.g., user hit Escape before
                # the first frame was captured) — save_episode() raises on empty.
                buffer_size = dataset.episode_buffer.get("size", 0) if dataset.episode_buffer else 0
                if buffer_size == 0:
                    logging.warning("Skipping save_episode(): no frames captured.")
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if cfg.dataset.push_to_hub:
            try:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            except Exception as e:
                # Don't lose the clean shutdown over an auth/network failure —
                # the local dataset under HF_LEROBOT_HOME is the source of truth.
                logging.warning(
                    f"push_to_hub failed: {e}. Local dataset is preserved; "
                    f"run `huggingface-cli login` with a write-enabled token, "
                    f"then `huggingface-cli upload {cfg.dataset.repo_id} "
                    f"{HF_LEROBOT_HOME / cfg.dataset.repo_id} --repo-type=dataset` to retry."
                )
    finally:
        # Tear down hardware regardless of how we exit (normal end, Ctrl-C,
        # or an unexpected exception). Each disconnect is guarded so a
        # failure in one doesn't skip the others.
        try:
            robot.disconnect()
        except Exception as e:
            logging.warning(f"robot.disconnect() failed: {e}")
        if teleop is not None:
            try:
                teleop.disconnect()
            except Exception as e:
                logging.warning(f"teleop.disconnect() failed: {e}")
        if not is_headless() and listener is not None:
            try:
                listener.stop()
            except Exception:
                pass

    log_say("Exiting", cfg.play_sounds)
    return dataset


def main() -> None:
    register_third_party_plugins()
    record_with_reset(build_config(), clear_existing_cache=CLEAR_EXISTING_CACHE)


if __name__ == "__main__":
    main()
