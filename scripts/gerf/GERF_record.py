"""Record a teleoperated dataset with the GERF SO-101 leader/follower setup + USB camera.

This is the dataset-recording counterpart to GERF_teleoperate.py. The leader arm
drives the follower; every tick the follower's joint state + camera frame(s) are
stashed into a LeRobotDataset (parquet for joints, mp4 for video).

Edit the CONFIG block below, then:

    python GERF_record.py

During recording (needs a focused window, non-headless):
    Right Arrow : stop the current episode early and move on
    Left  Arrow : re-record the current episode
    Esc         : stop recording entirely

Nothing here is GERF-specific about the camera — `SO101FollowerConfig` already
has full USB-camera support, identical to the meca500 wrapper. You only supply a
camera index in the CONFIG block; no extra integration code is required.
"""

import time

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

# ============================== CONFIG ==============================
# --- Arms (same as GERF_teleoperate.py) ---
FOLLOWER_PORT = "COM3"
FOLLOWER_ID = "my_yellow_robot_arm"
LEADER_PORT = "COM4"
LEADER_ID = "my_purple_leader_arm"

# Safety: max per-motor position change per command (degrees / gripper units).
# None = no clamp. A small value (e.g. 5.0) guards against sudden jumps.
MAX_RELATIVE_TARGET = None

# --- Camera(s) ---
# Change CAMERA_INDEX to whatever your USB camera enumerates as on this machine.
# List indices with:  python -m lerobot.find_cameras opencv
CAMERA_INDEX = 0
CAMERA_NAME = "front_cam"   # becomes observation.images.front_cam in the dataset
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# To add a second camera (e.g. a wrist cam), uncomment its block below.
CAMERAS = {
    CAMERA_NAME: OpenCVCameraConfig(
        index_or_path=CAMERA_INDEX,
        fps=CAMERA_FPS,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    ),
    # "wrist_cam": OpenCVCameraConfig(
    #     index_or_path=2, fps=30, width=640, height=480,
    # ),
}

# --- Dataset / recording (mirrors lerobot-record's DatasetRecordConfig) ---
REPO_ID = "adambaxelrod/gerf_dataset"     # "{hf_username}/{dataset_name}"
SINGLE_TASK = "Pick up the object and place it in the box."
ROOT = None                 # local storage dir; None = default HF cache location
NUM_EPISODES = 10           # number of demos to record
FPS = 30                    # control + dataset frame rate
EPISODE_TIME_S = 30         # seconds of recording per episode
RESET_TIME_S = 10           # seconds to reset the scene between episodes
PUSH_TO_HUB = False         # upload to the HF Hub when finished
PRIVATE = False             # if pushing, make the repo private
TAGS = None                 # e.g. ["gerf", "so101"]
USE_VIDEO = True            # encode camera frames to mp4 (vs raw png)

# --- UX ---
DISPLAY_DATA = True        # stream cameras/state to a rerun viewer
PLAY_SOUNDS = True          # spoken episode prompts
NUM_IMAGE_WRITER_THREADS_PER_CAMERA = 4
# ===================================================================


def record_episode(robot, teleop, dataset, events, control_time_s):
    """Run the teleop->send->log loop for up to `control_time_s` seconds.

    When `dataset` is None the loop still drives the arm (used for the reset
    window between episodes) but records nothing.
    """
    timestamp = 0.0
    start_t = time.perf_counter()
    while timestamp < control_time_s:
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        action = teleop.get_action()
        robot.send_action(action)

        if dataset is not None:
            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
            frame = {**obs_frame, **action_frame, "task": SINGLE_TASK}
            dataset.add_frame(frame)

        if DISPLAY_DATA:
            from lerobot.utils.visualization_utils import log_rerun_data

            log_rerun_data(observation=obs, action=action)

        dt = time.perf_counter() - loop_start
        precise_sleep(max(0.0, 1.0 / FPS - dt))
        timestamp = time.perf_counter() - start_t


def main():
    init_logging()

    if DISPLAY_DATA:
        from lerobot.utils.visualization_utils import init_rerun

        init_rerun(session_name="gerf_recording")

    robot = SO101Follower(
        SO101FollowerConfig(
            port=FOLLOWER_PORT,
            id=FOLLOWER_ID,
            cameras=CAMERAS,
            max_relative_target=MAX_RELATIVE_TARGET,
        )
    )
    teleop = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID))

    # Build dataset features from the robot's own observation/action schema, so
    # the camera(s) are picked up automatically from CAMERAS above.
    dataset_features = combine_feature_dicts(
        hw_to_dataset_features(robot.action_features, ACTION, USE_VIDEO),
        hw_to_dataset_features(robot.observation_features, OBS_STR, USE_VIDEO),
    )

    dataset = LeRobotDataset.create(
        REPO_ID,
        FPS,
        root=ROOT,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=USE_VIDEO,
        image_writer_processes=0,
        image_writer_threads=NUM_IMAGE_WRITER_THREADS_PER_CAMERA * len(robot.cameras),
    )

    robot.connect()
    teleop.connect()

    listener, events = init_keyboard_listener()

    try:
        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < NUM_EPISODES and not events["stop_recording"]:
                log_say(f"Recording episode {recorded + 1} of {NUM_EPISODES}", PLAY_SOUNDS)
                record_episode(robot, teleop, dataset, events, EPISODE_TIME_S)

                # Reset window (skip after the final episode, unless re-recording).
                if not events["stop_recording"] and (
                    recorded < NUM_EPISODES - 1 or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", PLAY_SOUNDS)
                    record_episode(robot, teleop, dataset=None, events=events, control_time_s=RESET_TIME_S)

                if events["rerecord_episode"]:
                    log_say("Re-record episode", PLAY_SOUNDS)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded += 1

        log_say("Stop recording", PLAY_SOUNDS, blocking=True)
    finally:
        try:
            robot.disconnect()
        except Exception as e:
            print(f"Error disconnecting robot: {e}")
        try:
            teleop.disconnect()
        except Exception as e:
            print(f"Error disconnecting teleop: {e}")
        if not is_headless() and listener is not None:
            listener.stop()

    if PUSH_TO_HUB:
        dataset.push_to_hub(tags=TAGS, private=PRIVATE)

    log_say("Exiting", PLAY_SOUNDS)


if __name__ == "__main__":
    main()
