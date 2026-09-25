"""Evaluate differential IK over many randomized table targets."""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batch-test Meca500 IK target reachability.")
parser.add_argument("--trials", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=25)
parser.add_argument("--steps_per_trial", type=int, default=300)
parser.add_argument("--success_threshold", type=float, default=0.012)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms

from reach_env import MecaReachEnv, MecaReachEnvCfg


def main() -> None:
    num_envs = min(args_cli.batch_size, args_cli.trials)

    cfg = MecaReachEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 60.0

    env = MecaReachEnv(cfg)
    env.reset()

    robot = env.robot
    joint_ids = env._joint_ids
    ee_body_id = env._ee_body_id
    jacobian_index = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

    controller_cfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )
    controller = DifferentialIKController(
        controller_cfg,
        num_envs=num_envs,
        device=env.device,
    )

    sim_dt = env.sim.get_physics_dt()
    identity_quat = torch.zeros((num_envs, 4), device=env.device)
    identity_quat[:, 0] = 1.0

    successes = 0
    tested = 0
    final_errors_all = []
    failed_local_targets = []

    num_batches = math.ceil(args_cli.trials / num_envs)

    for batch in range(num_batches):
        active_count = min(num_envs, args_cli.trials - tested)
        env_ids = torch.arange(num_envs, device=env.device)
        env._reset_idx(env_ids)

        for _ in range(10):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(sim_dt)

        desired_pos_w = env.goal_pos_w.clone()
        root_pose_w = robot.data.root_pose_w
        desired_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            desired_pos_w,
            identity_quat,
        )

        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        _, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

        controller.reset()
        controller.set_command(desired_pos_b, ee_quat=ee_quat_b)

        reached = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        error = torch.full((num_envs,), float("inf"), device=env.device)

        for _ in range(args_cli.steps_per_trial):
            jacobian = robot.root_physx_view.get_jacobians()[
                :, jacobian_index, :, joint_ids
            ]
            ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, joint_ids]

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3],
                ee_pose_w[:, 3:7],
            )

            joint_pos_des = controller.compute(
                ee_pos_b,
                ee_quat_b,
                jacobian,
                joint_pos,
            )

            limits = robot.data.joint_pos_limits[:, joint_ids]
            joint_pos_des = torch.clamp(
                joint_pos_des,
                min=limits[..., 0],
                max=limits[..., 1],
            )

            robot.set_joint_position_target(joint_pos_des, joint_ids=joint_ids)
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(sim_dt)

            current_pos_w = robot.data.body_pose_w[:, ee_body_id, 0:3]
            error = torch.linalg.vector_norm(desired_pos_w - current_pos_w, dim=-1)
            reached |= error < args_cli.success_threshold

            if reached[:active_count].all():
                break

        active_errors = error[:active_count].detach().cpu()
        active_reached = reached[:active_count].detach().cpu()
        final_errors_all.extend(active_errors.tolist())
        successes += int(active_reached.sum().item())

        local_targets = (
            env._target_pos_w[:active_count] - env.scene.env_origins[:active_count]
        ).detach().cpu()

        for index in range(active_count):
            if not active_reached[index]:
                failed_local_targets.append(local_targets[index].tolist())

        tested += active_count
        print(
            f"[IK batch] {tested}/{args_cli.trials} tested, "
            f"running success rate={100.0 * successes / tested:.1f}%",
            flush=True,
        )

    errors = torch.tensor(final_errors_all)
    print("", flush=True)
    print(f"[IK batch] success: {successes}/{tested} ({100.0 * successes / tested:.1f}%)", flush=True)
    print(f"[IK batch] mean final error: {errors.mean().item():.4f} m", flush=True)
    print(f"[IK batch] worst final error: {errors.max().item():.4f} m", flush=True)

    if failed_local_targets:
        failed = torch.tensor(failed_local_targets)
        print(
            "[IK batch] failed local target bounds: "
            f"x=[{failed[:, 0].min().item():.4f}, {failed[:, 0].max().item():.4f}], "
            f"y=[{failed[:, 1].min().item():.4f}, {failed[:, 1].max().item():.4f}]",
            flush=True,
        )
    else:
        print("[IK batch] all sampled targets were reachable.", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
