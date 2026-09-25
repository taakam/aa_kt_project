"""Scripted differential-IK reach test for the Meca500 environment.

This does not train a policy. It verifies that the robot, target coordinates,
end-effector body, Jacobian indexing, and joint controller are all consistent.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Isaac Lab differential IK on the Meca500 reach task.")
parser.add_argument("--steps", type=int, default=1200, help="Maximum simulation steps.")
parser.add_argument(
    "--approach_height",
    type=float,
    default=0.03,
    help="Height in metres above the red sphere centre.",
)
parser.add_argument(
    "--success_threshold",
    type=float,
    default=0.012,
    help="Position-error threshold in metres.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms

from reach_env import MecaReachEnv, MecaReachEnvCfg


def main() -> None:
    cfg = MecaReachEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 60.0

    env = MecaReachEnv(cfg)
    env.reset()

    robot = env.robot
    joint_ids = env._joint_ids
    ee_body_id = env._ee_body_id

    # PhysX omits the root link from the Jacobian array for fixed-base robots.
    ee_jacobian_index = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

    ik_cfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )
    ik_controller = DifferentialIKController(
        ik_cfg,
        num_envs=env.num_envs,
        device=env.device,
    )

    # Let PhysX initialize articulation buffers and Jacobians.
    sim_dt = env.sim.get_physics_dt()
    for _ in range(10):
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(sim_dt)

    # Aim slightly above the sphere so the tool does not drive into the desk.
    desired_pos_w = env._target_pos_w.clone()
    desired_pos_w[:, 2] += args_cli.approach_height

    # Convert the desired position from world frame to robot-base frame.
    root_pose_w = robot.data.root_pose_w
    identity_quat_w = torch.zeros((env.num_envs, 4), device=env.device)
    identity_quat_w[:, 0] = 1.0
    desired_pos_b, _ = subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        desired_pos_w,
        identity_quat_w,
    )

    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3],
        ee_pose_w[:, 3:7],
    )

    ik_controller.reset()
    ik_controller.set_command(
        desired_pos_b,
        ee_quat=ee_quat_b,
    )

    initial_ee_pos_w = robot.data.body_pose_w[:, ee_body_id, 0:3]
    initial_error = torch.linalg.vector_norm(desired_pos_w - initial_ee_pos_w, dim=-1)
    minimum_error = initial_error.clone()

    print(f"[IK] fixed base: {robot.is_fixed_base}", flush=True)
    print(f"[IK] EE body ID: {ee_body_id}", flush=True)
    print(f"[IK] Jacobian index: {ee_jacobian_index}", flush=True)
    print(f"[IK] target world position: {desired_pos_w[0].tolist()}", flush=True)
    print(f"[IK] initial error: {initial_error.item():.4f} m", flush=True)

    reached = False

    for step in range(args_cli.steps):
        jacobian = robot.root_physx_view.get_jacobians()[
            :, ee_jacobian_index, :, joint_ids
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

        joint_pos_des = ik_controller.compute(
            ee_pos_b,
            ee_quat_b,
            jacobian,
            joint_pos,
        )

        # Respect the articulation's joint limits.
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

        current_ee_pos_w = robot.data.body_pose_w[:, ee_body_id, 0:3]
        error = torch.linalg.vector_norm(desired_pos_w - current_ee_pos_w, dim=-1)
        minimum_error = torch.minimum(minimum_error, error)

        if step % 60 == 0:
            print(
                f"[IK] step={step:04d} error={error.item():.4f} m "
                f"minimum={minimum_error.item():.4f} m",
                flush=True,
            )

        if error.item() < args_cli.success_threshold:
            print(
                f"[IK] SUCCESS at step {step}: error={error.item():.4f} m",
                flush=True,
            )
            reached = True
            break

    if not reached:
        print(
            f"[IK] TIMEOUT: final error={error.item():.4f} m, "
            f"minimum error={minimum_error.item():.4f} m",
            flush=True,
        )

    # Keep the final pose visible briefly.
    for _ in range(240):
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(sim_dt)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
