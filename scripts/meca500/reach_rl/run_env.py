"""Launch the Meca500 reach environment and step it with random actions."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-test the Meca500 reach RL environment.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from reach_env import MecaReachEnv, MecaReachEnvCfg


def main() -> None:
    cfg = MecaReachEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    env = MecaReachEnv(cfg)
    env.reset()

    # Aim at the source environment's robot.
    base = torch.tensor((-0.649353, -0.084213, 0.911893))
    env.sim.set_camera_view(
        eye=tuple((base + torch.tensor((0.6, 0.6, 0.4))).tolist()),
        target=tuple((base + torch.tensor((0.0, 0.0, 0.15))).tolist()),
    )

    for step in range(args_cli.steps):
        # Tiny random actions are enough to verify the complete RL step pipeline.
        actions = 0.15 * torch.randn((env.num_envs, 6), device=env.device)
        obs, reward, terminated, truncated, info = env.step(actions)

        if step % 60 == 0:
            print(
                f"[reach-env] step={step:04d} "
                f"reward_mean={reward.mean().item():.3f} "
                f"obs_shape={tuple(obs['policy'].shape)}",
                flush=True,
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
