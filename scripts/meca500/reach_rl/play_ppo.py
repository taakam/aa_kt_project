"""Visualize a trained Meca500 PPO checkpoint."""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained Meca500 reach PPO policy.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--real_time", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from ppo_cfg import MecaReachPPORunnerCfg
from reach_env import MecaReachEnv, MecaReachEnvCfg


def main() -> None:
    env_cfg = MecaReachEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 10.0
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    agent_cfg = MecaReachPPORunnerCfg()
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    env = MecaReachEnv(env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    dt = env.unwrapped.step_dt

    print(f"[PPO] loaded checkpoint: {args_cli.checkpoint}", flush=True)

    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        if args_cli.real_time:
            remaining = dt - (time.time() - start)
            if remaining > 0:
                time.sleep(remaining)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
