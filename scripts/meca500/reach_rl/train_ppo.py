"""Train PPO on the Meca500 reaching environment with RSL-RL."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Meca500 reach PPO.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=500)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from ppo_cfg import MecaReachPPORunnerCfg
from reach_env import MecaReachEnv, MecaReachEnvCfg


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    env_cfg = MecaReachEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    agent_cfg = MecaReachPPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.run_name = args_cli.run_name
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{agent_cfg.run_name}" if agent_cfg.run_name else ""
    log_dir = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, timestamp + suffix)
    )
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    env_cfg.log_dir = log_dir

    print(f"[PPO] environments: {env_cfg.scene.num_envs}", flush=True)
    print(f"[PPO] iterations: {agent_cfg.max_iterations}", flush=True)
    print(f"[PPO] logs: {log_dir}", flush=True)

    env = MecaReachEnv(env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
