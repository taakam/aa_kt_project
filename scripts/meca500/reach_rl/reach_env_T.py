from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


# This file lives at scripts/meca500/reach_rl/reach_env.py.
REPO_ROOT = Path(__file__).resolve().parents[3]
MECA_USD = REPO_ROOT / "mecademic_description" / "urdf" / "meca_arm_only" / "meca_arm_end_effector.usd"
DESK_USD = REPO_ROOT / "mecademic_description" / "desk.usd"

ARM_BASE_POS = (-0.649353, -0.084213, 0.911893)
EE_BODY_NAME = "meca_axis_6_link"


@configclass
class MecaReachEnvCfg(DirectRLEnvCfg):
    """Meca500 joint-space reaching task for PPO."""

    # Environment interface.
    decimation = 4
    episode_length_s = 5.0
    action_space = 6
    observation_space = 21
    state_space = 0

    # 120 Hz physics and 30 Hz policy.
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=MECA_USD.as_posix()),
        init_state=ArticulationCfg.InitialStateCfg(pos=ARM_BASE_POS),
        actuators={
            "meca_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=400.0,
                damping=40.0,
            ),
        },
    )

    # Policy action: normalized joint-position increment.
    action_scale = 0.01
    required_hold_steps = 15

    # Target sphere location in each environment's local coordinates.
    target_x_range = (-0.55, -0.40)
    target_y_range = (-0.16, 0.00)
    target_z = 0.922

    # The actual reach goal is above the sphere to avoid driving into the desk.
    approach_height = 0.03

    success_threshold = 0.012

    # Reward weights.
    progress_reward_scale = 40.0
    distance_reward_scale = 1.0
    success_bonus = 10.0
    action_penalty_scale = 0.01
    joint_velocity_penalty_scale = 0.002


class MecaReachEnv(DirectRLEnv):
    cfg: MecaReachEnvCfg

    def __init__(self, cfg: MecaReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, self._joint_names = self.robot.find_joints(".*")
        self._ee_body_ids, self._ee_body_names = self.robot.find_bodies(EE_BODY_NAME)
        if len(self._ee_body_ids) != 1:
            raise RuntimeError(
                f"Expected exactly one body named {EE_BODY_NAME!r}; "
                f"found {self._ee_body_names} with IDs {self._ee_body_ids}."
            )
        self._ee_body_id = self._ee_body_ids[0]

        self.actions = torch.zeros((self.num_envs, 6), device=self.device)
        self._joint_targets = self.robot.data.default_joint_pos.clone()
        self._target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._success_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/ReachTargets",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.1, 0.1),
                    ),
                ),
            },
        )
        self._target_markers = VisualizationMarkers(marker_cfg)

        print(f"[reach-env] joints: {self._joint_names}", flush=True)
        print(
            f"[reach-env] end effector: {self._ee_body_names[0]} "
            f"(body ID {self._ee_body_id})",
            flush=True,
        )

    @property
    def goal_pos_w(self) -> torch.Tensor:
        """Cartesian goal used for learning, above the visible target sphere."""
        goal = self._target_pos_w.clone()
        goal[:, 2] += self.cfg.approach_height
        return goal

    def _setup_scene(self) -> None:
        if not MECA_USD.is_file():
            raise FileNotFoundError(f"Meca500 USD not found: {MECA_USD}")
        if not DESK_USD.is_file():
            raise FileNotFoundError(f"Desk USD not found: {DESK_USD}")

        self.robot = Articulation(self.cfg.robot_cfg)

        desk_cfg = sim_utils.UsdFileCfg(
            usd_path=DESK_USD.as_posix(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        desk_cfg.func("/World/envs/env_0/Environment", desk_cfg)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)
        self._joint_targets = self._joint_targets + self.cfg.action_scale * self.actions
        limits = self.robot.data.joint_pos_limits
        self._joint_targets = torch.clamp(
            self._joint_targets,
            min=limits[..., 0],
            max=limits[..., 1],
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_targets)

    def _get_observations(self) -> dict:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_id]
        target_delta = self.goal_pos_w - ee_pos_w

        obs = torch.cat(
            (
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
                target_delta,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_id]
        distance = torch.linalg.vector_norm(self.goal_pos_w - ee_pos_w, dim=-1)
        progress = self._previous_distance - distance
        inside_target = distance < self.cfg.success_threshold
        action_mag = torch.sum(self.actions.square(), dim=-1)
        joint_speed = torch.sum(self.robot.data.joint_vel.square(), dim=-1)
        near = torch.where(distance < 0.03, 5.0, 1.0)
        reward = (
            self.cfg.progress_reward_scale * progress
            + self.cfg.distance_reward_scale * torch.exp(-20.0 * distance)
            + self.cfg.success_bonus * inside_target.float()
            - self.cfg.action_penalty_scale * near * action_mag
            - self.cfg.joint_velocity_penalty_scale * near * joint_speed
        )
        self._previous_distance = distance.detach()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_id]
        distance = torch.linalg.vector_norm(self.goal_pos_w - ee_pos_w, dim=-1)
        inside = distance < self.cfg.success_threshold
        self._success_counter = torch.where(inside,self._success_counter+1,torch.zeros_like(self._success_counter))
        reached = self._success_counter >= self.cfg.required_hold_steps
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return reached, timed_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Log final episode statistics before clearing buffers.
        if len(env_ids) > 0:
            ee_pos_w = self.robot.data.body_pos_w[env_ids, self._ee_body_id]
            final_distance = torch.linalg.vector_norm(
                self.goal_pos_w[env_ids] - ee_pos_w,
                dim=-1,
            )
            success = final_distance < self.cfg.success_threshold
            self.extras["log"] = {
                "Episode/final_distance": final_distance.mean(),
                "Episode/success_rate": success.float().mean(),
            }

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        self._joint_targets[env_ids] = joint_pos
        self.actions[env_ids] = 0.0
        self._success_counter[env_ids] = 0

        num_resets = len(env_ids)
        target_local = torch.empty((num_resets, 3), device=self.device)
        target_local[:, 0].uniform_(*self.cfg.target_x_range)
        target_local[:, 1].uniform_(*self.cfg.target_y_range)
        target_local[:, 2] = self.cfg.target_z

        # Convert each local target to world coordinates for parallel environments.
        self._target_pos_w[env_ids] = target_local + self.scene.env_origins[env_ids]

        ee_pos_w = self.robot.data.body_pos_w[env_ids, self._ee_body_id]
        self._previous_distance[env_ids] = torch.linalg.vector_norm(
            self.goal_pos_w[env_ids] - ee_pos_w,
            dim=-1,
        )

        self._target_markers.visualize(translations=self._target_pos_w)

    def _post_physics_step(self) -> None:
        self._target_markers.visualize(translations=self._target_pos_w)
