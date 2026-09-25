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
MECA_USD = REPO_ROOT / "mecademic_description" / "urdf" / "meca_arm_only" / "meca_arm_only.usd"
DESK_USD = REPO_ROOT / "mecademic_description" / "desk.usd"

ARM_BASE_POS = (-0.649353, -0.084213, 0.911893)
EE_BODY_NAME = "meca_axis_6_link"


@configclass
class MecaReachEnvCfg(DirectRLEnvCfg):
    """First RL scaffold: joint-delta control toward a visible 3-D target."""

    # RL interface
    decimation = 4
    episode_length_s = 5.0
    action_space = 6
    observation_space = 21  # q(6) + qdot(6) + ee->target(3) + previous action(6)
    state_space = 0

    # Simulation: 120 Hz physics, 30 Hz policy.
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    # Start with one environment. Increase only after the smoke test works.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
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

    # Joint-position increment per policy step, in radians.
    action_scale = 0.025

    # Target is sampled near the current home end-effector position.
    target_x_range = (-0.55, -0.40)
    target_y_range = (-0.16, 0.00)

    target_z = 0.922

    success_threshold = 0.01
    success_bonus = 5.0
    action_penalty_scale = 0.002


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
        print(f"[reach-env] end effector: {self._ee_body_names[0]} (body ID {self._ee_body_id})", flush=True)

    def _setup_scene(self) -> None:
        if not MECA_USD.is_file():
            raise FileNotFoundError(f"Meca500 USD not found: {MECA_USD}")
        if not DESK_USD.is_file():
            raise FileNotFoundError(f"Desk USD not found: {DESK_USD}")

        self.robot = Articulation(self.cfg.robot_cfg)

        # Spawn the desk in the source environment before cloning.
        desk_cfg = sim_utils.UsdFileCfg(
            usd_path=DESK_USD.as_posix(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        desk_cfg.func("/World/envs/env_0/Environment", desk_cfg)

        # Clone env_0 into the other environment namespaces.
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        # Global ground and lighting.
        sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)
        self._joint_targets = self.robot.data.joint_pos + self.cfg.action_scale * self.actions

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
        target_delta = self._target_pos_w - ee_pos_w

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
        distance = torch.linalg.vector_norm(self._target_pos_w - ee_pos_w, dim=-1)

        # Dense reaching reward plus a discrete success bonus.
        reward = torch.exp(-20.0 * distance)
        reward += self.cfg.success_bonus * (distance < self.cfg.success_threshold)
        reward -= self.cfg.action_penalty_scale * torch.sum(self.actions.square(), dim=-1)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_id]
        distance = torch.linalg.vector_norm(self._target_pos_w - ee_pos_w, dim=-1)

        reached = distance < self.cfg.success_threshold
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return reached, timed_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

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

        # Use the current/home tool position as the centre of a small target box.
        num_resets = len(env_ids)
        target_x = torch.empty(num_resets, device=self.device).uniform_(
	    self.cfg.target_x_range[0],
            self.cfg.target_x_range[1],
        )
        target_y = torch.empty(num_resets, device=self.device).uniform_(
            self.cfg.target_y_range[0],
            self.cfg.target_y_range[1],
        )
        target_z = torch.full(
            (num_resets,),
            self.cfg.target_z,
            device=self.device,
        )
        self._target_pos_w[env_ids] = torch.stack(
            (target_x, target_y, target_z),
            dim=-1,
        )
        # VisualizationMarkers expects all marker positions, not a subset.
        self._target_markers.visualize(translations=self._target_pos_w)

    def _post_physics_step(self) -> None:
        # Keep target markers synchronized after resets.
        self._target_markers.visualize(translations=self._target_pos_w)
