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
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply


# This file lives at scripts/meca500/reach_rl/reach_env.py.
REPO_ROOT = Path(__file__).resolve().parents[3]
MECA_USD = REPO_ROOT / "mecademic_description" / "urdf" / "meca_arm_only" / "meca_arm_end_effector.usd"
DESK_USD = REPO_ROOT / "mecademic_description" / "desk.usd"

ARM_BASE_POS = (-0.649353, -0.084213, 0.911893)
EE_BODY_NAME = "meca_axis_6_link"
PIPETTE_TIP_OFFSET = (0.02382, -0.0029, -0.19674)


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
        clone_in_fabric=False,
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

    # Official Isaac Lab DirectRL TiledCamera pattern:
    # instantiate TiledCamera before clone_environments(), then register it.
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.14756,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            vertical_aperture=15.2908,
            clipping_range=(0.01, 10.0),
        ),
        width=128,
        height=128,
    )

    # Policy action: normalized joint-position increment.
    action_scale = 0.01

    # Target sphere location in each environment's local coordinates.
    target_x_range = (-0.55, -0.40)
    target_y_range = (-0.16, 0.00)
    target_z = 0.922

    # The actual reach goal is above the sphere to avoid driving into the desk.
    approach_height = 0.0

    success_threshold = 0.002

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
        self._pipette_tip_offset = torch.tensor(
            PIPETTE_TIP_OFFSET, dtype=torch.float32, device=self.device
        )
        self._wrist_cam_pos_offset = torch.tensor(
            (0.07669, -0.00202, 0.00198), dtype=torch.float32, device=self.device
        )
        self._wrist_cam_quat_offset = torch.tensor(
            (0.690260, 0.080374, 0.090543, -0.713325),
            dtype=torch.float32,
            device=self.device,
        )
        self._camera_debug_printed = False

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
        print(f"[reach-env] pipette tip offset: {PIPETTE_TIP_OFFSET}", flush=True)
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

        # Exact Isaac Lab DirectRL camera pattern (see cartpole_camera_env.py):
        # create articulation and TiledCamera BEFORE cloning.
        self.robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)

        desk_cfg = sim_utils.UsdFileCfg(
            usd_path=DESK_USD.as_posix(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        desk_cfg.func("/World/envs/env_0/Environment", desk_cfg)

        # Clone and replicate after robot + camera have been instantiated.
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Register both with InteractiveScene exactly like the official example.
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera

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

    def _get_pipette_tip_pos_w(self) -> torch.Tensor:
        """World position of the physical pipette tip for every environment."""
        link_pos_w = self.robot.data.body_pos_w[:, self._ee_body_id]
        link_quat_w = self.robot.data.body_quat_w[:, self._ee_body_id]
        tip_offset = self._pipette_tip_offset.unsqueeze(0).expand(self.num_envs, -1)
        return link_pos_w + quat_apply(link_quat_w, tip_offset)

    def _update_wrist_camera_pose(self) -> None:
        """Move each standalone tiled camera with meca_axis_6_link."""
        link_pos_w = self.robot.data.body_pos_w[:, self._ee_body_id]
        link_quat_w = self.robot.data.body_quat_w[:, self._ee_body_id]

        pos_offset = self._wrist_cam_pos_offset.unsqueeze(0).expand(self.num_envs, -1)
        quat_offset = self._wrist_cam_quat_offset.unsqueeze(0).expand(self.num_envs, -1)

        cam_pos_w = link_pos_w + quat_apply(link_quat_w, pos_offset)

        w1, x1, y1, z1 = link_quat_w.unbind(-1)
        w2, x2, y2, z2 = quat_offset.unbind(-1)
        cam_quat_w = torch.stack(
            (
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ),
            dim=-1,
        )

        self._tiled_camera.set_world_poses(
            positions=cam_pos_w,
            orientations=cam_quat_w,
            convention="usd",
        )

    def _extract_red_target_features(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return normalized [u, v, area] for the red target in each image."""
        image = rgb.to(torch.float32)
        if image.max() > 1.5:
            image = image / 255.0

        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]
        mask = (r > 0.45) & (r > 1.35 * g) & (r > 1.35 * b)

        weights = mask.to(torch.float32)
        area_px = weights.sum(dim=(1, 2))
        h, w = weights.shape[1], weights.shape[2]

        xs = torch.arange(w, device=self.device, dtype=torch.float32).view(1, 1, w)
        ys = torch.arange(h, device=self.device, dtype=torch.float32).view(1, h, 1)

        denom = area_px.clamp_min(1.0)
        u = (weights * xs).sum(dim=(1, 2)) / denom
        v = (weights * ys).sum(dim=(1, 2)) / denom

        u_norm = 2.0 * (u / float(max(w - 1, 1))) - 1.0
        v_norm = 2.0 * (v / float(max(h - 1, 1))) - 1.0
        area = area_px / float(h * w)

        visible = area_px > 0
        u_norm = torch.where(visible, u_norm, torch.zeros_like(u_norm))
        v_norm = torch.where(visible, v_norm, torch.zeros_like(v_norm))
        area = torch.where(visible, area, torch.zeros_like(area))

        return torch.stack((u_norm, v_norm, area), dim=-1)

    def _get_observations(self) -> dict:
        # Keep the tiled cameras rigidly attached to the wrist pose.
        self._update_wrist_camera_pose()

        # Official example accesses camera output directly from the TiledCamera.
        rgb = self._tiled_camera.data.output["rgb"]

        if not self._camera_debug_printed:
            print(
                "[wrist-camera] shape:", tuple(rgb.shape),
                "dtype:", rgb.dtype,
                "min:", rgb.min().item(),
                "max:", rgb.max().item(),
                flush=True,
            )
            self._camera_debug_printed = True

        vision_features = self._extract_red_target_features(rgb)

        # 6 joint pos + 6 joint vel + 3 visual features + 6 previous actions = 21
        obs = torch.cat(
            (
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
                vision_features,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        tip_pos_w = self._get_pipette_tip_pos_w()
        distance = torch.linalg.vector_norm(self.goal_pos_w - tip_pos_w, dim=-1)
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
        tip_pos_w = self._get_pipette_tip_pos_w()
        distance = torch.linalg.vector_norm(self.goal_pos_w - tip_pos_w, dim=-1)

        # End the episode immediately once the pipette tip reaches the
        # acceptable target region. No hold-time requirement.
        reached = distance < self.cfg.success_threshold
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return reached, timed_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Log final episode statistics before clearing buffers.
        if len(env_ids) > 0:
            tip_pos_w = self._get_pipette_tip_pos_w()[env_ids]
            final_distance = torch.linalg.vector_norm(
                self.goal_pos_w[env_ids] - tip_pos_w,
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

        num_resets = len(env_ids)
        target_local = torch.empty((num_resets, 3), device=self.device)
        target_local[:, 0].uniform_(*self.cfg.target_x_range)
        target_local[:, 1].uniform_(*self.cfg.target_y_range)
        target_local[:, 2] = self.cfg.target_z

        # Convert each local target to world coordinates for parallel environments.
        self._target_pos_w[env_ids] = target_local + self.scene.env_origins[env_ids]

        tip_pos_w = self._get_pipette_tip_pos_w()[env_ids]
        self._previous_distance[env_ids] = torch.linalg.vector_norm(
            self.goal_pos_w[env_ids] - tip_pos_w,
            dim=-1,
        )

        self._target_markers.visualize(translations=self._target_pos_w)

    def _post_physics_step(self) -> None:
        self._target_markers.visualize(translations=self._target_pos_w)
