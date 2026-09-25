from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul


# This file lives at scripts/meca500/reach_rl/reach_env.py.
REPO_ROOT = Path(__file__).resolve().parents[3]
MECA_USD = REPO_ROOT / "mecademic_description" / "urdf" / "meca_arm_only" / "meca_arm_end_effector.usd"
DESK_USD = REPO_ROOT / "mecademic_description" / "desk.usd"

ARM_BASE_POS = (-0.649353, -0.084213, 0.911893)
EE_BODY_NAME = "meca_axis_6_link"
PIPETTE_TIP_OFFSET = (0.02382, -0.0029, -0.19674)

OOCYTE_RADIUS = 0.010
YOUNGS_MODULUS = 7_340.0
POISSON_RATIO = 0.04
OOCYTE_DENSITY = 1000.0
NUM_PINNED_NODES = 6

# Aspiration proxy for the side holding pipette.
# The selected FEM patch is pulled slightly into the holder mouth instead of
# being hard-pinned at the undeformed oocyte surface.
# Aspiration is disabled in this baseline. We first test:
# dish support + side holder contact + deformable oocyte.
ASPIRATION_DEPTH = 0.0

# Because we do not yet model buoyancy from the culture medium, using full
# Earth gravity would make the oocyte artificially heavy. A small effective
# gravity is used only to settle it gently onto the dish.
EFFECTIVE_GRAVITY_Z = -0.5     # m/s^2

# Simple rigid dish-floor proxy.
DISH_THICKNESS = 0.004         # m
DISH_HALF_EXTENT_XY = 0.18     # m
DISH_SURFACE_Z_OFFSET = 0.0    # dish top surface is at target_z

# Scaled horizontal side-holding pipette with aspiration proxy.
HOLDING_PIPETTE_RADIUS = 0.0015
HOLDING_PIPETTE_HEIGHT = 0.025
HOLDING_PIPETTE_HALF_TOTAL_LENGTH = 0.5 * HOLDING_PIPETTE_HEIGHT + HOLDING_PIPETTE_RADIUS

# Capsule local axis is Z; +90 deg around Y aligns the holder horizontally along X.
HOLDING_PIPETTE_QUAT_WXYZ = (0.70710678, 0.0, 0.70710678, 0.0)

# Physical proxy collider for the injection pipette.
# It is aligned with the real pipette axis and terminates exactly at the
# PIPETTE_TIP_OFFSET point used by the reward.
TIP_COLLIDER_RADIUS = 0.0005   # 0.5 mm
TIP_COLLIDER_HEIGHT = 0.012    # 12 mm cylindrical section
TIP_COLLIDER_HALF_TOTAL_LENGTH = 0.5 * TIP_COLLIDER_HEIGHT + TIP_COLLIDER_RADIUS

# Stronger aspiration proxy now that the oocyte is supported by the dish.
# Only a tiny side patch is constrained; nodes are NOT pulled inward.
HOLDER_PINNED_NODES = 4

# Safety limits for the current scaled numerical oocyte.
MAX_SAFE_PENETRATION = 0.0010   # 1.0 mm
MAX_SAFE_NODE_SPEED = 0.05      # m/s



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
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, EFFECTIVE_GRAVITY_Z),
    )

    # Deformable FEM bodies cannot use Isaac Lab's normal PhysX replication
    # path. Each environment must build its own physics representation.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=2.0,
        replicate_physics=False,
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

    oocyte_cfg: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/Oocyte",
        spawn=sim_utils.MeshSphereCfg(
            radius=OOCYTE_RADIUS,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.0005,
                solver_position_iteration_count=12,
                collision_simplification=True,
                collision_simplification_remeshing=True,
                collision_simplification_remeshing_resolution=0,
                collision_simplification_target_triangle_count=0,
                collision_simplification_force_conforming=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.20, 0.05),
                opacity=1.0,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                density=OOCYTE_DENSITY,
                youngs_modulus=YOUNGS_MODULUS,
                poissons_ratio=POISSON_RATIO,
                dynamic_friction=0.2,
                elasticity_damping=0.005,
                damping_scale=1.0,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        debug_vis=False,
    )

    holding_pipette_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HoldingPipette",
        spawn=sim_utils.CapsuleCfg(
            radius=HOLDING_PIPETTE_RADIUS,
            height=HOLDING_PIPETTE_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.25, 1.0, 0.35),
                opacity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=HOLDING_PIPETTE_QUAT_WXYZ,
        ),
    )

    dish_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Dish",
        spawn=sim_utils.CuboidCfg(
            size=(
                2.0 * DISH_HALF_EXTENT_XY,
                2.0 * DISH_HALF_EXTENT_XY,
                DISH_THICKNESS,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.0005,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.35, 0.45, 0.55),
                opacity=0.35,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
    )

    tip_collider_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TipCollider",
        spawn=sim_utils.CapsuleCfg(
            radius=TIP_COLLIDER_RADIUS,
            height=TIP_COLLIDER_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.0003,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 1.0),
                opacity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
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
        self._holding_kinematic_target = None
        self._pinned_ids = None
        self._pinned_offsets = None
        self._pipette_tip_offset = torch.tensor(
            PIPETTE_TIP_OFFSET, dtype=torch.float32, device=self.device
        )

        # Local direction from the end-effector body origin toward the physical
        # pipette tip. The capsule's native axis is +Z, so compute a fixed local
        # quaternion that rotates +Z onto this direction.
        self._pipette_axis_local = self._pipette_tip_offset / torch.linalg.vector_norm(
            self._pipette_tip_offset
        )

        z_axis = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32, device=self.device)
        dot = torch.dot(z_axis, self._pipette_axis_local)
        cross = torch.cross(z_axis, self._pipette_axis_local, dim=0)

        if dot < -0.9999:
            # 180-degree fallback around X.
            self._tip_collider_local_quat = torch.tensor(
                (0.0, 1.0, 0.0, 0.0), dtype=torch.float32, device=self.device
            )
        else:
            q = torch.cat(
                (
                    (1.0 + dot).view(1),
                    cross,
                )
            )
            self._tip_collider_local_quat = q / torch.linalg.vector_norm(q)
        self._wrist_cam_pos_offset = torch.tensor(
            (0.07669, -0.00202, 0.00198), dtype=torch.float32, device=self.device
        )
        self._wrist_cam_quat_offset = torch.tensor(
            (0.690260, 0.080374, 0.090543, -0.713325),
            dtype=torch.float32,
            device=self.device,
        )
        self._camera_debug_printed = False

        print(f"[reach-env] joints: {self._joint_names}", flush=True)
        print(f"[reach-env] pipette tip offset: {PIPETTE_TIP_OFFSET}", flush=True)
        print(
            f"[reach-env] deformable oocyte: R={OOCYTE_RADIUS:.3f} m, "
            f"E={YOUNGS_MODULUS/1000:.2f} kPa, nu={POISSON_RATIO:.3f}",
            flush=True,
        )
        print(
            f"[reach-env] side holder: radius={HOLDING_PIPETTE_RADIUS:.4f} m, "
            f"height={HOLDING_PIPETTE_HEIGHT:.4f} m",
            flush=True,
        )
        print(
            f"[reach-env] dish support enabled, effective gravity z={EFFECTIVE_GRAVITY_Z:.2f} m/s^2",
            flush=True,
        )
        print(
            f"[reach-env] pipette capsule collider: radius={TIP_COLLIDER_RADIUS:.4f} m, "
            f"height={TIP_COLLIDER_HEIGHT:.4f} m",
            flush=True,
        )
        print(
            f"[reach-env] end effector: {self._ee_body_names[0]} "
            f"(body ID {self._ee_body_id})",
            flush=True,
        )

    @property
    def goal_pos_w(self) -> torch.Tensor:
        """Phase-A goal: reach the top (+Z) surface of each deformable oocyte."""
        goal = self._target_pos_w.clone()
        goal[:, 2] += OOCYTE_RADIUS + self.cfg.approach_height
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
        self.oocyte = DeformableObject(self.cfg.oocyte_cfg)
        self.holding_pipette = RigidObject(self.cfg.holding_pipette_cfg)
        self.dish = RigidObject(self.cfg.dish_cfg)
        self.tip_collider = RigidObject(self.cfg.tip_collider_cfg)

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
        self.scene.deformable_objects["oocyte"] = self.oocyte
        self.scene.rigid_objects["holding_pipette"] = self.holding_pipette
        self.scene.rigid_objects["dish"] = self.dish
        self.scene.rigid_objects["tip_collider"] = self.tip_collider

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)

    def _initialize_holding_patch(self) -> None:
        """Pin a tiny FEM patch at the side-holder mouth.

        This is an aspiration proxy, not a pressure model. The rigid dish
        supports the oocyte vertically while this small patch resists lateral
        translation during contact with the injection pipette.
        """
        node_pos = self.oocyte.data.nodal_pos_w
        targets = self.oocyte.data.nodal_kinematic_target.clone()
        targets[..., :3] = node_pos.clone()
        targets[..., 3] = 1.0

        center = node_pos.mean(dim=1)
        ideal_hold_point = center + torch.tensor(
            [-OOCYTE_RADIUS, 0.0, 0.0],
            dtype=node_pos.dtype,
            device=self.device,
        ).view(1, 3)

        distances = torch.linalg.vector_norm(
            node_pos - ideal_hold_point[:, None, :], dim=-1
        )
        k = min(HOLDER_PINNED_NODES, node_pos.shape[1])
        self._pinned_ids = torch.topk(
            distances, k=k, largest=False, dim=1
        ).indices

        env_index = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        pinned_pos = node_pos[env_index, self._pinned_ids].clone()
        self._pinned_offsets = pinned_pos - center[:, None, :]

        targets[env_index, self._pinned_ids, :3] = pinned_pos
        targets[env_index, self._pinned_ids, 3] = 0.0
        self._holding_kinematic_target = targets

        print(
            f"[reach-env] aspiration proxy: {k} side FEM nodes constrained per env",
            flush=True,
        )

    def _apply_holding_constraint(self) -> None:
        if self._holding_kinematic_target is None:
            return

        env_index = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        pinned_world = self._target_pos_w[:, None, :] + self._pinned_offsets

        # Every node remains free except the tiny side patch.
        self._holding_kinematic_target[..., 3] = 1.0
        self._holding_kinematic_target[
            env_index, self._pinned_ids, :3
        ] = pinned_world
        self._holding_kinematic_target[
            env_index, self._pinned_ids, 3
        ] = 0.0

        self.oocyte.write_nodal_kinematic_target_to_sim(
            self._holding_kinematic_target
        )

    def _update_dish_pose(self) -> None:
        """Place the rigid dish floor under each randomized oocyte."""
        pos = self._target_pos_w.clone()
        # target_z is the intended top surface of the dish.
        pos[:, 2] = self.cfg.target_z - 0.5 * DISH_THICKNESS
        quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        quat[:, 0] = 1.0
        self.dish.write_root_pose_to_sim(torch.cat((pos, quat), dim=-1))

    def _update_holding_pipette_pose(self) -> None:
        """Place the visible holding pipette horizontally on the -X side."""
        pos = self._target_pos_w.clone()

        # The pipette mouth sits against the -X surface. The aspiration target
        # then pulls a small FEM patch inward along -X.
        hold_inset = 0.0001
        pos[:, 0] -= (
            OOCYTE_RADIUS
            + HOLDING_PIPETTE_HALF_TOTAL_LENGTH
            - hold_inset
        )

        quat = torch.tensor(
            HOLDING_PIPETTE_QUAT_WXYZ,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 4).expand(self.num_envs, -1)

        self.holding_pipette.write_root_pose_to_sim(
            torch.cat((pos, quat), dim=-1)
        )

    def _update_tip_collider_pose(self) -> None:
        """Align the capsule with the real pipette and end it at the PPO tip point."""
        link_quat_w = self.robot.data.body_quat_w[:, self._ee_body_id]
        tip_pos_w = self._get_pipette_tip_pos_w()

        # Direction from the end-effector body toward the pipette tip in world space.
        axis_local = self._pipette_axis_local.unsqueeze(0).expand(self.num_envs, -1)
        axis_w = quat_apply(link_quat_w, axis_local)

        # Capsule center sits behind the physical tip so its forward end terminates
        # exactly at PIPETTE_TIP_OFFSET.
        center_pos_w = tip_pos_w - TIP_COLLIDER_HALF_TOTAL_LENGTH * axis_w

        local_align = self._tip_collider_local_quat.unsqueeze(0).expand(self.num_envs, -1)
        collider_quat_w = quat_mul(link_quat_w, local_align)

        pose = torch.cat((center_pos_w, collider_quat_w), dim=-1)
        self.tip_collider.write_root_pose_to_sim(pose)

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
        self._update_dish_pose()
        self._update_holding_pipette_pose()
        self._apply_holding_constraint()
        self._update_tip_collider_pose()

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

    def _extract_oocyte_features(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return normalized [u, v, area] for the orange-red oocyte."""
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

        vision_features = self._extract_oocyte_features(rgb)

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

    def _debug_print_oocyte_motion(self) -> None:
        """Print normalized oocyte translation/deformation for env 0."""
        if self.num_envs < 1:
            return

        nodal_pos = self.oocyte.data.nodal_pos_w
        center = nodal_pos.mean(dim=1)
        center_error = center - self._target_pos_w

        pmin = nodal_pos.amin(dim=1)
        pmax = nodal_pos.amax(dim=1)
        extents = pmax - pmin

        translation_fraction = torch.linalg.vector_norm(
            center_error, dim=-1
        ) / OOCYTE_RADIUS

        x_extent_fraction = extents[:, 0] / (2.0 * OOCYTE_RADIUS)
        y_extent_fraction = extents[:, 1] / (2.0 * OOCYTE_RADIUS)
        z_extent_fraction = extents[:, 2] / (2.0 * OOCYTE_RADIUS)

        if int(self.episode_length_buf[0].item()) % 15 == 0:
            print(
                "[oocyte-debug] "
                f"translation/R={translation_fraction[0].item():.3f} | "
                f"extent/(2R)=("
                f"{x_extent_fraction[0].item():.3f}, "
                f"{y_extent_fraction[0].item():.3f}, "
                f"{z_extent_fraction[0].item():.3f})",
                flush=True,
            )

    def _get_rewards(self) -> torch.Tensor:
        """Original reach reward; no contact-safety metric in holder-debug mode."""
        tip_pos_w = self._get_pipette_tip_pos_w()
        distance = torch.linalg.vector_norm(
            self.goal_pos_w - tip_pos_w,
            dim=-1,
        )

        progress = self._previous_distance - distance
        self._previous_distance = distance.detach()

        inside_target = distance < self.cfg.success_threshold
        action_mag = torch.sum(torch.square(self.actions), dim=-1)
        joint_speed = torch.sum(
            torch.square(self.robot.data.joint_vel[:, self._joint_ids]),
            dim=-1,
        )

        near = torch.where(distance < 0.03, 5.0, 1.0)

        reward = (
            self.cfg.progress_reward_scale * progress
            + self.cfg.distance_reward_scale * torch.exp(-20.0 * distance)
            + self.cfg.success_bonus * inside_target.float()
            - self.cfg.action_penalty_scale * near * action_mag
            - self.cfg.joint_velocity_penalty_scale * near * joint_speed
        )

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Holder-debug mode: terminate only on episode timeout."""
        terminated = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timed_out

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

        # The 5-cm FEM proxy sits on the desk instead of intersecting it.
        target_local[:, 2] = self.cfg.target_z + OOCYTE_RADIUS
        new_centers_w = target_local + self.scene.env_origins[env_ids]
        self._target_pos_w[env_ids] = new_centers_w

        # Translate the undeformed FEM nodal state to the randomized center.
        default_nodal = self.oocyte.data.default_nodal_state_w[env_ids].clone()
        default_center = default_nodal[..., :3].mean(dim=1)
        delta = new_centers_w - default_center
        default_nodal[..., :3] += delta[:, None, :]
        self.oocyte.write_nodal_state_to_sim(default_nodal, env_ids=env_ids)

        # Rebuild the tiny side-holding patch after every deformable reset so
        # its target positions match the newly randomized oocyte location.
        self._initialize_holding_patch()

        self._update_dish_pose()
        self._update_holding_pipette_pose()
        self._apply_holding_constraint()
        self._update_tip_collider_pose()

        tip_pos_w = self._get_pipette_tip_pos_w()[env_ids]
        self._previous_distance[env_ids] = torch.linalg.vector_norm(
            self.goal_pos_w[env_ids] - tip_pos_w,
            dim=-1,
        )


    def _post_physics_step(self) -> None:
        # Debug the holder using normalized quantities rather than arbitrary
        # millimetre thresholds on the enlarged numerical oocyte.
        self._debug_print_oocyte_motion()
