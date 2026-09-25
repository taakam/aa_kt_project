"""
Deformable-oocyte + kinematic pipette indentation test
Isaac Lab 2.3 / Isaac Sim 5.1

Key change from the previous version:
The pipette is now an Isaac Lab RigidObject and is moved through the PhysX
tensor API (`write_root_pose_to_sim`) instead of changing a USD xform op after
simulation startup. This ensures the kinematic collision shape actually moves
in PhysX.

Sequence:
    WAIT -> APPROACH -> INDENT -> HOLD -> RETRACT -> repeat
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deformable oocyte pipette contact test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import (
    DeformableObject,
    DeformableObjectCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.sim import SimulationContext


# =============================================================================
# Oocyte
# =============================================================================

OOCYTE_RADIUS = 0.05          # m -- enlarged numerical proxy
YOUNGS_MODULUS = 7_340.0      # Pa
POISSON_RATIO = 0.04
OOCYTE_DENSITY = 1000.0       # kg/m^3


# =============================================================================
# Pipette proxy
# =============================================================================

PIPETTE_RADIUS = 0.006        # m
PIPETTE_HEIGHT = 0.10         # cylindrical section length

# USD/PhysX capsule total end-to-end length is approximately height + 2*radius.
PIPETTE_HALF_TOTAL_LENGTH = 0.5 * PIPETTE_HEIGHT + PIPETTE_RADIUS

# Sphere +X surface is at x = +OOCYTE_RADIUS.
# Pipette approaches from +X, pointing toward -X.
CONTACT_CENTER_X = OOCYTE_RADIUS + PIPETTE_HALF_TOTAL_LENGTH

# Start with a clear gap.
PIPETTE_START_X = CONTACT_CENTER_X + 0.05

# Push 10% of the oocyte radius beyond nominal surface contact.
INDENT_FRACTION = 0.10
INDENT_DEPTH = INDENT_FRACTION * OOCYTE_RADIUS
PIPETTE_INDENT_X = CONTACT_CENTER_X - INDENT_DEPTH

# Holding pipette: stationary on the -X side.
# Its near tip sits at the nominal -X oocyte surface.
HOLDING_CONTACT_CENTER_X = -(OOCYTE_RADIUS + PIPETTE_HALF_TOTAL_LENGTH)

# Approximate suction/holding constraint:
# pin a small set of FEM simulation nodes closest to the -X pole.
# This is a first proxy for the holding pipette's aspiration grip.
NUM_PINNED_NODES = 8

# Local indentation metric:
# track a small patch of FEM nodes nearest the +X pole (injection side).
NUM_INJECTION_SURFACE_NODES = 12

# Timing
WAIT_TIME = 1.0
APPROACH_TIME = 2.5
INDENT_TIME = 1.0
HOLD_TIME = 1.5
RETRACT_TIME = 2.0

# Capsule local axis is Z. Rotate +90 deg around Y so it lies along X.
PIPETTE_QUAT_WXYZ = (0.70710678, 0.0, 0.70710678, 0.0)


# =============================================================================
# Scene
# =============================================================================

def design_scene():
    light_cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(1.0, 1.0, 1.0),
    )
    light_cfg.func("/World/Light", light_cfg)

    # ----------------------------- oocyte ------------------------------------
    deformable_props = sim_utils.DeformableBodyPropertiesCfg(
        rest_offset=0.0,
        contact_offset=0.001,
        solver_position_iteration_count=12,
        collision_simplification=True,
        collision_simplification_remeshing=True,
        collision_simplification_remeshing_resolution=0,
        collision_simplification_target_triangle_count=0,
        collision_simplification_force_conforming=True,
    )

    deformable_material = sim_utils.DeformableBodyMaterialCfg(
        density=OOCYTE_DENSITY,
        youngs_modulus=YOUNGS_MODULUS,
        poissons_ratio=POISSON_RATIO,
        dynamic_friction=0.2,
        elasticity_damping=0.005,
        damping_scale=1.0,
    )

    oocyte_cfg = DeformableObjectCfg(
        prim_path="/World/Oocyte",
        spawn=sim_utils.MeshSphereCfg(
            radius=OOCYTE_RADIUS,
            deformable_props=deformable_props,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.45, 0.10),
                opacity=1.0,
            ),
            physics_material=deformable_material,
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
        debug_vis=False,
    )
    oocyte = DeformableObject(oocyte_cfg)

    # ----------------------------- pipette -----------------------------------
    pipette_cfg = RigidObjectCfg(
        prim_path="/World/Pipette",
        spawn=sim_utils.CapsuleCfg(
            radius=PIPETTE_RADIUS,
            height=PIPETTE_HEIGHT,
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
                diffuse_color=(0.20, 0.65, 1.0),
                opacity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PIPETTE_START_X, 0.0, 0.0),
            rot=PIPETTE_QUAT_WXYZ,
        ),
    )
    pipette = RigidObject(pipette_cfg)

    # -------------------------- holding pipette -------------------------------
    # Stationary kinematic support on the opposite (-X) side of the oocyte.
    holding_pipette_cfg = RigidObjectCfg(
        prim_path="/World/HoldingPipette",
        spawn=sim_utils.CapsuleCfg(
            radius=PIPETTE_RADIUS,
            height=PIPETTE_HEIGHT,
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
            pos=(HOLDING_CONTACT_CENTER_X, 0.0, 0.0),
            rot=PIPETTE_QUAT_WXYZ,
        ),
    )
    holding_pipette = RigidObject(holding_pipette_cfg)

    return {
        "oocyte": oocyte,
        "pipette": pipette,
        "holding_pipette": holding_pipette,
    }


# =============================================================================
# Motion
# =============================================================================

def smoothstep(a):
    a = max(0.0, min(1.0, a))
    return a * a * (3.0 - 2.0 * a)


def interp(a, b, alpha):
    s = smoothstep(alpha)
    return a + (b - a) * s


def pipette_command(t):
    """Return center-X command and phase; sequence repeats automatically."""

    cycle = WAIT_TIME + APPROACH_TIME + INDENT_TIME + HOLD_TIME + RETRACT_TIME
    tc = t % cycle

    a = WAIT_TIME
    b = a + APPROACH_TIME
    c = b + INDENT_TIME
    d = c + HOLD_TIME

    if tc < a:
        return PIPETTE_START_X, "WAIT"

    if tc < b:
        return interp(PIPETTE_START_X, CONTACT_CENTER_X, (tc-a)/APPROACH_TIME), "APPROACH"

    if tc < c:
        return interp(CONTACT_CENTER_X, PIPETTE_INDENT_X, (tc-b)/INDENT_TIME), "INDENT"

    if tc < d:
        return PIPETTE_INDENT_X, "HOLD"

    return interp(PIPETTE_INDENT_X, PIPETTE_START_X, (tc-d)/RETRACT_TIME), "RETRACT"


# =============================================================================
# Holding-patch constraint
# =============================================================================

def make_holding_kinematic_target(oocyte):
    """Create a kinematic target that pins a small patch at the -X pole.

    Isaac Lab deformable kinematic targets have shape [N, V, 4]:
        [..., :3] = target node position
        [..., 3]  = 0.0 for constrained nodes, 1.0 for free nodes

    We choose the NUM_PINNED_NODES simulation vertices closest to the ideal
    holding-pipette contact point (-R, 0, 0) and keep them at their initial
    positions throughout the simulation.
    """
    targets = oocyte.data.nodal_kinematic_target.clone()

    # Start with every node free.
    targets[..., :3] = oocyte.data.nodal_pos_w.clone()
    targets[..., 3] = 1.0

    node_pos = oocyte.data.nodal_pos_w[0]
    ideal_hold_point = torch.tensor(
        [-OOCYTE_RADIUS, 0.0, 0.0],
        dtype=node_pos.dtype,
        device=node_pos.device,
    )

    distances = torch.linalg.vector_norm(node_pos - ideal_hold_point, dim=-1)

    k = min(NUM_PINNED_NODES, node_pos.shape[0])
    pinned_ids = torch.topk(distances, k=k, largest=False).indices

    # Keep the selected patch fixed at its initial world-frame locations.
    pinned_positions = node_pos[pinned_ids].clone()
    targets[0, pinned_ids, :3] = pinned_positions
    targets[0, pinned_ids, 3] = 0.0

    return targets, pinned_ids, pinned_positions


# =============================================================================
# Injection-side local deformation tracking
# =============================================================================

def make_injection_surface_reference(oocyte):
    """Select FEM nodes nearest the +X pole and store their initial positions.

    These nodes are NOT constrained. They are only used to measure local
    indentation caused by the injection pipette.
    """
    node_pos = oocyte.data.nodal_pos_w[0]

    ideal_injection_point = torch.tensor(
        [OOCYTE_RADIUS, 0.0, 0.0],
        dtype=node_pos.dtype,
        device=node_pos.device,
    )

    distances = torch.linalg.vector_norm(
        node_pos - ideal_injection_point,
        dim=-1,
    )

    k = min(NUM_INJECTION_SURFACE_NODES, node_pos.shape[0])
    injection_ids = torch.topk(
        distances,
        k=k,
        largest=False,
    ).indices

    initial_positions = node_pos[injection_ids].clone()
    initial_mean_x = initial_positions[:, 0].mean().clone()
    initial_max_x = initial_positions[:, 0].max().clone()

    return injection_ids, initial_positions, initial_mean_x, initial_max_x


# =============================================================================
# Diagnostics
# =============================================================================

def deformation_metrics(
    oocyte,
    injection_ids,
    injection_initial_positions,
    injection_initial_mean_x,
    injection_initial_max_x,
):
    pos = oocyte.data.nodal_pos_w
    vel = oocyte.data.nodal_vel_w

    pmin = pos.amin(dim=1)
    pmax = pos.amax(dim=1)
    extents = pmax - pmin

    center = pos.mean(dim=1)
    radial = torch.linalg.vector_norm(pos - center[:, None, :], dim=-1)

    # Local injection-side deformation:
    # positive indentation means the tracked +X surface patch moved inward (-X).
    local_now = pos[0, injection_ids]
    local_mean_x = local_now[:, 0].mean()
    local_max_x = local_now[:, 0].max()

    local_mean_indentation = injection_initial_mean_x - local_mean_x
    local_peak_indentation = injection_initial_max_x - local_max_x

    # RMS displacement of the tracked patch gives a general local deformation
    # magnitude, including Y/Z motion as the sphere bulges around the pipette.
    local_displacement = local_now - injection_initial_positions
    local_rms_displacement = torch.sqrt(
        torch.mean(torch.sum(local_displacement**2, dim=-1))
    )

    return {
        "extents": extents,
        "radial_std": radial.std(dim=1),
        "vmax": torch.linalg.vector_norm(vel, dim=-1).max(dim=1).values,
        "center": center,
        "xmin": pmin[:, 0],
        "xmax": pmax[:, 0],
        "x_extent_change": extents[:, 0] - 2.0 * OOCYTE_RADIUS,
        "local_mean_indentation": local_mean_indentation,
        "local_peak_indentation": local_peak_indentation,
        "local_rms_displacement": local_rms_displacement,
        "local_mean_x": local_mean_x,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        device=args_cli.device,
        gravity=(0.0, 0.0, 0.0),
    )
    sim = SimulationContext(sim_cfg)

    entities = design_scene()

    # Clear side/oblique view of the contact point.
    sim.set_camera_view(
        eye=(0.20, -0.22, 0.10),
        target=(0.035, 0.0, 0.0),
    )

    print("\n[INFO] Initializing deformable oocyte + kinematic pipette...", flush=True)
    sim.reset()

    oocyte = entities["oocyte"]
    pipette = entities["pipette"]
    holding_pipette = entities["holding_pipette"]

    # Create the FEM holding patch after PhysX has initialized the deformable
    # simulation mesh. The selected vertices are fixed in world space.
    holding_kinematic_target, pinned_ids, pinned_positions = make_holding_kinematic_target(oocyte)

    # Create a reference patch on the +X side for local indentation measurement.
    (
        injection_surface_ids,
        injection_initial_positions,
        injection_initial_mean_x,
        injection_initial_max_x,
    ) = make_injection_surface_reference(oocyte)

    dt = sim.get_physics_dt()
    device = sim.device

    quat = torch.tensor(
        PIPETTE_QUAT_WXYZ, dtype=torch.float32, device=device
    ).view(1, 4)

    holding_pos = torch.tensor(
        [[HOLDING_CONTACT_CENTER_X, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    holding_pose = torch.cat((holding_pos, quat), dim=-1)

    print("[INFO] Initialization successful.", flush=True)
    print(f"[INFO] Oocyte surface (+X)       : {OOCYTE_RADIUS:.4f} m", flush=True)
    print(f"[INFO] Pipette total half-length : {PIPETTE_HALF_TOTAL_LENGTH:.4f} m", flush=True)
    print(f"[INFO] Contact center X          : {CONTACT_CENTER_X:.4f} m", flush=True)
    print(f"[INFO] Indent center X           : {PIPETTE_INDENT_X:.4f} m", flush=True)
    print(f"[INFO] Target indentation        : {INDENT_DEPTH:.4f} m", flush=True)
    print(f"[INFO] Holding pipette center X  : {HOLDING_CONTACT_CENTER_X:.4f} m", flush=True)
    print("[INFO] Holding pipette remains stationary at the -X surface.", flush=True)
    print(f"[INFO] FEM holding patch nodes     : {pinned_ids.detach().cpu().tolist()}", flush=True)
    print(f"[INFO] Number of pinned FEM nodes : {len(pinned_ids)}", flush=True)
    print("[INFO] Pinned nodes emulate aspiration/holding rather than a simple wall.", flush=True)
    print(f"[INFO] Injection surface nodes     : {injection_surface_ids.detach().cpu().tolist()}", flush=True)
    print(f"[INFO] Number of tracked +X nodes : {len(injection_surface_ids)}", flush=True)
    print("[INFO] Local indentation is measured from this unconstrained +X surface patch.", flush=True)
    print("[INFO] Sequence repeats automatically.", flush=True)

    step = 0
    previous_phase = None

    while simulation_app.is_running():
        t = step * dt
        x_cmd, phase = pipette_command(t)

        # IMPORTANT: drive the kinematic rigid body through the PhysX tensor API.
        pos = torch.tensor(
            [[x_cmd, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        pose = torch.cat((pos, quat), dim=-1)
        pipette.write_root_pose_to_sim(pose)

        # Keep the visible/support holding pipette fixed.
        holding_pipette.write_root_pose_to_sim(holding_pose)

        # Apply the actual holding constraint to the deformable simulation mesh.
        # The pinned -X patch stays at its initial position; all other nodes are free.
        holding_kinematic_target[0, pinned_ids, :3] = pinned_positions
        holding_kinematic_target[0, pinned_ids, 3] = 0.0
        oocyte.write_nodal_kinematic_target_to_sim(holding_kinematic_target)

        oocyte.write_data_to_sim()

        sim.step()

        oocyte.update(dt)
        pipette.update(dt)
        holding_pipette.update(dt)

        if phase != previous_phase:
            print(f"\n[PHASE] {phase} at t={t:.2f}s", flush=True)
            previous_phase = phase

        if step % 30 == 0:
            metrics = deformation_metrics(
                oocyte,
                injection_surface_ids,
                injection_initial_positions,
                injection_initial_mean_x,
                injection_initial_max_x,
            )

            actual_x = pipette.data.root_pos_w[0, 0].item()
            holding_x = holding_pipette.data.root_pos_w[0, 0].item()

            # Injection pipette approaches from +X: its near end is center - half-length.
            actual_tip_x = actual_x - PIPETTE_HALF_TOTAL_LENGTH

            # Holding pipette is on -X: its near end is center + half-length.
            holding_tip_x = holding_x + PIPETTE_HALF_TOTAL_LENGTH

            xmax = metrics["xmax"][0].item()
            xmin = metrics["xmin"][0].item()
            inj_gap = actual_tip_x - xmax
            hold_gap = xmin - holding_tip_x

            ext = metrics["extents"][0]
            center = metrics["center"][0]
            center_shift_x = center[0].item()
            x_extent_change = metrics["x_extent_change"][0].item()

            local_mean_indent = metrics["local_mean_indentation"].item()
            local_peak_indent = metrics["local_peak_indentation"].item()
            local_rms = metrics["local_rms_displacement"].item()

            print(
                f"t={t:5.2f}s | {phase:8s} | "
                f"cmd_x={x_cmd:+.4f} | inj_tip={actual_tip_x:+.4f} | inj_gap={inj_gap:+.5f} | "
                f"hold_tip={holding_tip_x:+.4f} | hold_gap={hold_gap:+.5f} | "
                f"center_dx={center_shift_x:+.5f} | "
                f"local_indent={local_mean_indent:+.5f} | "
                f"peak_indent={local_peak_indent:+.5f} | "
                f"local_rms={local_rms:.5f} | "
                f"dX={x_extent_change:+.5f} | "
                f"XYZ=({ext[0].item():.4f}, {ext[1].item():.4f}, {ext[2].item():.4f}) | "
                f"radial_std={metrics['radial_std'][0].item():.5f} | "
                f"vmax={metrics['vmax'][0].item():.5f}",
                flush=True,
            )

        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
