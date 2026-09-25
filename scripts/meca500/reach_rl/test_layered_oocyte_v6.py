"""
Standalone layered-oocyte mechanics experiment for Isaac Sim 5.1.

Goal
----
Test a mechanically separate:
  1) outer surface-deformable zona pellucida (ZP), and
  2) inner volume-deformable oocyte/ooplasm,

before integrating the model back into the PPO environment.

This deliberately uses the lower-level Omni PhysX deformable API because
Isaac Lab 2.3's high-level DeformableObject wrapper predates the split between
surface and volume deformables.

Run:
    ~/IsaacLab/isaaclab.sh -p test_layered_oocyte_v6_render_binding.py --enable_cameras

Do NOT run headless for the first test: this is a visual mechanics diagnostic.
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports that require Kit/Isaac Sim to be running.
import omni.usd
from omni.physx import get_physx_cooking_interface
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.sim import SimulationCfg, SimulationContext


# ---------------------------------------------------------------------------
# Scaled geometry.
# ---------------------------------------------------------------------------

OUTER_RADIUS = 0.0100            # 10 mm numerical proxy
INNER_RADIUS = 0.0078            # inner cell/ooplasm proxy
ZP_SURFACE_THICKNESS = 0.0010    # 1 mm numerical shell thickness

# Small radial clearance between the effective inner edge of the ZP shell and
# the inner deformable. This avoids starting the two deformables interpenetrating.
#
# OUTER_RADIUS - ZP_SURFACE_THICKNESS = 9 mm inner ZP surface
# INNER_RADIUS = 7.8 mm -> ~1.2 mm numerical perivitelline clearance.
CENTER = Gf.Vec3f(0.0, 0.0, 0.020)

# Material values are starting hypotheses, not a validated complete oocyte model.
ZP_YOUNGS = 7_340.0
ZP_POISSON = 0.04
ZP_DENSITY = 1000.0

# Intentionally much softer inner body for the first mechanical experiment.
# This value must later be calibrated from literature/experimental response.
INNER_YOUNGS = 1_000.0
INNER_POISSON = 0.45
INNER_DENSITY = 1000.0

# Debug pipette.
PIPETTE_RADIUS = 0.0005
PIPETTE_HALF_LENGTH = 0.012
PIPETTE_START_Z = CENTER[2] + OUTER_RADIUS + 0.010
PIPETTE_END_Z = CENTER[2] - 0.003
PIPETTE_SPEED = 0.0015  # m/s; deliberately slow diagnostic insertion


def create_uv_sphere_mesh(stage, path: str, radius: float, center: Gf.Vec3f,
                          lat_segments: int = 20, lon_segments: int = 32):
    """Create a closed triangle-only UsdGeom.Mesh sphere."""
    mesh = UsdGeom.Mesh.Define(stage, path)

    points = []
    # north pole
    points.append(Gf.Vec3f(center[0], center[1], center[2] + radius))

    # latitude rings excluding poles
    for i in range(1, lat_segments):
        theta = math.pi * i / lat_segments
        st = math.sin(theta)
        ct = math.cos(theta)
        for j in range(lon_segments):
            phi = 2.0 * math.pi * j / lon_segments
            points.append(
                Gf.Vec3f(
                    center[0] + radius * st * math.cos(phi),
                    center[1] + radius * st * math.sin(phi),
                    center[2] + radius * ct,
                )
            )

    south_idx = len(points)
    points.append(Gf.Vec3f(center[0], center[1], center[2] - radius))

    faces = []

    # north cap
    first_ring = 1
    for j in range(lon_segments):
        a = first_ring + j
        b = first_ring + (j + 1) % lon_segments
        faces.append((0, a, b))

    # middle quads split into triangles
    for i in range(lat_segments - 2):
        ring0 = 1 + i * lon_segments
        ring1 = ring0 + lon_segments
        for j in range(lon_segments):
            jn = (j + 1) % lon_segments
            a = ring0 + j
            b = ring0 + jn
            c = ring1 + j
            d = ring1 + jn
            faces.append((a, c, b))
            faces.append((b, c, d))

    # south cap
    last_ring = 1 + (lat_segments - 2) * lon_segments
    for j in range(lon_segments):
        a = last_ring + j
        b = last_ring + (j + 1) % lon_segments
        faces.append((a, south_idx, b))

    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set([3] * len(faces))
    mesh.GetFaceVertexIndicesAttr().Set([v for tri in faces for v in tri])
    mesh.GetSubdivisionSchemeAttr().Set("none")
    return mesh

def force_visible_subtree(stage, root_path: str, opacity: float | None = None):
    """Force all imageable descendants to normal viewport visibility/purpose."""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"[visibility] missing: {root_path}")
        return

    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Imageable):
            img = UsdGeom.Imageable(prim)
            img.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            img.GetPurposeAttr().Set(UsdGeom.Tokens.default_)

        if prim.IsA(UsdGeom.Gprim):
            g = UsdGeom.Gprim(prim)
            # Direct USD display attributes are a fallback if material binding
            # does not render on a deformable prim.
            if opacity is not None:
                g.GetDisplayOpacityAttr().Set([opacity])

        print(
            f"[visibility] {prim.GetPath()} "
            f"type={prim.GetTypeName()} "
            f"vis={UsdGeom.Imageable(prim).GetVisibilityAttr().Get() if prim.IsA(UsdGeom.Imageable) else '-'} "
            f"purpose={UsdGeom.Imageable(prim).GetPurposeAttr().Get() if prim.IsA(UsdGeom.Imageable) else '-'}"
        )


def create_display_material(stage, path: str, color, opacity: float):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def bind_display_material(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def create_surface_physics_material(stage, path: str):
    mat = UsdShade.Material.Define(stage, path)
    prim = mat.GetPrim()

    prim.ApplyAPI("OmniPhysicsBaseMaterialAPI")
    prim.GetAttribute("omniphysics:dynamicFriction").Set(0.20)
    prim.GetAttribute("omniphysics:density").Set(ZP_DENSITY)

    prim.ApplyAPI("OmniPhysicsDeformableMaterialAPI")
    prim.GetAttribute("omniphysics:youngsModulus").Set(ZP_YOUNGS)
    prim.GetAttribute("omniphysics:poissonsRatio").Set(ZP_POISSON)

    prim.ApplyAPI("OmniPhysicsSurfaceDeformableMaterialAPI")
    prim.GetAttribute("omniphysics:surfaceThickness").Set(ZP_SURFACE_THICKNESS)

    # Omni PhysX currently expects bend stiffness to be supplied explicitly.
    bend_stiffness = ZP_YOUNGS / (12.0 * (1.0 - ZP_POISSON**2))
    prim.GetAttribute("omniphysics:surfaceBendStiffness").Set(bend_stiffness)

    prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.01)
    prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(0.01)
    return mat


def create_volume_physics_material(stage, path: str):
    mat = UsdShade.Material.Define(stage, path)
    prim = mat.GetPrim()

    prim.ApplyAPI("OmniPhysicsBaseMaterialAPI")
    prim.GetAttribute("omniphysics:dynamicFriction").Set(0.10)
    prim.GetAttribute("omniphysics:density").Set(INNER_DENSITY)

    prim.ApplyAPI("OmniPhysicsDeformableMaterialAPI")
    prim.GetAttribute("omniphysics:youngsModulus").Set(INNER_YOUNGS)
    prim.GetAttribute("omniphysics:poissonsRatio").Set(INNER_POISSON)

    prim.ApplyAPI("PhysxDeformableMaterialAPI")
    prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.02)
    return mat


def bind_physics_material(prim, material):
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")



def add_bind_pose(prim, points, instance_name: str = "custom"):
    """Register a mesh's bind pose for deformable embedding."""
    purposes_attr = f"deformablePose:{instance_name}:omniphysics:purposes"
    points_attr = f"deformablePose:{instance_name}:omniphysics:points"

    prim.ApplyAPI("OmniPhysicsDeformablePoseAPI", instance_name)
    if not prim.HasAPI("OmniPhysicsDeformablePoseAPI", instance_name):
        raise RuntimeError(f"Could not apply OmniPhysicsDeformablePoseAPI:{instance_name} to {prim.GetPath()}")

    prim.GetAttribute(purposes_attr).Set(["bindPose"])
    prim.GetAttribute(points_attr).Set(points)


def force_render_mesh(mesh: UsdGeom.Mesh, color, opacity: float, material):
    """Make a graphics mesh unambiguously renderable."""
    prim = mesh.GetPrim()
    img = UsdGeom.Imageable(prim)
    img.GetPurposeAttr().Set(UsdGeom.Tokens.default_)
    img.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)

    gprim = UsdGeom.Gprim(prim)
    gprim.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    gprim.GetDisplayOpacityAttr().Set([opacity])
    bind_display_material(prim, material)


def create_zp_surface(stage):
    """Hierarchical surface deformable with a dedicated graphics mesh."""
    root_path = "/World/ZonaPellucida"
    source_path = "/World/CookingSources/ZP_Source"
    sim_path = root_path + "/SimulationMesh"
    render_path = root_path + "/RenderMesh"

    UsdGeom.Scope.Define(stage, "/World/CookingSources")
    root = UsdGeom.Xform.Define(stage, root_path)
    root_prim = root.GetPrim()

    # Source lives OUTSIDE the deformable subtree as recommended for render-only meshes.
    source = create_uv_sphere_mesh(
        stage, source_path, OUTER_RADIUS, CENTER,
        lat_segments=18, lon_segments=28
    )
    UsdGeom.Imageable(source.GetPrim()).MakeInvisible()

    # Separate graphics mesh inside deformable root.
    render_mesh = create_uv_sphere_mesh(
        stage, render_path, OUTER_RADIUS, CENTER,
        lat_segments=24, lon_segments=40
    )

    if not hasattr(deformableUtils, "create_auto_surface_deformable_hierarchy"):
        raise RuntimeError("create_auto_surface_deformable_hierarchy is unavailable in this Isaac Sim build.")

    ok = deformableUtils.create_auto_surface_deformable_hierarchy(
        stage=stage,
        root_prim_path=root_prim.GetPath(),
        simulation_mesh_path=sim_path,
        cooking_src_mesh_path=source.GetPrim().GetPath(),
        cooking_src_simplification_enabled=False,
        set_visibility_with_guide_purpose=True,
    )
    if not ok:
        raise RuntimeError("Failed to create hierarchical surface-deformable ZP.")

    root_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    if root_prim.HasAPI("PhysxSurfaceDeformableBodyAPI"):
        root_prim.GetAttribute("physxDeformableBody:selfCollision").Set(True)
        root_prim.GetAttribute("physxDeformableBody:disableGravity").Set(True)

    sim_prim = stage.GetPrimAtPath(sim_path)
    if not sim_prim.IsValid():
        raise RuntimeError(f"Missing ZP simulation mesh: {sim_path}")

    sim_mesh = UsdGeom.Mesh(sim_prim)
    sim_points = sim_mesh.GetPointsAttr().Get()
    render_points = render_mesh.GetPointsAttr().Get()

    # Explicit bind poses: this is the critical v6 change.
    add_bind_pose(sim_prim, sim_points)
    add_bind_pose(render_mesh.GetPrim(), render_points)

    # Simulation mesh is physics-only.
    UsdGeom.Imageable(sim_prim).GetPurposeAttr().Set(UsdGeom.Tokens.guide)

    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(sim_prim)
    collision_api.GetRestOffsetAttr().Set(0.00005)
    collision_api.GetContactOffsetAttr().Set(0.00030)

    physics_mat = create_surface_physics_material(stage, "/World/Materials/ZP_Physics")
    bind_physics_material(root_prim, physics_mat)

    visual_mat = create_display_material(
        stage, "/World/Materials/ZP_Visual", (1.0, 0.35, 0.05), 0.42
    )
    force_render_mesh(render_mesh, (1.0, 0.35, 0.05), 0.42, visual_mat)

    print(
        f"[v6-zp] root={root_prim.GetPath()} "
        f"sim_points={len(sim_points)} render_points={len(render_points)} "
        f"render={render_mesh.GetPrim().GetPath()}",
        flush=True,
    )
    return root_prim


def create_inner_volume(stage):
    """Hierarchical volume deformable with explicit graphics-mesh embedding."""
    root_path = "/World/InnerOocyte"
    source_path = "/World/CookingSources/Inner_Source"
    sim_path = root_path + "/SimulationMesh"
    collision_path = root_path + "/CollisionMesh"
    render_path = root_path + "/RenderMesh"

    UsdGeom.Scope.Define(stage, "/World/CookingSources")
    root = UsdGeom.Xform.Define(stage, root_path)
    root_prim = root.GetPrim()

    # Source outside hierarchy.
    source = create_uv_sphere_mesh(
        stage, source_path, INNER_RADIUS, CENTER,
        lat_segments=12, lon_segments=20
    )
    UsdGeom.Imageable(source.GetPrim()).MakeInvisible()

    # Higher-resolution graphics mesh inside deformable hierarchy.
    render_mesh = create_uv_sphere_mesh(
        stage, render_path, INNER_RADIUS, CENTER,
        lat_segments=20, lon_segments=32
    )

    ok = deformableUtils.create_auto_volume_deformable_hierarchy(
        stage=stage,
        root_prim_path=root_prim.GetPath(),
        simulation_tetmesh_path=sim_path,
        collision_tetmesh_path=collision_path,
        cooking_src_mesh_path=source.GetPrim().GetPath(),
        simulation_hex_mesh_enabled=False,
        cooking_src_simplification_enabled=False,
        set_visibility_with_guide_purpose=True,
    )
    if not ok:
        raise RuntimeError("Failed to create hierarchical inner volume deformable.")

    root_prim.ApplyAPI("PhysxBaseDeformableBodyAPI")
    if root_prim.HasAPI("PhysxBaseDeformableBodyAPI"):
        root_prim.GetAttribute("physxDeformableBody:disableGravity").Set(True)
        root_prim.GetAttribute("physxDeformableBody:selfCollision").Set(False)

    # Force cooking before querying generated points.
    get_physx_cooking_interface().cook_auto_deformable_body(str(root_prim.GetPath()))

    sim_prim = stage.GetPrimAtPath(sim_path)
    collision_prim = stage.GetPrimAtPath(collision_path)
    if not sim_prim.IsValid() or not collision_prim.IsValid():
        raise RuntimeError(
            f"Generated inner meshes missing: sim={sim_prim.IsValid()} collision={collision_prim.IsValid()}"
        )

    sim_points = sim_prim.GetAttribute("points").Get()
    collision_points = collision_prim.GetAttribute("points").Get()
    render_points = render_mesh.GetPointsAttr().Get()

    # Explicitly register all three meshes in the SAME bind-pose space.
    add_bind_pose(sim_prim, sim_points)
    add_bind_pose(collision_prim, collision_points)
    add_bind_pose(render_mesh.GetPrim(), render_points)

    UsdGeom.Imageable(sim_prim).GetPurposeAttr().Set(UsdGeom.Tokens.guide)
    UsdGeom.Imageable(collision_prim).GetPurposeAttr().Set(UsdGeom.Tokens.guide)

    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collision_prim)
    collision_api.GetRestOffsetAttr().Set(0.00005)
    collision_api.GetContactOffsetAttr().Set(0.00025)

    physics_mat = create_volume_physics_material(stage, "/World/Materials/Inner_Physics")
    bind_physics_material(root_prim, physics_mat)

    visual_mat = create_display_material(
        stage, "/World/Materials/Inner_Visual", (1.0, 0.72, 0.55), 0.90
    )
    force_render_mesh(render_mesh, (1.0, 0.72, 0.55), 0.90, visual_mat)

    print(
        f"[v6-inner] root={root_prim.GetPath()} "
        f"sim_points={len(sim_points)} collision_points={len(collision_points)} "
        f"render_points={len(render_points)} render={render_mesh.GetPrim().GetPath()}",
        flush=True,
    )
    return root_prim


def print_deformable_tree(stage, root_path: str):
    root = stage.GetPrimAtPath(root_path)
    print(f"[v6-tree] ===== {root_path} =====", flush=True)
    if not root.IsValid():
        print("[v6-tree] MISSING ROOT", flush=True)
        return
    for prim in Usd.PrimRange(root):
        vis = "-"
        purpose = "-"
        pts = "-"
        if prim.IsA(UsdGeom.Imageable):
            img = UsdGeom.Imageable(prim)
            vis = str(img.GetVisibilityAttr().Get())
            purpose = str(img.GetPurposeAttr().Get())
        if prim.HasAttribute("points"):
            p = prim.GetAttribute("points").Get()
            pts = str(len(p) if p is not None else 0)
        print(
            f"[v6-tree] {prim.GetPath()} type={prim.GetTypeName()} "
            f"vis={vis} purpose={purpose} points={pts} "
            f"apis={list(prim.GetAppliedSchemas())}",
            flush=True,
        )


def create_kinematic_pipette(stage):
    """Create a simple vertical capsule used only for the mechanics test."""
    path = "/World/InjectionPipette"
    capsule = UsdGeom.Capsule.Define(stage, path)
    capsule.GetRadiusAttr().Set(PIPETTE_RADIUS)
    capsule.GetHeightAttr().Set(2.0 * PIPETTE_HALF_LENGTH)
    capsule.GetAxisAttr().Set("Z")

    xform = UsdGeom.Xformable(capsule.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, PIPETTE_START_Z))

    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
    rigid = UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim())
    rigid.GetKinematicEnabledAttr().Set(True)

    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(capsule.GetPrim())
    physx_collision.GetRestOffsetAttr().Set(0.00005)
    physx_collision.GetContactOffsetAttr().Set(0.00025)

    visual_mat = create_display_material(
        stage, "/World/Materials/Pipette_Visual", (0.95, 0.05, 0.95), 1.0
    )
    bind_display_material(capsule.GetPrim(), visual_mat)
    return capsule


def main():
    sim = SimulationContext(
        SimulationCfg(
            dt=1.0 / 120.0,
            render_interval=1,
            gravity=(0.0, 0.0, 0.0),
        )
    )
    sim.set_camera_view(
        eye=(0.045, -0.055, 0.045),
        target=(0.0, 0.0, CENTER[2]),
    )

    stage = omni.usd.get_context().get_stage()

    # A visual reference plane only; gravity is disabled in this first test.
    physicsUtils.add_ground_plane(
        stage, "/World/GroundPlane", "Z", 1.0,
        Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.15, 0.15, 0.15)
    )

    print("[layered-test] V5 loaded: return bug fixed + post-cooking visibility diagnostics")
    print("[layered-test] creating surface-deformable ZP...")
    create_zp_surface(stage)

    print("[layered-test] creating volume-deformable inner oocyte...")
    print("[layered-test] v6: explicit bindPose render meshes + hierarchy diagnostics")
    create_inner_volume(stage)

    print("[layered-test] creating kinematic pipette...")
    pipette = create_kinematic_pipette(stage)

    # Tiny center marker for viewport diagnostics only.
    marker = UsdGeom.Sphere.Define(stage, "/World/OocyteCenterMarker")
    marker.GetRadiusAttr().Set(0.00025)
    UsdGeom.Xformable(marker.GetPrim()).AddTranslateOp().Set(CENTER)
    marker_mat = create_display_material(
        stage, "/World/Materials/CenterMarker_Visual", (0.1, 0.9, 0.2), 1.0
    )
    bind_display_material(marker.GetPrim(), marker_mat)

    sim.reset()

    # IMPORTANT: PhysX cooking/reset can overwrite visibility/purpose authored
    # before reset. Re-apply it after the deformables have finished cooking.
    print("[layered-test] v6: forcing graphics meshes visible after reset...")
    force_visible_subtree(stage, "/World/ZonaPellucida", opacity=0.35)
    force_visible_subtree(stage, "/World/InnerOocyte", opacity=0.82)
    print_deformable_tree(stage, "/World/ZonaPellucida")
    print_deformable_tree(stage, "/World/InnerOocyte")

    print("")
    print("[layered-test] READY")
    print(f"  outer ZP radius       : {OUTER_RADIUS:.4f} m")
    print(f"  ZP surface thickness  : {ZP_SURFACE_THICKNESS:.4f} m")
    print(f"  inner radius          : {INNER_RADIUS:.4f} m")
    print("  visualization         : translucent ZP + inner body + green center marker")
    print("")
    print("The pipette will descend slowly. Watch for:")
    print("  1. ZP indentation")
    print("  2. ZP moving inward toward the inner body")
    print("  3. subsequent inner-body deformation")
    print("This script does NOT yet implement puncture/tearing.")
    print("")

    translate_op = None
    for op in UsdGeom.Xformable(pipette.GetPrim()).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = UsdGeom.Xformable(pipette.GetPrim()).AddTranslateOp()

    z = PIPETTE_START_Z
    direction = -1.0

    while simulation_app.is_running():
        # Slow cyclic indentation/retraction test.
        z += direction * PIPETTE_SPEED * sim.get_physics_dt()

        if z <= PIPETTE_END_Z:
            direction = 1.0
        elif z >= PIPETTE_START_Z:
            direction = -1.0

        translate_op.Set(Gf.Vec3f(0.0, 0.0, z))
        sim.step()

    simulation_app.close()


if __name__ == "__main__":
    main()
