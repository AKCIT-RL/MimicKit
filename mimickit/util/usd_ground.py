"""Ground colliders authored with pxr only (no Kit), shared by the Isaac Lab engine and its
CPU test: a static box for `plane`, or one static triangle mesh per tile plus a safety floor
for `uneven` (the Isaac Gym engine's layout). The root prim is a kinematic rigid body so
contact sensors can filter against it; everything below it is a collision shape of it."""

import numpy as np

import util.terrain_util as terrain_util
from util.logger import Logger


def build_ground(stage, ground_path, config, env_offset_max):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, Vt

    config = config if (config is not None) else dict()
    ground_type = config.get("type", "plane")
    static_friction = float(config.get("static_friction", 1.0))
    dynamic_friction = float(config.get("dynamic_friction", 1.0))
    restitution = float(config.get("restitution", 0.0))

    UsdGeom.Xform.Define(stage, ground_path)
    root = stage.GetPrimAtPath(ground_path)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.RigidBodyAPI.Get(stage, ground_path).GetKinematicEnabledAttr().Set(True)

    material = UsdShade.Material.Define(stage, ground_path + "/physics_material")
    mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat_api.CreateStaticFrictionAttr(static_friction)
    mat_api.CreateDynamicFrictionAttr(dynamic_friction)
    mat_api.CreateRestitutionAttr(restitution)

    def bind(prim):
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")

    def add_floor(top_z, half_extent):
        floor = UsdGeom.Cube.Define(stage, ground_path + "/floor")
        floor.CreateSizeAttr(1.0)
        thickness = 2.0
        floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, top_z - 0.5 * thickness))
        floor.AddScaleOp().Set(Gf.Vec3d(2.0 * half_extent, 2.0 * half_extent, thickness))
        floor.CreateVisibilityAttr("invisible")
        bind(floor.GetPrim())

    if (ground_type == "plane"):
        add_floor(0.0, 2.0 * env_offset_max + 1000.0)
    elif (ground_type == "uneven"):
        tile_centers = config.get("tile_centers", None)
        tile_size = config.get("tile_size", None)
        assert (tile_centers is not None and tile_size is not None), \
            "ground.tile_centers [N, 2] and ground.tile_size [2] (m) are required for " \
            "uneven ground; the soccer env injects them from its field grid"
        horizontal_scale = float(config.get("horizontal_scale", 0.5))
        amplitude = float(config.get("random_height", 0.02))
        centers = np.asarray(tile_centers, dtype=np.float64)
        half = float(np.abs(centers).max()) + max(tile_size) + 100.0
        # safety net just below the deepest dip so nothing falls into the void
        add_floor(-amplitude, half)

        UsdGeom.Scope.Define(stage, ground_path + "/tiles")
        total_tris = 0
        for i, tile_center in enumerate(tile_centers):
            heights = terrain_util.build_uneven_tile(float(tile_size[0]), float(tile_size[1]),
                                                     horizontal_scale, amplitude)
            nx, ny = heights.shape
            x_offset = float(tile_center[0]) - 0.5 * (nx - 1) * horizontal_scale
            y_offset = float(tile_center[1]) - 0.5 * (ny - 1) * horizontal_scale
            verts, tris = terrain_util.heightfield_to_trimesh(heights, horizontal_scale,
                                                              x_offset=x_offset, y_offset=y_offset)
            mesh = UsdGeom.Mesh.Define(stage, "{}/tiles/tile_{:d}".format(ground_path, i))
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(tris.shape[0], 3, dtype=np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.astype(np.int32).flatten()))
            mesh.CreateVisibilityAttr("invisible")
            prim = mesh.GetPrim()
            bind(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
            total_tris += tris.shape[0]
        Logger.print("Built uneven ground: {:d} tiles of {:.0f}x{:.0f} m, +-{:.3f} m bumps, "
                     "{:d} triangles total".format(len(tile_centers), float(tile_size[0]),
                                                   float(tile_size[1]), amplitude, total_tris))
    else:
        raise ValueError("Unsupported ground type: {}".format(ground_type))
    return
