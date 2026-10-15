"""Pure-numpy heightfield helpers for uneven ground (engine-agnostic).

Kept free of simulator imports so the geometry is unit-testable on CPU; the
Isaac Gym engine converts the output into a PhysX triangle mesh.
"""

import numpy as np


def build_uneven_heightfield(size_x, size_y, horizontal_scale, amplitude, rng=None):
    """Random uniform heightfield covering ``size_x`` x ``size_y`` meters.

    Heights are sampled iid from U(-amplitude, amplitude) on a grid with
    ``horizontal_scale`` spacing (paper sec 4.2: "slightly uneven terrain").

    Returns a float32 array [nx, ny] of heights in meters.
    """
    if (size_x <= 0.0 or size_y <= 0.0):
        raise ValueError("heightfield size must be positive, got ({}, {})".format(size_x, size_y))
    if (horizontal_scale <= 0.0):
        raise ValueError("horizontal_scale must be positive, got {}".format(horizontal_scale))
    if (amplitude < 0.0):
        raise ValueError("amplitude must be non-negative, got {}".format(amplitude))

    nx = int(np.ceil(size_x / horizontal_scale)) + 1
    ny = int(np.ceil(size_y / horizontal_scale)) + 1

    if (rng is None):
        rng = np.random
    heights = rng.uniform(-amplitude, amplitude, size=(nx, ny)).astype(np.float32)
    return heights


def build_uneven_tile(size_x, size_y, horizontal_scale, amplitude, rng=None):
    """Heightfield tile whose border ring is flattened to z = 0.

    Adjacent tiles laid edge to edge then meet at z = 0, so a grid of
    independently random tiles is continuous at the seams (no lips). Used to
    give every env its own small ground mesh instead of one world-sized mesh,
    which blows up the PhysX GPU broadphase pair count.
    """
    heights = build_uneven_heightfield(size_x, size_y, horizontal_scale, amplitude, rng=rng)
    heights[0, :] = 0.0
    heights[-1, :] = 0.0
    heights[:, 0] = 0.0
    heights[:, -1] = 0.0
    return heights


def heightfield_to_trimesh(heights, horizontal_scale, x_offset=0.0, y_offset=0.0):
    """Convert a [nx, ny] heightfield into a triangle mesh.

    Returns (vertices [nx*ny, 3] float32, triangles [2*(nx-1)*(ny-1), 3] uint32)
    with CCW winding seen from +z. Vertex (i, j) sits at
    (x_offset + i*hs, y_offset + j*hs, heights[i, j]).
    """
    heights = np.asarray(heights, dtype=np.float32)
    if (heights.ndim != 2 or heights.shape[0] < 2 or heights.shape[1] < 2):
        raise ValueError("heights must be [nx>=2, ny>=2], got {}".format(heights.shape))
    nx, ny = heights.shape

    xs = x_offset + horizontal_scale * np.arange(nx, dtype=np.float32)
    ys = y_offset + horizontal_scale * np.arange(ny, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    vertices = np.stack([grid_x, grid_y, heights], axis=-1).reshape(-1, 3).astype(np.float32)

    # two CCW triangles per cell: (v00, v10, v11) and (v00, v11, v01)
    i = np.arange(nx - 1, dtype=np.uint32)
    j = np.arange(ny - 1, dtype=np.uint32)
    grid_i, grid_j = np.meshgrid(i, j, indexing="ij")
    v00 = (grid_i * ny + grid_j).ravel()
    v01 = v00 + 1
    v10 = v00 + np.uint32(ny)
    v11 = v10 + 1
    tris = np.empty([2 * v00.shape[0], 3], dtype=np.uint32)
    tris[0::2] = np.stack([v00, v10, v11], axis=-1)
    tris[1::2] = np.stack([v00, v11, v01], axis=-1)
    return vertices, tris


# per-tile overrides allowed in ground.tile_levels (anything else is a typo)
TILE_LEVEL_KEYS = {"random_height", "horizontal_scale", "step_height_range", "step_size", "step_density"}


def resolve_tile_configs(config, num_tiles):
    """Per-tile terrain parameters, [num_tiles] dicts.

    Without ``tile_levels`` every tile uses the base ground config (the old
    single-amplitude behaviour). With ``tile_levels: [{...}, ...]`` tile i
    takes level ``i % len(tile_levels)`` (round-robin, so the mixture is fixed
    by the env index and needs no seed), the level dict overriding the base
    keys. Steps (``step_height_range``) are off unless set.
    """
    base = {
        "random_height": float(config.get("random_height", 0.02)),
        "horizontal_scale": float(config.get("horizontal_scale", 0.5)),
        "step_height_range": config.get("step_height_range", None),
        "step_size": config.get("step_size", [1.0, 2.0]),
        "step_density": float(config.get("step_density", 0.2)),
    }
    levels = config.get("tile_levels", None)
    if (levels is None or len(levels) == 0):
        return [base] * num_tiles
    tile_cfgs = []
    for level in levels:
        unknown = set(level) - TILE_LEVEL_KEYS
        if (len(unknown) > 0):
            raise ValueError("unknown ground.tile_levels key(s): {}".format(sorted(unknown)))
        cfg = dict(base)
        cfg.update(level)
        tile_cfgs.append(cfg)
    return [tile_cfgs[i % len(tile_cfgs)] for i in range(num_tiles)]


def tile_max_height(tile_cfg):
    """Upper bound of |height| on a tile: bump amplitude plus the tallest step (m)."""
    h = float(tile_cfg["random_height"])
    if (tile_cfg.get("step_height_range", None) is not None):
        h += float(tile_cfg["step_height_range"][1])
    return h


def add_step_patches(heights, horizontal_scale, step_height_range, step_size, step_density, rng=None):
    """Booster Gym-style discrete obstacles on top of a heightfield (in place).

    Rectangles of side U(step_size) m are placed uniformly, step_density per m^2,
    each with height s * U(step_height_range), s = +-1. A later rectangle
    overwrites an earlier one (as in Booster's discrete_obstacles_terrain), so
    every step offset is 0 or has |offset| in step_height_range. The edge of a
    patch is a ramp one horizontal_scale wide (the mesh is a heightfield).
    """
    lo, hi = float(step_height_range[0]), float(step_height_range[1])
    if (lo < 0.0 or hi < lo):
        raise ValueError("step_height_range must be 0 <= lo <= hi, got {}".format(step_height_range))
    if (rng is None):
        rng = np.random
    nx, ny = heights.shape
    area = (nx - 1) * (ny - 1) * horizontal_scale * horizontal_scale
    num_rects = int(round(step_density * area))
    offset = np.zeros_like(heights)
    for _ in range(num_rects):
        w = max(1, int(round(rng.uniform(step_size[0], step_size[1]) / horizontal_scale)))
        l = max(1, int(round(rng.uniform(step_size[0], step_size[1]) / horizontal_scale)))
        i0 = rng.randint(0, max(1, nx - w))
        j0 = rng.randint(0, max(1, ny - l))
        sign = 1.0 if rng.uniform(0.0, 1.0) < 0.5 else -1.0
        offset[i0:i0 + w, j0:j0 + l] = sign * rng.uniform(lo, hi)
    heights += offset.astype(heights.dtype)
    return heights


def build_tile(size_x, size_y, tile_cfg, rng=None):
    """One ground tile from its resolved config (see resolve_tile_configs).

    Without steps this is exactly build_uneven_tile (same draws from rng);
    with steps the patches are added and the border ring is flattened again.
    """
    hs = float(tile_cfg["horizontal_scale"])
    heights = build_uneven_tile(size_x, size_y, hs, float(tile_cfg["random_height"]), rng=rng)
    if (tile_cfg.get("step_height_range", None) is not None):
        add_step_patches(heights, hs, tile_cfg["step_height_range"], tile_cfg["step_size"],
                         float(tile_cfg["step_density"]), rng=rng)
        heights[0, :] = 0.0
        heights[-1, :] = 0.0
        heights[:, 0] = 0.0
        heights[:, -1] = 0.0
    return heights
