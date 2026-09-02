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
