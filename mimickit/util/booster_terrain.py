"""Booster Gym's terrain, ported line by line (booster_gym/utils/terrain.py,
envs/t1.py _get_env_origins / _teleport_robot, config envs/T1.yaml).

One shared map: `num_terrains` sub-terrains of terrain_width x terrain_length
laid along +x, a flat border of border_size around them, all on one int16
heightfield (vertical_scale per unit) turned into one PhysX triangle mesh
placed at (-border, -border, 0). Sub-terrain i is chosen by its index against
the cumulative terrain_proportions, exactly as Booster Gym does: with
[plane 0, slope 0, random 0.5, discrete 0.5] and 8 terrains, terrains 0-3 are
random uniform and 4-7 discrete obstacles.

The generators are isaacgym.terrain_utils (the same functions Booster Gym
calls) and they draw from the GLOBAL np.random, as in Booster Gym; MimicKit
seeds it from --rand_seed, so the map follows the run seed.

The numpy side (heightfield, origins, teleport) lives here so it is testable
without a simulator; heightfield_height_t is the torch twin of Booster Gym's
Terrain.terrain_heights, used for every quantity that needs "height above the
local ground" instead of absolute z.
"""

import numpy as np
import torch

# envs/T1.yaml, terrain block, verbatim. A config may override any key, but
# the experiment configs leave them alone: identity with Booster Gym is the
# point of this module.
BOOSTER_T1_TERRAIN = {
    "static_friction": 1.0,
    "dynamic_friction": 1.0,
    "restitution": 0.0,
    "terrain_length": 10.0,
    "terrain_width": 10.0,
    "border_size": 5.0,
    "num_terrains": 8,
    "terrain_proportions": [0.0, 0.0, 0.5, 0.5],
    "slope": 0.1,
    "random_height": 0.1,
    "discrete_height": 0.02,
    "horizontal_scale": 0.1,
    "vertical_scale": 0.005,
    "slope_threshold": 2.0,
}


def resolve_config(ground_config):
    cfg = dict(BOOSTER_T1_TERRAIN)
    for k in BOOSTER_T1_TERRAIN:
        if k in ground_config:
            cfg[k] = ground_config[k]
    return cfg


def map_dims(cfg):
    """(env_width along x, env_length along y, border) in meters, Booster's names."""
    return (cfg["num_terrains"] * cfg["terrain_width"], cfg["terrain_length"], cfg["border_size"])


def build_heightfield(cfg):
    """Booster Gym Terrain._create_trimesh up to (not including) the mesh call.
    Returns the int16 heightfield [X, Y] (units of vertical_scale)."""
    from isaacgym import terrain_utils

    hs = cfg["horizontal_scale"]
    border_pixels = int(cfg["border_size"] / hs)
    terrain_width_pixels = int(cfg["terrain_width"] / hs)
    terrain_length_pixels = int(cfg["terrain_length"] / hs)
    height_field_raw = np.zeros(
        (cfg["num_terrains"] * terrain_width_pixels + 2 * border_pixels,
         terrain_length_pixels + 2 * border_pixels), dtype=np.int16)
    proportions = [cfg["num_terrains"] * np.sum(cfg["terrain_proportions"][: i + 1])
                   / np.sum(cfg["terrain_proportions"])
                   for i in range(len(cfg["terrain_proportions"]))]
    for i in range(cfg["num_terrains"]):
        terrain = terrain_utils.SubTerrain("terrain", width=terrain_width_pixels,
                                           length=terrain_length_pixels,
                                           vertical_scale=cfg["vertical_scale"],
                                           horizontal_scale=hs)
        if i < proportions[0]:
            pass
        elif i < proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=cfg["slope"], platform_size=3.0)
        elif i < proportions[2]:
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=-0.5 * cfg["random_height"],
                                                 max_height=0.5 * cfg["random_height"],
                                                 step=0.005, downsampled_scale=0.2)
        else:
            terrain_utils.discrete_obstacles_terrain(terrain, max_height=cfg["discrete_height"],
                                                     min_size=1.0, max_size=2.0, num_rects=20,
                                                     platform_size=3.0)
        start_x = border_pixels + i * terrain_width_pixels
        end_x = border_pixels + (i + 1) * terrain_width_pixels
        start_y = border_pixels
        end_y = border_pixels + terrain_length_pixels
        height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw
    return height_field_raw


def heightfield_to_mesh(height_field_raw, cfg):
    from isaacgym import terrain_utils
    return terrain_utils.convert_heightfield_to_trimesh(
        height_field_raw, cfg["horizontal_scale"], cfg["vertical_scale"], cfg["slope_threshold"])


def heightfield_height_np(height_field_raw, cfg, xy):
    """Booster Gym Terrain.terrain_heights, verbatim (bilinear, no clamping)."""
    hs = cfg["horizontal_scale"]
    border_pixels = int(cfg["border_size"] / hs)
    x = border_pixels + xy[:, 0] / hs
    y = border_pixels + xy[:, 1] / hs
    x1 = np.floor(x).astype(int)
    x2 = x1 + 1
    y1 = np.floor(y).astype(int)
    y2 = y1 + 1
    hf = height_field_raw
    return ((x2 - x) * (y2 - y) * hf[x1, y1] + (x - x1) * (y2 - y) * hf[x2, y1]
            + (x2 - x) * (y - y1) * hf[x1, y2] + (x - x1) * (y - y1) * hf[x2, y2]) \
        * cfg["vertical_scale"]


def heightfield_height_t(hf_t, border_pixels, horizontal_scale, vertical_scale, xy):
    """Torch twin of heightfield_height_np for xy [..., 2]. Indices are clamped
    to the map (Booster Gym does not clamp; there a robot past the border would
    raise in numpy, here it reads the edge height), the only difference."""
    x = border_pixels + xy[..., 0] / horizontal_scale
    y = border_pixels + xy[..., 1] / horizontal_scale
    x1f = torch.floor(x)
    y1f = torch.floor(y)
    nx, ny = hf_t.shape[0], hf_t.shape[1]
    x1 = x1f.long().clamp(0, nx - 2)
    y1 = y1f.long().clamp(0, ny - 2)
    x2 = x1 + 1
    y2 = y1 + 1
    fx = (x - x1.to(x.dtype)).clamp(0.0, 1.0)
    fy = (y - y1.to(y.dtype)).clamp(0.0, 1.0)
    h = ((1.0 - fx) * (1.0 - fy) * hf_t[x1, y1] + fx * (1.0 - fy) * hf_t[x2, y1]
         + (1.0 - fx) * fy * hf_t[x1, y2] + fx * fy * hf_t[x2, y2])
    return h * vertical_scale


def env_origins(num_envs, cfg, height_field_raw):
    """Booster Gym _get_env_origins (trimesh branch), verbatim. [N, 3] float32.
    Note the fixed env -> sub-terrain assignment: env i always spawns at the
    same spot, so which robot sees steps and which sees bumps is set by its
    index, as in Booster Gym."""
    env_width, env_length, _ = map_dims(cfg)
    num_cols = max(1.0, np.floor(np.sqrt(num_envs * env_length / env_width)))
    num_rows = np.ceil(num_envs / num_cols)
    xx, yy = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing="ij")
    origins = np.zeros([num_envs, 3], dtype=np.float32)
    origins[:, 0] = env_width / (num_rows + 1) * (xx.flatten()[:num_envs] + 1)
    origins[:, 1] = env_length / (num_cols + 1) * (yy.flatten()[:num_envs] + 1)
    origins[:, 2] = heightfield_height_np(height_field_raw, cfg, origins[:, 0:2])
    return origins


def teleport_shift(xy, env_width, env_length, border):
    """Booster Gym _teleport_robot: the xy shift for robots past 0.75 border.
    Works for numpy arrays and torch tensors, [..., 2] -> [..., 2]."""
    lo = -0.75 * border
    shift = xy * 0.0
    shift[..., 0] += (xy[..., 0] < lo) * (env_width + border)
    shift[..., 0] -= (xy[..., 0] > env_width + 0.75 * border) * (env_width + border)
    shift[..., 1] += (xy[..., 1] < lo) * (env_length + border)
    shift[..., 1] -= (xy[..., 1] > env_length + 0.75 * border) * (env_length + border)
    return shift
