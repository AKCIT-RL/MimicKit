"""
Quick visual sanity check: renders the G1 held in a static pose (default: the
proposed "stand" reference pose) to a PNG, using plain MuJoCo -- no Isaac Lab, no
physics stepping, just forward kinematics + a render. Meant to eyeball a pose before
committing it as an AMP reference clip.

Usage:
    python tools/sim2sim_mujoco/preview_pose.py --out_file output/sim2sim_mujoco/stand_pose.png
"""
import argparse
import os
import sys

import numpy as np
import torch
import mujoco
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keyboard_test import CHAR_FILE, xyzw_to_mj_quat, load_mj_model_with_ground

MIMICKIT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(MIMICKIT_ROOT, "mimickit"))

import util.torch_util as torch_util
NUM_DOF = 29

# Same pose as init_pose in data/envs/amp_steering_g1_env.yaml / tools/gen_stand_motion.py
STAND_POSE = [0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              1.57, 0, 0, 0, 0, 0, 0, 1.57, 0, 0, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", default=os.path.join(MIMICKIT_ROOT, "output/sim2sim_mujoco/stand_pose.png"))
    parser.add_argument("--pose", default=None, help="comma-separated 35 floats (root_pos3+root_rot_expmap3+dof29); defaults to STAND_POSE")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    pose = np.array(STAND_POSE if args.pose is None else [float(v) for v in args.pose.split(",")], dtype=np.float64)

    mj_model = load_mj_model_with_ground(CHAR_FILE)
    # default offscreen framebuffer is 640x480; bump it before creating the renderer
    # instead of editing the shared g1.xml asset.
    mj_model.vis.global_.offwidth = 960
    mj_model.vis.global_.offheight = 720
    mj_data = mujoco.MjData(mj_model)

    root_rot_expmap = torch.tensor(pose[3:6], dtype=torch.float32)
    root_rot_xyzw = torch_util.exp_map_to_quat(root_rot_expmap).numpy()
    mj_data.qpos[0:3] = pose[0:3]
    mj_data.qpos[3:7] = xyzw_to_mj_quat(root_rot_xyzw)
    mj_data.qpos[7:7 + NUM_DOF] = pose[6:6 + NUM_DOF]
    mujoco.mj_forward(mj_model, mj_data)

    renderer = mujoco.Renderer(mj_model, height=720, width=960)
    # level, straight-on front view (elevation=0, azimuth=90 faces +Y) with the
    # ground plane in frame -- makes true verticality actually judgeable, unlike an
    # oblique angle with no floor reference.
    cam = mujoco.MjvCamera()
    cam.lookat = [pose[0], pose[1], 0.9]
    cam.distance = 3.0
    cam.azimuth = 0
    cam.elevation = 0
    renderer.update_scene(mj_data, camera=cam)

    img = renderer.render()
    Image.fromarray(img).save(args.out_file)
    print("Saved pose preview to {:s}".format(args.out_file))
    return


if __name__ == "__main__":
    main()
