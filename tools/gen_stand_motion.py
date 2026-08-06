"""
Synthesizes a static "stand" reference clip for the G1 by holding Unitree's own
default ready-stand pose for a few seconds. No mocap needed -- this is the standard
bootstrap trick for giving AMP a "real" near-zero-velocity example to imitate when
no idle/standing capture is available.

The pose comes from `default_angles` in twist3/unitree_sdk2/example_python/configs/g1.yaml
(identical copy also in twist3/TWIST2/deploy_real/robot_control/configs/g1.yaml) -- the
pose Unitree's own low-level examples and TWIST2's real-robot deploy code hold the G1 in
before any policy takes over. Cross-checked: that file's kps/kds match g1.xml's joint
stiffness/damping exactly, confirming it's the same robot's real PD gains, not a
different G1 variant.

Usage:
    python tools/gen_stand_motion.py --out_file data/motions/g1/g1_stand.pkl
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mimickit"))

from anim.motion import Motion, LoopMode

# [root_pos(3), root_rot_expmap(3), dof_pos(29)]
# Root pos/rot kept the same as the old bootstrap pose (0.8m pelvis height, upright) --
# only dof_pos is replaced with Unitree's default_angles (slight knee bend, arms down).
STAND_POSE = [0, 0, 0.8, 0, 0, 0,
              -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
              -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
              0, 0, 0,
              0, 0.4, 0, 1.2, 0.0, 0.0, 0.0,
              0, -0.4, 0, 1.2, 0.0, 0.0, 0.0]

FPS = 30
DURATION_SECONDS = 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", required=True)
    args = parser.parse_args()

    num_frames = int(FPS * DURATION_SECONDS)
    frame = np.array(STAND_POSE, dtype=np.float64)
    frames = np.tile(frame, (num_frames, 1))

    motion = Motion(loop_mode=LoopMode.WRAP, fps=FPS, frames=frames)
    motion.save(args.out_file)
    print("Saved stand motion ({:d} frames @ {:d} fps) to {:s}".format(num_frames, FPS, args.out_file))
    return


if __name__ == "__main__":
    main()
