"""
Same sim2sim tester as keyboard_test.py, but opens a live MuJoCo viewer window
instead of running headless with periodic video export. Requires a real (or X11
forwarded) display.

Requirements:
  - Connect over SSH with X11 forwarding: `ssh -X user@host` (not a plain SSH
    session -- without -X, no window will appear, no matter what this script does).
  - Do NOT set MUJOCO_GL=egl/osmesa for this script (those are for the OFFSCREEN
    renderer used by keyboard_test.py/preview_pose.py). Leave MUJOCO_GL unset, or
    `unset MUJOCO_GL` first, so MuJoCo uses its default GLFW windowed backend.

Uses the MimicKit g1.xml asset (same one the checkpoint was trained against), not
the unitree_description MJCF used by motion_tracking_controller -- picked
specifically to keep joint order/PD gains guaranteed consistent with the checkpoint.

Controls (same mapping as keyboard_test.py, read via the viewer window's key
callback instead of the terminal): w=throttle up, x=throttle down, a=steer left,
d=steer right, s=stop. Close the window or Ctrl+C in the terminal to quit.

Usage:
    python tools/sim2sim_mujoco/keyboard_test_viewer.py --checkpoint output/g1_locomotion/model.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keyboard_test import (
    CHAR_FILE, KEY_BODIES, NUM_DOF, CONTROL_FREQ, SIM_SUBSTEPS,
    xyzw_to_mj_quat, G1Sim2SimPolicy, KeyboardCommand, build_observation,
    load_mj_model_with_ground, apply_pd_target, load_init_pose,
)

MIMICKIT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(MIMICKIT_ROOT, "mimickit"))
import util.torch_util as torch_util
import anim.mjcf_char_model as mjcf_char_model

# GLFW key codes for printable ASCII letters equal their uppercase ASCII value.
KEY_CHAR_MAP = {ord("W"): "w", ord("X"): "x", ord("A"): "a", ord("D"): "d", ord("S"): "s"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init_pose", default="synthetic", choices=["synthetic", "walk", "run"])
    parser.add_argument("--pd_gain_scale", type=float, default=1.0)
    args = parser.parse_args()

    kin_model = mjcf_char_model.MJCFCharModel(args.device)
    kin_model.load(CHAR_FILE)
    body_names = kin_model.get_body_names()
    key_body_ids = torch.tensor([body_names.index(b) for b in KEY_BODIES], dtype=torch.long)

    mj_model = load_mj_model_with_ground(CHAR_FILE)
    if args.pd_gain_scale != 1.0:
        mj_model.jnt_stiffness[1:1 + NUM_DOF] *= args.pd_gain_scale
        mj_model.dof_damping[6:6 + NUM_DOF] *= args.pd_gain_scale
    mj_data = mujoco.MjData(mj_model)

    init_pose = load_init_pose(args.init_pose)
    root_rot_expmap = torch.tensor(init_pose[3:6], dtype=torch.float32)
    root_rot_xyzw = torch_util.exp_map_to_quat(root_rot_expmap).numpy()
    mj_data.qpos[0:3] = init_pose[0:3]
    mj_data.qpos[3:7] = xyzw_to_mj_quat(root_rot_xyzw)
    mj_data.qpos[7:7 + NUM_DOF] = init_pose[6:6 + NUM_DOF]
    mujoco.mj_forward(mj_model, mj_data)

    policy = G1Sim2SimPolicy(args.checkpoint, device=args.device)
    cmd = KeyboardCommand()

    def key_callback(keycode):
        letter = KEY_CHAR_MAP.get(keycode)
        if letter is not None:
            cmd.handle_key(letter)
        return

    dt = 1.0 / CONTROL_FREQ
    print("Window controls: w=throttle up  x=throttle down  a=steer left  d=steer right  s=stop")
    print("Close the window or Ctrl+C here to quit.")

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            tar_dir = cmd.get_tar_dir()
            tar_speed = cmd.get_tar_speed()

            obs, root_pos, root_vel = build_observation(kin_model, key_body_ids, mj_model, mj_data, tar_dir, tar_speed)
            action = policy.act(obs).squeeze(0).numpy()
            target_pos = action

            apply_pd_target(mj_model, mj_data, target_pos)
            for _ in range(SIM_SUBSTEPS):
                mujoco.mj_step(mj_model, mj_data)

            viewer.sync()

            speed_xy = np.linalg.norm(root_vel[0:2])
            print("\rcmd: dir={:+.0f}deg speed={:.1f} | measured: speed={:.2f} height={:.2f}   ".format(
                np.degrees(cmd.theta), tar_speed, speed_xy, root_pos[2]), end="", flush=True)

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    print("\nDone.")
    return


if __name__ == "__main__":
    main()
