"""
Sim2sim tester: runs a trained MimicKit G1 `task_steering` checkpoint directly in
native MuJoCo (no ONNX, no ROS2 -- loads the .pt checkpoint straight into PyTorch),
driven by keyboard instead of a joystick.

Controls emulate a left analog stick -- longitudinal speed + turning rate, not four
fixed compass directions (matching how tar_dir/tar_speed are meant to be driven: the
character always walks in the direction it's currently facing and steers by turning,
per the 360-degree tar_dir range it was trained on). Read from the terminal, no GUI
window needed -- works over SSH:
  w  -> throttle up   (each press/repeat-while-held adds SPEED_STEP m/s to tar_speed)
  x  -> throttle down (each press/repeat-while-held subtracts SPEED_STEP m/s, floor 0)
  a  -> steer left  (each press/repeat-while-held turns the heading left by TURN_STEP)
  d  -> steer right (each press/repeat-while-held turns the heading right by TURN_STEP)
  s  -> stop (tar_speed = 0, heading unchanged)
  q  -> quit

Since this runs headless over SSH, there's no live viewer window. Instead it
periodically renders offscreen and appends frames to a rolling video buffer that
gets flushed to an .mp4 file every VIDEO_FLUSH_SECONDS, so you can scp it down and
watch it after the fact. Live telemetry (current command + measured root
speed/height) prints to the terminal every step.

Usage:
    python tools/sim2sim_mujoco/keyboard_test.py --checkpoint output/g1_locomotion/model.pt

Known-risk areas (flagged honestly, not yet empirically verified against Isaac Lab):
  - PD gains are read from the MJCF's per-joint <joint stiffness=".." damping=".."/>
    via MuJoCo's compiled model (model.jnt_stiffness / model.dof_damping) -- this is
    assumed to match what Isaac Lab used internally during training, but hasn't been
    cross-checked line-by-line against isaac_lab_engine.py's actuator model.
  - Quaternion convention: MimicKit uses xyzw internally (confirmed by the
    quat-reordering code in anim/mjcf_char_model.py's MJCF loader); MuJoCo's native
    qpos/quat attributes are wxyz. Converted explicitly below -- double check first if
    the character's initial pose looks twisted.
  - Root angular velocity frame: MuJoCo's free-joint qvel[3:6] is in the root's local
    frame; compute_char_obs's heading-relative transform expects world-frame angular
    velocity, so it's rotated into world frame first (see `build_observation` below).
"""
import argparse
import os
import select
import sys
import termios
import time
import tty

import numpy as np
import torch
import mujoco

MIMICKIT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(MIMICKIT_ROOT, "mimickit"))

import util.torch_util as torch_util
import anim.mjcf_char_model as mjcf_char_model
import envs.char_env as char_env
import envs.task_steering_env as task_steering_env

CHAR_FILE = os.path.join(MIMICKIT_ROOT, "data/assets/g1/g1.xml")
KEY_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link", "head_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
INIT_POSE = [0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.57, 0, 0, 0, 0, 0, 0, 1.57, 0, 0, 0]
NUM_DOF = 29
CONTROL_FREQ = 30.0  # matches control_freq in data/engines/isaac_lab_engine.yaml
SIM_FREQ = 120.0     # matches sim_freq in data/engines/isaac_lab_engine.yaml
SIM_SUBSTEPS = int(SIM_FREQ / CONTROL_FREQ)  # physics substeps per control step

SPEED_STEP = 0.15  # per detected keypress event; w/x repeat while held via terminal auto-repeat
SPEED_MAX = 3.5
TURN_STEP = np.radians(4.0)  # heading change per detected keypress event, same held-key logic

VIDEO_FLUSH_SECONDS = 15.0
VIDEO_OUT_DIR = os.path.join(MIMICKIT_ROOT, "output", "sim2sim_mujoco")
RENDER_WIDTH, RENDER_HEIGHT = 640, 480


def mj_quat_to_xyzw(q_wxyz):
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def xyzw_to_mj_quat(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


class RawTerminal:
    """Non-blocking single-keypress reader over a plain SSH terminal (no curses/GUI)."""

    def __enter__(self):
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *args):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        return False

    def read_key(self):
        if select.select([sys.stdin], [], [], 0.0)[0]:
            return sys.stdin.read(1)
        return None


class G1Sim2SimPolicy:
    def __init__(self, checkpoint_path, device="cpu"):
        self._device = device
        sd = torch.load(checkpoint_path, map_location=device)

        self._obs_mean = sd["_obs_norm._mean"].to(device)
        self._obs_std = sd["_obs_norm._std"].to(device)
        self._act_mean = sd["_a_norm._mean"].to(device)
        self._act_std = sd["_a_norm._std"].to(device)

        self._l0_w = sd["_model._actor_layers.0.weight"].to(device)
        self._l0_b = sd["_model._actor_layers.0.bias"].to(device)
        self._l2_w = sd["_model._actor_layers.2.weight"].to(device)
        self._l2_b = sd["_model._actor_layers.2.bias"].to(device)
        self._mean_w = sd["_model._action_dist._mean_net.weight"].to(device)
        self._mean_b = sd["_model._action_dist._mean_net.bias"].to(device)
        return

    @torch.no_grad()
    def act(self, obs):
        norm_obs = (obs - self._obs_mean) / self._obs_std
        h = torch.relu(norm_obs @ self._l0_w.T + self._l0_b)
        h = torch.relu(h @ self._l2_w.T + self._l2_b)
        norm_action = h @ self._mean_w.T + self._mean_b
        action = norm_action * self._act_std + self._act_mean
        return action


class KeyboardCommand:
    """Emulates the left analog stick: w/x = longitudinal speed (fwd/back), a/d =
    turning rate (continuously integrates heading), matching how tar_dir/tar_speed
    are meant to be driven at deploy time (steer + throttle, not 4 fixed compass
    directions). Holding a key relies on the terminal's own key-repeat to keep
    sending the character while held; a single tap just nudges it once.
    """

    def __init__(self):
        self.theta = 0.0
        self.speed = 0.0
        return

    def handle_key(self, key):
        if key is None:
            return
        key = key.lower()
        if key == "w":
            self.speed = min(self.speed + SPEED_STEP, SPEED_MAX)
        elif key == "x":
            self.speed = max(self.speed - SPEED_STEP, 0.0)
        elif key == "a":
            self.theta = (self.theta + TURN_STEP + np.pi) % (2 * np.pi) - np.pi
        elif key == "d":
            self.theta = (self.theta - TURN_STEP + np.pi) % (2 * np.pi) - np.pi
        elif key == "s":
            self.speed = 0.0
        return

    def get_tar_dir(self):
        return np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float64)

    def get_tar_speed(self):
        return self.speed


def build_observation(kin_model, key_body_ids, mj_model, mj_data, tar_dir, tar_speed):
    root_pos = mj_data.qpos[0:3].copy()
    root_rot_xyzw = mj_quat_to_xyzw(mj_data.qpos[3:7])
    dof_pos = mj_data.qpos[7:7 + NUM_DOF].copy()

    root_vel = mj_data.qvel[0:3].copy()
    root_ang_vel_local = mj_data.qvel[3:6].copy()
    dof_vel = mj_data.qvel[6:6 + NUM_DOF].copy()

    t_root_pos = torch.tensor(root_pos, dtype=torch.float32).unsqueeze(0)
    t_root_rot = torch.tensor(root_rot_xyzw, dtype=torch.float32).unsqueeze(0)
    t_root_vel = torch.tensor(root_vel, dtype=torch.float32).unsqueeze(0)

    # qvel[3:6] for a MuJoCo free joint is in the root's local frame; rotate to world
    # frame since compute_char_obs applies its own heading-relative rotation on top.
    t_root_ang_vel_local = torch.tensor(root_ang_vel_local, dtype=torch.float32).unsqueeze(0)
    t_root_ang_vel = torch_util.quat_rotate(t_root_rot, t_root_ang_vel_local)

    t_dof_pos = torch.tensor(dof_pos, dtype=torch.float32).unsqueeze(0)
    t_dof_vel = torch.tensor(dof_vel, dtype=torch.float32).unsqueeze(0)

    joint_rot = kin_model.dof_to_rot(t_dof_pos)
    body_pos, _ = kin_model.forward_kinematics(t_root_pos, t_root_rot, joint_rot)
    key_pos = body_pos[..., key_body_ids, :]

    obs = char_env.compute_char_obs(
        root_pos=t_root_pos, root_rot=t_root_rot, root_vel=t_root_vel,
        root_ang_vel=t_root_ang_vel, joint_rot=joint_rot, dof_vel=t_dof_vel,
        key_pos=key_pos, global_obs=False, root_height_obs=True)

    t_tar_dir = torch.tensor(tar_dir, dtype=torch.float32).unsqueeze(0)
    t_tar_speed = torch.tensor([tar_speed], dtype=torch.float32)
    t_face_dir = t_tar_dir.clone()  # rand_face_dir: False in training -> face_dir == tar_dir

    steering_obs = task_steering_env.compute_steering_observations(t_root_rot, t_tar_dir, t_tar_speed, t_face_dir)
    obs = torch.cat([obs, steering_obs], dim=-1)
    return obs, root_pos, root_vel


def load_mj_model_with_ground(char_file):
    """Loads char_file with a MuJoCo-Menagerie-style scene injected (checkered
    groundplane texture/material, flat skybox, headlight -- same convention used by
    talos_2026/motion/twist3/TWIST2/assets/g1/scene.xml and most G1 MuJoCo repos in
    this workspace; the bare character asset has no floor/scale reference on its
    own). Also fixes up meshdir to stay absolute since we load from a temp copy in a
    different directory, and pins the physics timestep to match sim_freq in
    data/engines/isaac_lab_engine.yaml (g1.xml declares no <option>, so MuJoCo would
    otherwise default to 0.002s)."""
    with open(char_file, "r") as f:
        xml_str = f.read()
    meshdir = os.path.dirname(char_file)
    xml_str = xml_str.replace(
        '<compiler angle="radian"/>',
        '<compiler angle="radian" meshdir="{:s}"/>\n'
        '  <option timestep="{:.8f}" integrator="implicitfast"/>\n'
        '  <visual>\n'
        '    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>\n'
        '    <rgba haze="0.15 0.25 0.35 1"/>\n'
        '    <global azimuth="-140" elevation="-20"/>\n'
        '  </visual>'.format(meshdir, 1.0 / SIM_FREQ),
        1,
    )
    xml_str = xml_str.replace(
        "<asset>",
        '<asset>\n'
        '    <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>\n'
        '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" '
        'rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>\n'
        '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>\n',
        1,
    )
    xml_str = xml_str.replace(
        "<worldbody>",
        '<worldbody>\n    <geom name="ground" type="plane" size="0 0 0.05" material="groundplane"/>\n'
        '    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>\n',
        1,
    )
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    tmp_xml = os.path.join(tmp_dir, os.path.basename(char_file))
    with open(tmp_xml, "w") as f:
        f.write(xml_str)
    return mujoco.MjModel.from_xml_path(tmp_xml)


def load_init_pose(source, frame_idx=0):
    """'synthetic' -> the hand-authored STAND_POSE (off-distribution vs training,
    which always RSI-initializes mid-clip -- useful baseline but may be unstable).
    'walk'/'run' -> an actual frame from the corresponding training clip, exactly
    matching what Reference State Initialization would have sampled during training.
    Diagnostic: if 'walk'/'run' is stable but 'synthetic' isn't, the fall is about
    the initial pose being off-distribution, not a bug in this script."""
    if source == "synthetic":
        return np.array(INIT_POSE, dtype=np.float64)

    from anim.motion import load_motion
    clip = {"walk": "g1_walk.pkl", "run": "g1_run.pkl"}[source]
    motion_path = os.path.join(MIMICKIT_ROOT, "data/motions/g1", clip)
    motion = load_motion(motion_path)
    frame = motion.frames[frame_idx]
    return np.array(frame, dtype=np.float64)


def apply_pd_target(mj_model, mj_data, target_pos):
    """Sets the joints' native spring reference to target_pos and zeroes the motors,
    so MuJoCo's own implicit spring-damper integration (using the joint's declared
    stiffness/damping) holds the pose -- instead of a hand-rolled explicit PD loop
    that would double-count the same passive stiffness/damping MuJoCo already applies
    every step regardless of data.ctrl."""
    mj_model.qpos_spring[7:7 + NUM_DOF] = target_pos
    mj_data.ctrl[:] = 0.0
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init_pose", default="synthetic", choices=["synthetic", "walk", "run"],
                         help="'synthetic' = hand-authored stand pose (off-distribution vs training); "
                              "'walk'/'run' = an actual frame from that training clip (on-distribution, "
                              "useful to isolate whether a fall is about the initial pose or a real bug)")
    parser.add_argument("--pd_gain_scale", type=float, default=1.0,
                         help="multiplies jnt_stiffness/dof_damping -- diagnostic knob to test whether "
                              "MuJoCo's spring-damper needs stronger gains than PhysX's implicit actuator "
                              "to hold the same nominal stiffness/damping values under body weight")
    args = parser.parse_args()

    os.makedirs(VIDEO_OUT_DIR, exist_ok=True)

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

    renderer = mujoco.Renderer(mj_model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
    frames = []
    last_flush = time.time()
    dt = 1.0 / CONTROL_FREQ
    step_count = 0

    print("Controls: w=throttle up  x=throttle down  a=steer left  d=steer right  s=stop  q=quit")
    print("(keys read directly from this terminal, no Enter needed; hold for continuous effect)")

    with RawTerminal() as term:
        while True:
            key = term.read_key()
            if key is not None and key.lower() == "q":
                break
            cmd.handle_key(key)

            tar_dir = cmd.get_tar_dir()
            tar_speed = cmd.get_tar_speed()

            obs, root_pos, root_vel = build_observation(kin_model, key_body_ids, mj_model, mj_data, tar_dir, tar_speed)
            action = policy.act(obs).squeeze(0).numpy()

            target_pos = action  # zero_center_action: True in training -> action IS the joint target directly
            # Drive the joint's own native passive spring toward target_pos instead of
            # computing PD torque by hand -- the <joint stiffness=".." damping=".."/>
            # attributes are already applied by MuJoCo every step regardless of
            # data.ctrl, so a hand-rolled explicit PD loop on top of that double-counts
            # the same stiffness/damping and is also less numerically stable than
            # MuJoCo's implicit spring integration.
            apply_pd_target(mj_model, mj_data, target_pos)
            for _ in range(SIM_SUBSTEPS):
                mujoco.mj_step(mj_model, mj_data)

            step_count += 1
            if step_count % 5 == 0:
                renderer.update_scene(mj_data)
                frames.append(renderer.render().copy())

            speed_xy = np.linalg.norm(root_vel[0:2])
            print("\rcmd: dir={:+.0f}deg speed={:.1f} | measured: speed={:.2f} height={:.2f}   ".format(
                np.degrees(cmd.theta), tar_speed, speed_xy, root_pos[2]), end="", flush=True)

            if time.time() - last_flush > VIDEO_FLUSH_SECONDS and len(frames) > 0:
                out_path = os.path.join(VIDEO_OUT_DIR, "keyboard_test_{:d}.mp4".format(int(time.time())))
                import util.video as video
                vid = video.Video(fps=CONTROL_FREQ / 5)
                for f in frames:
                    vid.add_frame(f)
                vid.save(out_path)
                print("\n[saved video: {:s}]".format(out_path))
                frames = []
                last_flush = time.time()

            time.sleep(dt)

    print("\nDone.")
    return


if __name__ == "__main__":
    main()
