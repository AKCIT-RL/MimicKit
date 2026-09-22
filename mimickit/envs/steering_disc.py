"""Discriminator observation for the steering task (arXiv:2511.03996, Table 3).

The Disc. column of Table 3, and only it: projected gravity, base angular
velocity, joint position offset from the default, joint velocity, base linear
velocity, and the feet in the torso frame. No root position, no previous
action, no head or elbow key bodies.

TWO DIFFERENCES FROM THE GENERIC MimicKit FORMULATION IT REPLACES

  features   amp_env.compute_disc_obs sends 197 numbers per frame: root
             position, a 6D tan-norm root rotation, a 6D tan-norm rotation per
             joint (144), five key bodies, and the velocities. This is 61.
             The trajectory of the root is not lost by dropping its position -
             it is carried by the per-frame linear velocity instead.

  frame      the generic one expresses the WHOLE window in the heading frame
             (yaw only) of a single reference frame, the newest in the window,
             so every frame reads as a displacement from the current state.
             Table 3 says "in the robot frame", so here each frame is rotated
             by the conjugate of ITS OWN root rotation and is self-contained -
             the same convention steering_util.compute_proprio_frame uses for
             the actor. Position and heading invariance still hold, because
             after the change no absolute quantity survives.

Not jit-scripted on purpose: it runs once per step for the agent and once per
batch for the demo, and the un-scripted version is easier to keep correct
through the reshapes.
"""

import torch

import util.torch_util as torch_util

NUM_FEET = 2
# proj gravity (3) + ang vel (3) + joint offset (D) + joint vel (D)
# + lin vel (3) + feet (NUM_FEET * 3)
FRAME_DIM_BASE = 3 + 3 + 3 + NUM_FEET * 3


def frame_dim(num_dofs, num_feet=NUM_FEET):
    return 3 + 3 + 2 * num_dofs + 3 + 3 * num_feet


def compute_disc_obs(root_pos, root_rot, root_vel, root_ang_vel,
                     dof_pos, dof_vel, foot_pos, init_dof_pos):
    """One window of discriminator frames, flattened.

    Shapes in: root_pos/root_vel/root_ang_vel [N, T, 3], root_rot [N, T, 4]
    (xyzw), dof_pos/dof_vel [N, T, D], foot_pos [N, T, F, 3] in WORLD, and
    init_dof_pos [D]. Out: [N, T * frame_dim].

    The same function serves the agent and the motion demo; that is the point,
    since a discriminator only means something if both sides are built the
    same way.
    """
    n, t = root_rot.shape[0], root_rot.shape[1]
    num_feet = foot_pos.shape[-2]

    flat_rot = root_rot.reshape(n * t, 4)
    inv_rot = torch_util.quat_conjugate(flat_rot)

    gravity = torch.zeros(n * t, 3, dtype=root_rot.dtype, device=root_rot.device)
    gravity[..., 2] = -1.0
    proj_gravity = torch_util.quat_rotate(inv_rot, gravity)

    ang_vel = torch_util.quat_rotate(inv_rot, root_ang_vel.reshape(n * t, 3))
    lin_vel = torch_util.quat_rotate(inv_rot, root_vel.reshape(n * t, 3))

    dof_offset = (dof_pos - init_dof_pos).reshape(n * t, -1)
    joint_vel = dof_vel.reshape(n * t, -1)

    # feet relative to the root, then into the base frame: "relative position
    # of the feet in the torso frame"
    rel_foot = foot_pos - root_pos.unsqueeze(-2)
    rel_flat = rel_foot.reshape(n * t * num_feet, 3)
    inv_expand = inv_rot.unsqueeze(-2).repeat(1, num_feet, 1).reshape(n * t * num_feet, 4)
    foot_local = torch_util.quat_rotate(inv_expand, rel_flat).reshape(n * t, num_feet * 3)

    frame = torch.cat([proj_gravity, ang_vel, dof_offset, joint_vel,
                       lin_vel, foot_local], dim=-1)
    return frame.reshape(n, -1)
