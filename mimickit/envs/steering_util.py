"""Steering task observation blocks (arXiv:2511.03996, Table 3).

Pure-torch, batch-first functions for the measurable steering env. Split along
the line the paper draws: `compute_proprio_frame` returns only what the T1 can
read on board (IMU + joint encoders + the policy's own last output), while
`compute_privileged_block` and `compute_root_lin_vel_b` return simulator-only
state, which is why they feed the critic and the decoder target rather than the
actor. No simulator dependencies.

Deliberately independent from envs/soccer_util.py: that module belongs to the
soccer track and carries ball/goal semantics, so the two evolve separately.
"""

import torch

import util.torch_util as torch_util


@torch.jit.script
def compute_proprio_frame(root_rot, root_ang_vel, dof_pos, dof_vel, prev_action,
                          init_dof_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    """Measurable proprioceptive frame (paper Table 3, Actor column).

    Projected gravity and base angular velocity in the BASE frame (full
    rotation removed, which is what an onboard IMU reports), joint position
    offsets from the default pose, joint velocities and the previous action.
    No linear velocity, no base height, no key-body positions.
    Returns [N, 6 + 3 * D].
    """
    inv_rot = torch_util.quat_conjugate(root_rot)

    gravity = torch.zeros_like(root_ang_vel)
    gravity[..., 2] = -1.0
    proj_gravity = torch_util.quat_rotate(inv_rot, gravity)

    local_ang_vel = torch_util.quat_rotate(inv_rot, root_ang_vel)
    dof_offset = dof_pos - init_dof_pos

    return torch.cat([proj_gravity, local_ang_vel, dof_offset, dof_vel, prev_action],
                     dim=-1)


@torch.jit.script
def compute_root_lin_vel_b(root_rot, root_vel):
    # type: (Tensor, Tensor) -> Tensor
    """Root linear velocity in the BASE frame. Decoder reconstruction target
    (paper Table 3, Recon. column) and part of the privileged critic block.

    Same frame convention as the angular velocity in compute_proprio_frame --
    the full rotation is removed, not just the heading. Keeping the pair in one
    frame is what lets the decoder estimate this from the proprioceptive
    history; mixing a heading frame in here would make the target depend on
    roll/pitch that the input encodes differently.
    Returns [N, 3].
    """
    return torch_util.quat_rotate(torch_util.quat_conjugate(root_rot), root_vel)


@torch.jit.script
def compute_privileged_block(root_pos, root_rot, root_vel):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """Simulator-only state for the asymmetric critic (paper Table 3, the rows
    marked Critic and not Actor): base-frame linear velocity and base height.

    Mass randomization, the third critic-only row of the table, is absent
    because the steering env has no domain randomization yet.
    Returns [N, 4].
    """
    lin_vel = compute_root_lin_vel_b(root_rot, root_vel)
    root_h = root_pos[..., 2:3]
    return torch.cat([lin_vel, root_h], dim=-1)
