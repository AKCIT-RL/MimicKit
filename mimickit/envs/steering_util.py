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


def build_dof_group_ids(kin_char_model, body_names):
    """DOF indices of the joints attached to body_names, in asset order.

    Used to split the action vector into the groups Table 4 weighs differently
    (head -15, legs -1). Returns a plain list of ints.

    BY NAME, never by index. The three torque profiles (t1.xml,
    t1_catalog_peak.xml, t1_firmware_derated.xml) share their joints today, so
    a hardcoded index would work -- right up until an asset reorders, at which
    point a reward term would start weighing the wrong joints and nothing would
    say so. A name that stops existing raises instead: get_body_id asserts on
    an unknown body.

    get_body_id, NOT get_joint_id. The latter returns body_id - 1 because it
    indexes arrays that exclude the root, while self._joints includes it, so
    get_joint(get_joint_id(name)) hands back the joint of the PREVIOUS body.
    On the T1 that silently shifts every group by one and drags in the fixed
    camera joint (dof_dim 0) -- the dof_dim assert below is what catches it.
    """
    ids = []
    for name in body_names:
        j = kin_char_model.get_body_id(name)
        idx = kin_char_model.get_joint_dof_idx(j)
        dim = kin_char_model.get_joint_dof_dim(j)
        assert dim == 1, \
            "expected a 1-DOF hinge at '{}', got dof_dim {}".format(name, dim)
        ids.extend(range(idx, idx + dim))

    assert len(set(ids)) == len(ids), \
        "overlapping DOF groups in {}".format(body_names)
    return ids


# config key -> slice of the measurable frame it corrupts. prev_action and the
# steering command are absent ON PURPOSE: the first is the policy's own output
# and is exact on the robot, the second is a command, not a measurement.
OBS_NOISE_KEYS = ("obs_noise_gravity", "obs_noise_ang_vel",
                  "obs_noise_dof_pos", "obs_noise_dof_vel")


def build_obs_noise_std(env_config, num_dofs, task_dim):
    """Per-dimension std of the actor's sensor noise, [6 + 3D + task_dim].

    Layout matches compute_proprio_frame + the task block:
      gravity(3) ang_vel(3) dof_pos(D) dof_vel(D) prev_action(D) task(task_dim)
    Each obs_noise_* key is one std for its whole block (default 0), times
    obs_noise_scale (default 1), which exists so an evaluation can sweep the
    level without editing the four stds. A single std per block is also what
    keeps the noise mirror-symmetric: the frame's mirror map only permutes
    and flips signs WITHIN a block.

    Gaussian and white on purpose -- the minimum model. Nothing here has been
    measured on the robot yet; see the env yaml for where the values came from.
    """
    std = {k: float(env_config.get(k, 0.0)) for k in OBS_NOISE_KEYS}
    scale = float(env_config.get("obs_noise_scale", 1.0))
    for k, v in std.items():
        assert v >= 0.0, "{} must be >= 0, got {}".format(k, v)
    assert scale >= 0.0, "obs_noise_scale must be >= 0, got {}".format(scale)

    blocks = [torch.full([3], std["obs_noise_gravity"]),
              torch.full([3], std["obs_noise_ang_vel"]),
              torch.full([num_dofs], std["obs_noise_dof_pos"]),
              torch.full([num_dofs], std["obs_noise_dof_vel"]),
              torch.zeros([num_dofs]),
              torch.zeros([task_dim])]
    return scale * torch.cat(blocks)


def apply_obs_noise(frame, noise_std):
    """Additive white Gaussian noise, frame [N, F] + noise_std [F] * N(0, 1).

    Returns the input tensor itself (not a copy) when every std is zero, so a
    disabled operator is bit-exact disabled -- no randn is drawn and the RNG
    stream the rest of the run consumes is untouched.
    """
    if (not bool(torch.any(noise_std > 0))):
        return frame
    return frame + noise_std * torch.randn_like(frame)
