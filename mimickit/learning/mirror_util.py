"""Left-right mirror operators for the observation and action spaces.

Used by the mirror symmetry loss of
"Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots"
(arXiv:2511.03996, Table 1):

    L_sym = || mu(o) - M_a( mu( M_o(o) ) ) ||^2

M_o and M_a reflect the character across its sagittal (XZ) plane, so the map
is a permutation (swap left and right slots) plus a per-slot sign.

Both maps are built from the kinematic model the environment itself loaded, so
a change of embodiment or of the asset cannot silently desynchronise them from
the observation the policy actually sees.

Sign derivation, for M = diag(1, -1, 1):

  scalars (root height, target speed)   unchanged
  true vectors (positions, velocities)  (x, -y, z)
  angular velocity                      (-x, y, -z)   pseudovector: det(M) = -1
  tan-norm rotations                    y flips in BOTH halves

The last one is the trap. `quat_to_tan_norm` uses ref_tan = e_x and
ref_norm = e_z (torch_util.py), and M fixes both, so tan' = M R M e_x = M tan
and norm' = M R M e_z = M norm: a plain y flip on each half. Had ref_norm been
e_y - the natural guess - the norm half would need (-x, y, -z) instead, and an
involution test would pass either way. Do not "simplify" this by assuming the
two halves must differ.

Observation layout mirrored here (char_env.compute_char_obs):

    [root_h] root_rot(6) root_vel(3) root_ang_vel(3)
    joint_rot(num_joints-1 x 6) dof_vel(dof_size) [key_pos(K x 3)] task(...)

Note that the joint_rot block has one entry per non-root joint INCLUDING
zero-DOF (FIXED) joints, while dof_vel has one entry per DOF. On the G1 these
differ - a FIXED `head_link` sits between the waist and the left shoulder - so
the two blocks need separate permutations. Deriving both from the model rather
than from a hand-written table is what keeps them aligned.
"""

import numpy as np
import torch


def mirror(x, perm, signs):
    """Apply a mirror map to the last axis of x.

    Args:
        x: [..., N] tensor.
        perm: [N] long tensor, gather indices.
        signs: [N] tensor of +1/-1.
    """
    return x[..., perm] * signs


def _partner_name(name):
    if (name.startswith("left_")):
        return "right_" + name[len("left_"):]
    if (name.startswith("right_")):
        return "left_" + name[len("right_"):]
    return name


def _pair_indices(names, what):
    """Index permutation that swaps left/right entries of a name list."""
    lookup = {n: i for i, n in enumerate(names)}
    perm = []
    for name in names:
        partner = _partner_name(name)
        if (partner not in lookup):
            raise ValueError("{} '{}' has no mirror partner '{}'".format(what, name, partner))
        perm.append(lookup[partner])
    return perm


def _hinge_sign(axis):
    """+1 for a hinge about y, -1 for a hinge about x or z.

    A reflection across XZ preserves rotations about y and reverses rotations
    about x and z. Non-canonical axes are rejected rather than approximated:
    a hinge tilted off the principal axes does not mirror to a scalar sign.
    """
    if (torch.is_tensor(axis)):
        # the live environment builds its model on the training device
        axis = axis.detach().cpu()
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    for idx, sign in ((0, -1.0), (1, 1.0), (2, -1.0)):
        if (abs(abs(axis[idx]) - 1.0) < 1e-6):
            return sign
    raise ValueError("hinge axis {} is not aligned with a principal axis; "
                     "its mirror is not a per-DOF sign flip".format(axis.tolist()))


def build_dof_mirror(char_model):
    """Mirror map over the DOF vector: dof_pos, dof_vel and the action.

    Returns:
        (perm, signs) as plain lists of length dof_size.
    """
    names, signs_by_dof, dof_of_joint = [], [], []
    for j in range(1, char_model.get_num_joints()):
        joint = char_model.get_joint(j)
        dof_dim = joint.get_dof_dim()
        if (dof_dim == 0):
            continue
        if (dof_dim != 1):
            raise ValueError("joint '{}' has {} DOFs; only 1-DOF hinges are "
                             "supported by the sign-flip mirror".format(joint.name, dof_dim))
        names.append(joint.name)
        signs_by_dof.append(_hinge_sign(joint.axis))
        dof_of_joint.append(joint.dof_idx)

    if (dof_of_joint != sorted(dof_of_joint)):
        raise ValueError("DOF indices are not in joint order; the mirror would be misaligned")
    if (len(names) != char_model.get_dof_size()):
        raise ValueError("counted {} 1-DOF joints but the model reports dof_size {}".format(
            len(names), char_model.get_dof_size()))

    slot_perm = _pair_indices(names, "joint")

    # a partner pair must agree on its sign, otherwise the map is not an involution
    for i, p in enumerate(slot_perm):
        if (signs_by_dof[i] != signs_by_dof[p]):
            raise ValueError("joints '{}' and '{}' disagree on mirror sign".format(
                names[i], names[p]))

    perm = [dof_of_joint[p] for p in slot_perm]
    return perm, signs_by_dof


def build_joint_rot_mirror(char_model):
    """Mirror map over the joint_rot tan-norm block (6 numbers per joint).

    Covers every non-root joint, zero-DOF ones included, matching
    KinCharModel.dof_to_rot which emits num_joints - 1 rotations.
    """
    names = [char_model.get_joint(j).name for j in range(1, char_model.get_num_joints())]
    slot_perm = _pair_indices(names, "joint")

    perm, signs = [], []
    for p in slot_perm:
        base = 6 * p
        perm.extend([base + 0, base + 1, base + 2, base + 3, base + 4, base + 5])
        signs.extend([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])   # y flips in tan and norm
    return perm, signs


def build_key_body_mirror(key_body_names):
    """Mirror map over the flattened key-body position block (3 per body)."""
    slot_perm = _pair_indices(list(key_body_names), "key body")
    perm, signs = [], []
    for p in slot_perm:
        base = 3 * p
        perm.extend([base + 0, base + 1, base + 2])
        signs.extend([1.0, -1.0, 1.0])
    return perm, signs


# Task observation blocks, appended after the character block by each task env.
# Sign only - no task block reorders its own slots.
#
# Keyed by environment class name, which is readable from the live env object.
# Adding the soccer task at the S5 merge is a single entry here; nothing in
# mimickit/envs/ has to change.
TASK_OBS_MIRROR = {
    # task_steering_env.compute_steering_observations:
    #   local_tar_dir (2), tar_speed (1), local_face_dir (2)
    "TaskSteeringEnv": [1.0, -1.0, 1.0, 1.0, -1.0],
    # task_soccer_env._compute_task_block: the steering block above, then
    # soccer_util.compute_soccer_observations — local_ball (2), local_goal (2),
    # local_goal_dir (2), all heading-frame planar (x, y) pairs — and the ball
    # detection mask (1), which is side-blind.
    "TaskSoccerEnv": [1.0, -1.0, 1.0, 1.0, -1.0,
                      1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
                      1.0],
}


def build_char_obs_mirror(char_model, key_body_names, root_height_obs):
    """Mirror map over the character observation block."""
    perm, signs = [], []

    def add(block_perm, block_signs):
        offset = len(perm)
        perm.extend([offset + p for p in block_perm])
        signs.extend(block_signs)

    if (root_height_obs):
        add([0], [1.0])                                          # root height
    add(list(range(6)), [1.0, -1.0, 1.0, 1.0, -1.0, 1.0])        # root rot tan-norm
    add(list(range(3)), [1.0, -1.0, 1.0])                        # root linear velocity
    add(list(range(3)), [-1.0, 1.0, -1.0])                       # root angular velocity

    jr_perm, jr_signs = build_joint_rot_mirror(char_model)
    add(jr_perm, jr_signs)

    dof_perm, dof_signs = build_dof_mirror(char_model)
    add(dof_perm, dof_signs)                                     # dof_vel

    if (len(key_body_names) > 0):
        kb_perm, kb_signs = build_key_body_mirror(key_body_names)
        add(kb_perm, kb_signs)

    return perm, signs


def build_obs_mirror(task_key, char_model, key_body_names, root_height_obs, obs_size):
    """Full observation mirror: character block followed by the task block.

    obs_size is checked, not trusted: a layout drift on the environment side
    shows up here as an immediate error instead of a policy that quietly
    learns the wrong symmetry.
    """
    perm, signs = build_char_obs_mirror(char_model, key_body_names, root_height_obs)

    if (task_key not in TASK_OBS_MIRROR):
        raise KeyError("no task observation mirror registered for '{}'. Add its "
                       "block to mirror_util.TASK_OBS_MIRROR.".format(task_key))
    task_signs = TASK_OBS_MIRROR[task_key]
    offset = len(perm)
    perm.extend(range(offset, offset + len(task_signs)))
    signs.extend(task_signs)

    if (len(perm) != obs_size):
        raise ValueError(
            "mirror covers {} dims but the environment observation is {}. The "
            "layout changed; fix mirror_util before training.".format(len(perm), obs_size))
    return perm, signs


def to_tensors(perm, signs, device, dtype=torch.float32):
    perm_t = torch.as_tensor(perm, dtype=torch.long, device=device)
    signs_t = torch.as_tensor(signs, dtype=dtype, device=device)
    return perm_t, signs_t


def check_involution(perm, signs):
    """True when applying the map twice is the identity.

    Necessary but far from sufficient: a wrong sign on a symmetric pair
    survives this test. The fixed-point and physical tests are what catch that.
    """
    n = len(perm)
    for i in range(n):
        if (perm[perm[i]] != i):
            return False
        if (abs(signs[i] * signs[perm[i]] - 1.0) > 1e-9):
            return False
    return True


def check_normalizer_equivariance(mean, std, perm, signs, tol=1e-5):
    """Whether a normalizer commutes with the mirror map.

    Mirroring in normalized space equals mirroring in raw space only when
    mirror(mean) == mean and std[perm] == std. It holds for the G1 action
    normalizer because the MJCF joint limits are exact mirror images
    (left_hip_roll [-0.5236, 2.9671] vs right_hip_roll [-2.9671, 0.5236]), but
    that is a property of the asset, not a guarantee - hence this check.

    Returns (ok, mean_err, std_err).
    """
    perm_t = torch.as_tensor(perm, dtype=torch.long, device=mean.device)
    signs_t = torch.as_tensor(signs, dtype=mean.dtype, device=mean.device)

    mean_err = torch.max(torch.abs(mean[perm_t] * signs_t - mean)).item()
    std_err = torch.max(torch.abs(std[perm_t] - std)).item()
    return (mean_err <= tol and std_err <= tol), mean_err, std_err
