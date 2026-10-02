"""Regularization rewards for the steering task (arXiv:2511.03996, Table 4).

Pure torch, batch-first, no simulator state, so each term can be asserted on
its own in CPU tests.

Only the rows of Table 4 that apply to steering land here. What is deliberately
absent:

  Stagnation   the paper penalizes standing still (-100 after 1 s nearly
               motionless) because its robot must always chase the ball.
               Steering is commanded to stop, so copying that term would fight
               the task. Not implemented, on purpose.

  Arm action   Table 4 has action-rate terms for the HEAD (-15) and the LEGS
  rate         (-1) and none for the arms. Not adding one is a decision, not an
               oversight: the arm behaviour the campaign targets is covered by
               the collision term instead.

The paper gives a description and a weight per row, never a formula, so every
functional form below is a choice. Each one says which choice it made and what
the alternative would have cost, because the weight only means something
against the shape it multiplies.

A weight is also not transferable on its own: what sets the strength of a term
is weight x the raw value it multiplies, against a steering task reward of
order 1. foot_proximity came in at the paper's -5 and measured -0.0125/step,
1.5% of the task reward -- nearly inert. Every term here is measured raw before
a weight is chosen.

Deliberately independent from envs/soccer_util.py, which belongs to the soccer
track, even though the arithmetic of three of these is the same. A change made
there must not silently alter a run made here.
"""

import math

import torch


@torch.jit.script
def compute_foot_proximity_penalty(left_foot_pos, right_foot_pos, min_dist):
    # type: (Tensor, Tensor, float) -> Tensor
    """Positive magnitude when the feet are closer than min_dist (planar).

    Zero once the feet are at least min_dist apart, so it is a one-sided
    penalty: it pushes the stance open and then stops acting. Paper weight -5.
    """
    d = torch.linalg.norm(left_foot_pos[..., 0:2] - right_foot_pos[..., 0:2], dim=-1)
    return torch.clamp_min(min_dist - d, 0.0)


@torch.jit.script
def compute_action_rate_penalty(action, prev_action):
    # type: (Tensor, Tensor) -> Tensor
    """Sum of squared per-step action changes over whatever DOFs are passed in.

    The caller slices the group (head / leg) before calling, so which joints
    count is an env concern and this stays an operator that can be asserted on
    its own. Paper weights: head -15, legs -1.

    QUADRATIC, not |.|: the target is a spasm, which is rare and large, not the
    dither of a normal gait. A square punishes the rare event super-linearly
    and leaves small motion nearly free. L1 does the opposite -- its derivative
    is constant, so moving 0.01 rad costs proportionally as much as moving
    0.5 rad, which rewards saving up movement into fewer, bigger steps.

    SUMMED over DOFs, not averaged. The paper's 15:1 ratio between head (2 DOFs)
    and legs (12 DOFs) was set on this same robot against unnormalized sums.
    Averaging would turn the effective ratio into 15*(12/2):1 = 90:1, and the
    weights would no longer be the paper's.

    FIRST difference, not second. A one-step oscillation a -> -a -> a already
    scores 4a^2 here; jerk is not needed to see it, and it is logged as a
    diagnostic instead of being paid for twice.

    CALLER BEWARE: with `zero_center_action`, our action is an ABSOLUTE joint
    target in radians, so this comes out in rad^2. If the paper's action is a
    scaled offset, its weight is in different units and cannot be copied
    without converting -- which is why the raw value is measured first.
    """
    return torch.sum(torch.square(action - prev_action), dim=-1)


@torch.jit.script
def compute_joint_limit_penalty(dof_pos, soft_low, soft_high):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """One-sided linear excursion past the soft limits, summed over the DOFs.

    Exactly zero strictly inside [soft_low, soft_high]: this is a guard band,
    not a posture prior, so it has to stop acting once the joint is safe.
    soft_low/soft_high broadcast over the batch: [D] or [N, D]. Paper weight
    -100.

    LINEAR, not quadratic. The physics engine already enforces the asset's
    joint limits, so a real excursion is on the order of 1e-3 rad; squared that
    is 1e-6 and no weight rescues it. Linear keeps a usable gradient at the
    edge, which is the only place the term is supposed to act.

    ANY guard band fights the demonstrations, so the caller's default margin is
    0 and the band is opt-in. Measured over the 16 motions of
    dataset_t1_locomotion_wrturn, the knee (asset range [0, 2.145]) has
    min 0.0000, p01 0.0000, p05 0.0435, median 0.4513 rad -- it sits at the
    hard lower limit for part of every stride. Fraction of reference frames a
    band would penalize:

        ratio 0.9 around the midpoint   10.36%   (soft floor 0.107 rad)
        absolute margin 0.05 rad         5.54%
        absolute margin 0.02 rad         2.65%
        margin 0                         0%

    A nonzero margin therefore penalizes exactly the pose the AMP discriminator
    is paying the policy to reproduce. If a band is ever wanted it must be an
    ABSOLUTE margin in radians, never a ratio around the midpoint: the knee
    range is one-sided, so a ratio puts its floor at 0.107 rad and punishes a
    normally extended knee hardest.

    With margin 0 this measures genuine excursion past the hard limit, which
    the physics engine already clamps -- so it may well read ~0 and be inert.
    That is a measurement to make, not a reason to widen the band: the quantity
    that actually saturates is the COMMANDED target, which nothing clamps.
    """
    below = torch.clamp_min(soft_low - dof_pos, 0.0)
    above = torch.clamp_min(dof_pos - soft_high, 0.0)
    return torch.sum(below + above, dim=-1)


@torch.jit.script
def compute_base_accel_penalty(root_vel, prev_root_vel, dt):
    # type: (Tensor, Tensor, float) -> Tensor
    """Squared magnitude of the base linear acceleration, in (m/s^2)^2.

    Paper weight -0.001. 3D on purpose: the z component is what penalizes a
    bouncing gait, and dropping it would leave half the term's job undone.
    Not normalized per axis -- the weight multiplies a sum of three.

    dt is the CONTROL step, so prev_root_vel has to be the velocity captured in
    _pre_physics_step of this same step. It must also be re-seeded on reset:
    the character is teleported to a motion frame, so a stale value makes the
    first step of every episode pay |v_new - v_old| / dt, which at 2.5 m/s and
    dt = 1/30 is ~5600 -- a constant bias indistinguishable from the term
    working.

    Scales with control frequency: the same velocity step at 50 Hz scores
    (50/30)^2 = 2.8x what it scores at 30 Hz. A weight tuned at one frequency
    is silently wrong at the other.
    """
    accel = (root_vel - prev_root_vel) / dt
    return torch.sum(torch.square(accel), dim=-1)


@torch.jit.script
def compute_collision_penalty(contact_forces, body_ids, force_thresh):
    # type: (Tensor, Tensor, float) -> Tensor
    """1.0 on any step where a non-exempt body carries contact force, else 0.

    contact_forces: [N, B, 3] NET cartesian force per body, newtons, world
    frame. body_ids: [K], the bodies that must not touch anything. Paper weight
    -100, described as "collision on body parts except the feet".

    BINARY, not force-proportional and not counted per body. A contact impulse
    spikes to kilonewtons for one step, so a magnitude-weighted term has
    unbounded variance and a single graze can outweigh an entire episode. And
    because these are NET forces, both sides of a self-contact register, so
    summing over bodies double-counts every pair -- at -100 an arm touching
    torso and hip would cost -300, worse than falling over. The -100 reads as a
    constant per event.

    Use the RAW contact forces here, never the engine's ground-filtered ones:
    those zero every body above ground_contact_height (0.3 m), which is exactly
    where an arm-into-hip contact lives.
    """
    f = contact_forces.index_select(-2, body_ids)
    mag = torch.linalg.norm(f, dim=-1)
    return torch.any(mag > force_thresh, dim=-1).to(contact_forces.dtype)


@torch.jit.script
def compute_collision_count(contact_forces, body_ids, force_thresh):
    # type: (Tensor, Tensor, float) -> Tensor
    """Diagnostic twin of compute_collision_penalty: HOW MANY bodies are in
    contact.

    Not a reward. It exists because the binary term cannot distinguish "fires
    on every step because a real event is constant" from "fires on every step
    because two adjacent collision primitives permanently interpenetrate". A
    count pinned near K is the signature of the second, and without it that
    failure looks like a term that is simply always on.
    """
    f = contact_forces.index_select(-2, body_ids)
    mag = torch.linalg.norm(f, dim=-1)
    return torch.sum((mag > force_thresh).to(contact_forces.dtype), dim=-1)


# ----------------------------------------------------------------------------
# Gait terms from Booster Gym (arXiv 2506.15132, the framework the paper cites as
# [45]; booster_gym/envs/t1.py). NOT in the paper's own reward table, whose only
# foot term is foot proximity: these are adopted for two symptoms seen on the
# real robot (a dragging foot; the foot turning inward above ~0.8 m/s), not for
# fidelity. Booster Gym's weights do not transfer -- its stream is single and
# clipped positive -- so every term is measured raw before a weight is chosen.
# Foot order everywhere: index 0 = left, 1 = right.
# ----------------------------------------------------------------------------


@torch.jit.script
def wrap_to_pi(x):
    # type: (Tensor) -> Tensor
    return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi


@torch.jit.script
def compute_feet_slip_penalty(contact_vel_xy, contact):
    # type: (Tensor, Tensor) -> Tensor
    """Sum over feet of the squared planar contact-point speed while in contact.

    contact_vel_xy [N, F, 2] (steering_util.compute_sole_contact_state),
    contact [N, F] bool. Booster Gym _reward_feet_slip, weight -0.1, with two
    deliberate differences:

      POINT    the lowest box corner, not the body origin. Booster Gym
               differences the origin, which during the heel-to-toe roll moves
               with the toe planted -- measured on our policies as up to
               1.7 m/s of slip that is not there. Penalizing that would punish
               the roll, not the drag.
      PLANAR   x and y only. A foot dragging is a foot moving ALONG the ground;
               the vertical speed at touchdown and liftoff is the step itself.

    The contact rule is Booster Gym's: geometric, the sole within 1 cm of the
    ground (it tests four sole corners against 0.01 m; this tests the lowest of
    the eight box corners, the same quantity). A foot grazing the ground in
    mid-swing is exactly "in contact and moving", so it pays here.
    """
    speed_sq = torch.sum(torch.square(contact_vel_xy), dim=-1)
    return torch.sum(speed_sq * contact.to(speed_sq.dtype), dim=-1)


@torch.jit.script
def compute_feet_yaw_diff_penalty(foot_yaw):
    # type: (Tensor) -> Tensor
    """(yaw_right - yaw_left)^2, wrapped. Booster Gym _reward_feet_yaw_diff, -1.

    foot_yaw [N, 2] world yaw of each foot. Zero when the feet are parallel,
    whatever their common heading. A symmetric toe-in (both points inward)
    makes the feet NON-parallel, so it scores here -- unlike feet_yaw_mean,
    which averages the two feet and cancels it out.
    """
    return torch.square(wrap_to_pi(foot_yaw[..., 1] - foot_yaw[..., 0]))


@torch.jit.script
def compute_feet_yaw_mean_penalty(base_yaw, foot_yaw):
    # type: (Tensor, Tensor) -> Tensor
    """(base_yaw - mean foot yaw)^2, wrapped. Booster Gym _reward_feet_yaw_mean.

    Measured only, not trained: it is blind to a symmetric toe-in (the mean of
    +a and -a is 0), and what it does see -- the torso twisting against the
    feet -- is the per-step hip yaw the reference motions themselves contain.
    The +pi branch is Booster Gym's, for feet whose yaws straddle the wrap.
    """
    straddle = torch.abs(foot_yaw[..., 1] - foot_yaw[..., 0]) > math.pi
    mean = foot_yaw.mean(dim=-1) + math.pi * straddle.to(foot_yaw.dtype)
    return torch.square(wrap_to_pi(base_yaw - mean))


@torch.jit.script
def compute_feet_roll_penalty(foot_roll):
    # type: (Tensor) -> Tensor
    """Sum of squared foot roll. Booster Gym _reward_feet_roll. Measured only."""
    return torch.sum(torch.square(foot_roll), dim=-1)


@torch.jit.script
def compute_feet_lateral_distance(base_yaw, left_pos, right_pos):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """Lateral (sideways) separation of the feet in the base heading frame, m.

    Booster Gym's formula. Unlike the planar Euclidean distance that
    foot_proximity uses, this ignores how far one foot is AHEAD of the other,
    so a long stride cannot satisfy it with the feet on one line.
    """
    dx = right_pos[..., 0] - left_pos[..., 0]
    dy = right_pos[..., 1] - left_pos[..., 1]
    return torch.abs(torch.cos(base_yaw) * dy - torch.sin(base_yaw) * dx)


@torch.jit.script
def compute_feet_lateral_distance_penalty(base_yaw, left_pos, right_pos, ref, cap):
    # type: (Tensor, Tensor, Tensor, float, float) -> Tensor
    """clip(ref - lateral separation, 0, cap). Booster Gym _reward_feet_distance
    (ref 0.2 m, cap 0.1 m, weight -1). One-sided: zero once the feet are ref
    apart sideways. Added ON TOP of foot_proximity (decision 2026-10-01), whose
    Euclidean distance a long stride satisfies by fore-aft separation alone."""
    d = compute_feet_lateral_distance(base_yaw, left_pos, right_pos)
    return torch.clamp(ref - d, min=0.0, max=cap)


@torch.jit.script
def compute_feet_swing_reward(phase, freq, contact, swing_period):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """Booster Gym _reward_feet_swing (weight +3): 1 per foot that is OFF the
    ground inside its swing window of the commanded gait clock.

    phase/freq [N], contact [N, 2] bool (left, right). Left window centred at
    phase 0.25, right at 0.75, half-width swing_period / 2 (Booster Gym 0.2).
    Zero while the clock is stopped. A REWARD, positive: it pays the foot for
    being in the air when the clock says swing, which is Booster Gym's only
    anti-drag mechanism -- it has no foot-height term.
    """
    half = 0.5 * swing_period
    running = freq > 0.0
    left = (torch.abs(phase - 0.25) < half) & running
    right = (torch.abs(phase - 0.75) < half) & running
    return ((left & ~contact[..., 0]).to(phase.dtype)
            + (right & ~contact[..., 1]).to(phase.dtype))


@torch.jit.script
def compute_toe_in(base_yaw, foot_yaw):
    # type: (Tensor, Tensor) -> Tensor
    """Signed toe-in of each foot relative to the base heading, rad, [N, 2].

    Positive = the toe points INWARD on either side (left foot yawed right,
    right foot yawed left), so the two columns are comparable and a mirror-
    symmetric gait gives equal values. A metric, not a reward.
    """
    rel = wrap_to_pi(foot_yaw - base_yaw.unsqueeze(-1))
    return torch.stack([-rel[..., 0], rel[..., 1]], dim=-1)


@torch.jit.script
def compute_location_reward(root_pos, prev_root_pos, tar_pos, tar_speed, dt,
                            pos_err_scale, vel_err_scale, pos_w, vel_w, stop_radius):
    # type: (Tensor, Tensor, Tensor, float, float, float, float, float, float, float) -> Tensor
    """Target-location reward of AMP (Peng et al. 2021, appendix A, eq. 12) with
    a stopping mask.

      r = pos_w * exp(-pos_err_scale * ||x* - x_root||^2)
        + vel_w * exp(-vel_err_scale * max(0, v* - d* . xdot_root)^2)

    planar (x, y). Paper values: pos_w 0.7, vel_w 0.3, pos_err_scale 0.5,
    vel_err_scale 1.0. The paper's xdot is the centre-of-mass velocity; the
    root's finite-difference velocity is used here (MimicKit's own location
    task does the same).

    THE MASK, a deliberate addition: eq. 12 read literally pays LESS for
    standing at the target than for walking past it -- standing still gives
    max(0, v* - 0) = v* and the velocity term drops to exp(-v*^2). Our task has
    to stop there (walk_to_stand), so inside stop_radius the velocity term is 1,
    as MimicKit's task_location_env does with its dist_threshold. Moving away
    from the target outside the radius gets velocity term 0.
    """
    diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    dist_sq = torch.sum(diff * diff, dim=-1)
    pos_r = torch.exp(-pos_err_scale * dist_sq)

    tar_dir = torch.nn.functional.normalize(diff, dim=-1)
    root_vel = (root_pos[..., 0:2] - prev_root_pos[..., 0:2]) / dt
    speed_to_tar = torch.sum(tar_dir * root_vel, dim=-1)
    vel_err = torch.clamp_min(tar_speed - speed_to_tar, 0.0)
    vel_r = torch.exp(-vel_err_scale * vel_err * vel_err)
    vel_r = torch.where(speed_to_tar <= 0.0, torch.zeros_like(vel_r), vel_r)

    inside = dist_sq < stop_radius * stop_radius
    vel_r = torch.where(inside, torch.ones_like(vel_r), vel_r)
    return pos_w * pos_r + vel_w * vel_r
