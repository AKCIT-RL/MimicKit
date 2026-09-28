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
