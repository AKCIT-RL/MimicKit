"""Cow dribble ("drible da vaca") building blocks (doc 40).

Episode scenario: a still (later rolling) ball near the robot and a virtual opponent ahead.
The robot first controls the ball (phase 0), then gives one discrete touch past the
opponent (phase 1 -> 2) and recovers the ball on the other side. The side is chosen by the
policy: every term below is side-agnostic.

The opponent travels in the existing 5-dim steering block of the task obs
([ox/scale, oy/scale, vis, ovx, ovy], heading frame), so the obs dimension and every
warm-start stay unchanged. Pure torch, batch-first, no simulator dependencies.
"""

import torch

PH_CTRL = 0     # control the ball (gate: near + slow for hold_s)
PH_TOUCH = 1    # gate reached, waiting for the discrete touch
PH_FLIGHT = 2   # touched: ball in flight past the opponent, recover it


def compute_cow_obs(opp_rel_h, opp_vel_h, vis, pos_scale):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """[N, 5] = [ox/scale, oy/scale, vis, ovx, ovy]; opponent estimate in the robot heading frame.
    vis [N]: 1 = detected this step, 0 = extrapolated (held last estimate)."""
    rel = torch.clamp(opp_rel_h / pos_scale, -2.0, 2.0)
    return torch.cat([rel, vis.unsqueeze(-1), opp_vel_h], dim=-1)


def ball_pass_clearance(ball, vel, opp):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """Closest approach [N] of the ball (straight line along its velocity, t >= 0) to the opponent.
    A (nearly) still ball is just its current distance."""
    sp2 = torch.sum(vel * vel, dim=-1)
    rel = opp - ball
    t = torch.clamp(torch.sum(rel * vel, dim=-1) / torch.clamp(sp2, min=1.0e-6), min=0.0)
    closest = ball + vel * t.unsqueeze(-1)
    c = torch.linalg.norm(closest - opp, dim=-1)
    return torch.where(sp2 < 1.0e-4, torch.linalg.norm(rel, dim=-1), c)


def compute_cow_touch_reward(ball, vel, opp, radius, margin, v_target):
    # type: (Tensor, Tensor, Tensor, float, float, float) -> Tensor
    """Side-agnostic pass reward in [0, 1]: ball speed (saturating at v_target) x lateral clearance past
    the opponent (0 when the straight path hits the opponent, 1 at radius + margin) x forward cosine.
    Straight into the opponent (clearance 0) and sideways (cos 0) pay nothing: the optimum is a
    ~20-50 degree deviation to either side."""
    speed = torch.linalg.norm(vel, dim=-1)
    to_opp = opp - ball
    u = to_opp / torch.clamp(torch.linalg.norm(to_opp, dim=-1, keepdim=True), min=1.0e-6)
    cos = torch.sum(vel * u, dim=-1) / torch.clamp(speed, min=1.0e-6)
    c = ball_pass_clearance(ball, vel, opp)
    clear = torch.clamp((c - radius) / max(margin, 1.0e-6), 0.0, 1.0)
    return torch.clamp(speed / max(v_target, 1.0e-6), 0.0, 1.0) * clear * torch.clamp(cos, 0.0, 1.0)


def compute_cow_goal_term(ball, vel, goal, clear_factor, v_target):
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    """Tie-breaker between sides: cosine between the ball velocity and the ball-to-goal direction,
    only while the pass is clean (clear_factor in [0, 1])."""
    speed = torch.linalg.norm(vel, dim=-1)
    to_goal = goal - ball
    u = to_goal / torch.clamp(torch.linalg.norm(to_goal, dim=-1, keepdim=True), min=1.0e-6)
    cos = torch.sum(vel * u, dim=-1) / torch.clamp(speed, min=1.0e-6)
    return torch.clamp(cos, 0.0, 1.0) * torch.clamp(speed / max(v_target, 1.0e-6), 0.0, 1.0) * clear_factor


def cow_side(axis, ball, opp):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """+1 / -1: side of the opponent the ball is on, w.r.t. the touch axis (cross product sign)."""
    rel = ball - opp
    cr = axis[:, 0] * rel[:, 1] - axis[:, 1] * rel[:, 0]
    return torch.where(cr >= 0.0, torch.ones_like(cr), -torch.ones_like(cr))


def cow_phase_step(phase, hold, since, min_bo, min_ro, axis, ball, vel, robot, opp, dt, p):
    """One tick of the phase machine. Returns (phase, hold, since, min_bo, min_ro, axis, ev) where ev is a
    dict of [N] bool tensors: gate, touch, pass_ok, succ, fail_ball, fail_contact, fail_lost, fail_time.
    Any success/fail event ends the scenario (the env respawns it); `p` maps parameter names to floats."""
    n = phase.shape[0]
    dist_rb = torch.linalg.norm(ball - robot, dim=-1)
    sp = torch.linalg.norm(vel, dim=-1)
    d_bo = torch.linalg.norm(ball - opp, dim=-1)
    d_ro = torch.linalg.norm(robot - opp, dim=-1)

    # phase 0: control gate
    in0 = phase == PH_CTRL
    gate_ok = torch.logical_and(dist_rb < p["ctrl_max"], sp < p["v_stab"])
    hold = torch.where(in0, torch.where(gate_ok, hold + dt, torch.zeros_like(hold)), hold)
    gate = torch.logical_and(in0, hold >= p["hold_s"])
    since = torch.where(in0, since + dt, since)
    phase = torch.where(gate, torch.full_like(phase, PH_TOUCH), phase)
    since = torch.where(gate, torch.zeros_like(since), since)

    # phase 1: discrete touch = the ball leaves faster than touch_speed
    in1 = torch.logical_and(phase == PH_TOUCH, ~gate)
    since = torch.where(in1, since + dt, since)
    touch = torch.logical_and(in1, sp > p["touch_speed"])
    to_opp = opp - ball
    new_axis = to_opp / torch.clamp(torch.linalg.norm(to_opp, dim=-1, keepdim=True), min=1.0e-6)
    axis = torch.where(touch.unsqueeze(-1), new_axis, axis)
    phase = torch.where(touch, torch.full_like(phase, PH_FLIGHT), phase)
    since = torch.where(touch, torch.zeros_like(since), since)
    min_bo = torch.where(touch, d_bo, min_bo)
    min_ro = torch.where(touch, d_ro, min_ro)

    # phase 2: flight and recovery
    in2 = torch.logical_and(phase == PH_FLIGHT, ~touch)
    since = torch.where(in2, since + dt, since)
    min_bo = torch.where(in2, torch.minimum(min_bo, d_bo), min_bo)
    min_ro = torch.where(in2, torch.minimum(min_ro, d_ro), min_ro)
    past = torch.sum((ball - opp) * axis, dim=-1) > p["past_margin"]
    clean = min_bo >= (p["radius"] + p["margin"])
    pass_ok = torch.logical_and(in2, torch.logical_and(past, clean))
    succ = torch.logical_and(pass_ok, torch.logical_and(
        torch.logical_and(dist_rb < p["rec_dist"], sp < p["rec_speed"]),
        min_ro >= (p["radius"] + p["robot_pad"])))

    fail_ball = d_bo < p["radius"]
    fail_contact = d_ro < (p["radius"] + p["robot_pad"])
    fail_lost = dist_rb > p["lost_dist"]
    t_lim = torch.where(phase == PH_CTRL, torch.full_like(since, p["ctrl_s"]),
                        torch.where(phase == PH_TOUCH, torch.full_like(since, p["wait_s"]),
                                    torch.full_like(since, p["rec_s"])))
    fail_time = since > t_lim
    # a success outranks the same-step failures that would only be geometric noise
    fail_ball = torch.logical_and(fail_ball, ~succ)
    fail_contact = torch.logical_and(fail_contact, ~succ)
    fail_lost = torch.logical_and(fail_lost, ~succ)
    fail_time = torch.logical_and(fail_time, ~succ)
    ev = {"gate": gate, "touch": touch, "pass_ok": pass_ok, "succ": succ,
          "fail_ball": fail_ball, "fail_contact": fail_contact, "fail_lost": fail_lost, "fail_time": fail_time}
    return phase, hold, since, min_bo, min_ro, axis, ev


def sample_cow_opponent(ball, heading_dir, goal, d_range, ang_max_rad, goal_blend, pad_xy_lo, pad_xy_hi):
    # type: (Tensor, Tensor, Tensor, List[float], float, float, Tensor, Tensor) -> Tensor
    """Opponent position [N, 2]: from the ball along unit((1-b)*heading + b*to_goal) rotated by a random
    angle in [-ang_max, ang_max], at distance U(d_range), clamped into [pad_xy_lo, pad_xy_hi]."""
    n = ball.shape[0]
    dev = ball.device
    to_goal = goal - ball
    g = to_goal / torch.clamp(torch.linalg.norm(to_goal, dim=-1, keepdim=True), min=1.0e-6)
    u = (1.0 - goal_blend) * heading_dir + goal_blend * g
    u = u / torch.clamp(torch.linalg.norm(u, dim=-1, keepdim=True), min=1.0e-6)
    ang = (2.0 * torch.rand([n], device=dev) - 1.0) * ang_max_rad
    c, s = torch.cos(ang), torch.sin(ang)
    ur = torch.stack([c * u[:, 0] - s * u[:, 1], s * u[:, 0] + c * u[:, 1]], dim=-1)
    d = d_range[0] + (d_range[1] - d_range[0]) * torch.rand([n], device=dev)
    opp = ball + ur * d.unsqueeze(-1)
    return torch.minimum(torch.maximum(opp, pad_xy_lo), pad_xy_hi)
