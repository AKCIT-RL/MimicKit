"""Cow teacher v2 (doc 40): pure helpers of the staged curriculum (capture, rest prediction, touch
target, region behind the opponent). Planar [N, 2] tensors; heading-frame vectors have x forward."""
import math

import torch

PH_SEEK = 0  # stage A: reach and settle the ball
PH_HOLD = 1  # stage A: keep it close after the capture
PH_CTRL = 0  # stage B+: reach the ball and touch it (same id as PH_SEEK: the scenario's opening phase)
PH_TOUCH = 2  # stage B+: foot on the ball until it is > touch_end_dist from the foot
PH_ROLL = 3  # stage B+: ball rolling toward its rest point


def sample_ring(n, r_lo, r_hi, ang_max, device):
    """[n, 2] heading-frame points at radius U[r_lo, r_hi], angle U[-ang_max, ang_max] (rad)."""
    r = r_lo + (r_hi - r_lo) * torch.rand([n], device=device)
    a = (2.0 * torch.rand([n], device=device) - 1.0) * ang_max
    return torch.stack([r * torch.cos(a), r * torch.sin(a)], dim=-1)


def heading_to_world(vec, heading_dir):
    """Rotate heading-frame planar vectors by the heading unit vector (cos, sin)."""
    c, s = heading_dir[:, 0], heading_dir[:, 1]
    return torch.stack([c * vec[:, 0] - s * vec[:, 1], s * vec[:, 0] + c * vec[:, 1]], dim=-1)


def world_to_heading(vec, heading_dir):
    c, s = heading_dir[:, 0], heading_dir[:, 1]
    return torch.stack([c * vec[:, 0] + s * vec[:, 1], -s * vec[:, 0] + c * vec[:, 1]], dim=-1)


def sample_ball_vel(n, s_lo, s_hi, device):
    """[n, 2] world velocity, speed U[s_lo, s_hi], direction uniform on the circle."""
    sp = s_lo + (s_hi - s_lo) * torch.rand([n], device=device)
    a = 2.0 * math.pi * torch.rand([n], device=device)
    return torch.stack([sp * torch.cos(a), sp * torch.sin(a)], dim=-1)


def predict_rest(pos, vel, decel):
    """Rest point of a ball under constant deceleration: p + v |v| / (2 a). decel <= 0: the position."""
    if decel <= 0.0:
        return pos.clone()
    sp = torch.linalg.norm(vel, dim=-1, keepdim=True)
    return pos + vel * sp / (2.0 * decel)


def capture_step(hold, dist, speed, dt, cap_dist, cap_speed):
    """Time the ball has been close (< cap_dist) and slow (< cap_speed) without a break."""
    ok = (dist < cap_dist) & (speed < cap_speed)
    return torch.where(ok, hold + dt, torch.zeros_like(hold))


def touch_target_reward(rest, target, sigma):
    """exp(-(d / sigma)^2) of the predicted rest point to the target: >= 0, 'close is good enough'."""
    d = torch.linalg.norm(rest - target, dim=-1)
    return torch.exp(-torch.square(d / max(sigma, 1.0e-6)))


def behind_region_dist(pt, opp, axis, back_lo, back_hi, lat_max):
    """Distance from pt to the region behind the opponent (D6): along the attack axis between back_lo and
    back_hi past the opponent, laterally within +-lat_max (straight behind or diagonal behind)."""
    rel = pt - opp
    perp = torch.stack([-axis[:, 1], axis[:, 0]], dim=-1)
    along = (rel * axis).sum(-1)
    lat = (rel * perp).sum(-1)
    da = along - torch.clamp(along, back_lo, back_hi)
    dl = lat - torch.clamp(lat, -lat_max, lat_max)
    return torch.sqrt(da * da + dl * dl)


def sample_behind_target(opp, axis, back_lo, back_hi, lat_max):
    """A point of the region behind the opponent (P for the actor in stage D)."""
    n = opp.shape[0]
    perp = torch.stack([-axis[:, 1], axis[:, 0]], dim=-1)
    a = back_lo + (back_hi - back_lo) * torch.rand([n], device=opp.device)
    l = (2.0 * torch.rand([n], device=opp.device) - 1.0) * lat_max
    return opp + a.unsqueeze(-1) * axis + l.unsqueeze(-1) * perp


def sample_target(ball, heading_dir, fwd, lat, ring_ang):
    """P around the ball, heading frame: ring_ang > 0 -> radius U[fwd] within +-ring_ang (rad); else a box
    x U[fwd], y U[-lat, lat]. Returns (world P [n, 2], lateral sign of P in the heading frame)."""
    n = ball.shape[0]
    if ring_ang > 0.0:
        rel = sample_ring(n, fwd[0], fwd[1], ring_ang, ball.device)
    else:
        x = fwd[0] + (fwd[1] - fwd[0]) * torch.rand([n], device=ball.device)
        y = (2.0 * torch.rand([n], device=ball.device) - 1.0) * lat
        rel = torch.stack([x, y], dim=-1)
    return ball + heading_to_world(rel, heading_dir), torch.sign(rel[:, 1])


def target_cmd(ball, target, heading_dir, scale, decel=0.0, v_arr=0.0):
    """P as a dribble command (heading frame): direction ball->P, magnitude |P - ball| / scale; with decel > 0
    the magnitude is the launch speed that rests the ball at P, sqrt(2 decel |P - ball|) (the dribble command
    is a ball velocity, doc 40 v2-28). With v_arr > 0 the ball is to PASS the target at v_arr (it keeps rolling
    and decelerates): sqrt(v_arr^2 + 2 decel |P - ball|)."""
    rel = world_to_heading(target - ball, heading_dir)
    if decel <= 0.0:
        return rel / scale
    d = torch.linalg.norm(rel, dim=-1, keepdim=True)
    return rel / torch.clamp(d, min=1.0e-6) * torch.sqrt(v_arr * v_arr + 2.0 * decel * d)


def bypass_target(ball, opp, target, clear):
    """Stage D teacher waypoint: while the straight ball->target segment passes within `clear` of the opponent
    (and the opponent lies between them), aim at the point `clear` beside the opponent on the target's side;
    otherwise the target itself."""
    seg = target - ball
    L = torch.clamp(torch.linalg.norm(seg, dim=-1, keepdim=True), min=1.0e-6)
    u = seg / L
    w = opp - ball
    t = (w * u).sum(-1, keepdim=True)
    perp = w - t * u
    blocked = (t > 0.0) & (t < L) & (torch.linalg.norm(perp, dim=-1, keepdim=True) < clear)
    n = torch.stack([-u[:, 1], u[:, 0]], dim=-1)  # left normal of ball->target
    off = (perp * n).sum(-1, keepdim=True)  # opponent's lateral offset: pass on the other side
    side = torch.where(off > 0.0, -torch.ones_like(off), torch.ones_like(off))
    wp = opp + side * n * clear
    return torch.where(blocked, wp, target)


def touch_start(prev_fd, fd, prev_speed, speed, contact_dist, min_gain, min_speed):
    """A touch: with a foot within contact_dist, the ball speed crossed min_speed upward (a push spread over
    several steps counts once), or jumped by >= min_gain in one step while already above min_speed."""
    near = torch.minimum(prev_fd, fd) < contact_dist
    cross = (prev_speed < min_speed) & (speed >= min_speed)
    jump = ((speed - prev_speed) >= min_gain) & (speed >= min_speed)
    return near & (cross | jump)


def approach_dir(root, ball, heading_dir):
    """Unit robot->ball direction (the frame P is drawn in for B1); the heading when the ball is < 0.1 m away."""
    d = ball - root
    n = torch.linalg.norm(d, dim=-1, keepdim=True)
    return torch.where(n > 0.1, d / torch.clamp(n, min=1.0e-6), heading_dir)


def sample_region(n, opp_fwd, opp_lat, reg_dist, reg_ang, device):
    """Stage D (doc 40 sec 4b), heading frame: opponent at x U[opp_fwd], y U[-opp_lat, opp_lat]; region centre
    at U[reg_dist] from the opponent within +-reg_ang (rad) of straight ahead (beside or behind it).
    Returns (opp_rel [n, 2], region_rel [n, 2])."""
    ox = opp_fwd[0] + (opp_fwd[1] - opp_fwd[0]) * torch.rand([n], device=device)
    oy = (2.0 * torch.rand([n], device=device) - 1.0) * opp_lat
    opp = torch.stack([ox, oy], dim=-1)
    return opp, opp + sample_ring(n, reg_dist[0], reg_dist[1], reg_ang, device)
