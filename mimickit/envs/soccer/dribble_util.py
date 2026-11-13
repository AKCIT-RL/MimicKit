"""Dribble skill building blocks (Dribble Master, arXiv:2505.12679; doc 34).

The command is the desired BALL velocity (vx, vy) in the robot heading frame.
It travels in the existing 5-dim steering block of the task obs, so the obs
dimension (and every warm-start) is unchanged. Pure-torch, batch-first, no
simulator dependencies.
"""

import torch


@torch.jit.script
def compute_dribble_cmd_obs(cmd):
    # type: (Tensor) -> Tensor
    """Ball-velocity command [N, 2] (heading frame) in the steering-slot layout
    [dir_x, dir_y, |v|, 1, 0]; heading direction (1, 0) when the command is zero:
    the all-zero block drops the policy (doc 34, A8), a tiny command does not."""
    speed = torch.linalg.norm(cmd, dim=-1, keepdim=True)
    heading = torch.zeros_like(cmd)
    heading[:, 0] = 1.0
    direction = torch.where(speed > 1.0e-4, cmd / torch.clamp(speed, min=1.0e-6), heading)
    ones = torch.ones_like(speed)
    return torch.cat([direction, speed, ones, torch.zeros_like(speed)], dim=-1)


def sample_dribble_cmd(n, speed_range, dir_max_rad, still_prob, device):
    """Uniform speed in speed_range and direction in [-dir_max, dir_max] around
    the heading x axis; still_prob of the draws are the zero command."""
    speed = speed_range[0] + (speed_range[1] - speed_range[0]) * torch.rand([n], device=device)
    ang = (2.0 * torch.rand([n], device=device) - 1.0) * dir_max_rad
    cmd = torch.stack([speed * torch.cos(ang), speed * torch.sin(ang)], dim=-1)
    still = torch.rand([n], device=device) < still_prob
    cmd[still] = 0.0
    return cmd


@torch.jit.script
def compute_dribble_vel_reward(ball_vel_h, cmd, sigma):
    # type: (Tensor, Tensor, float) -> Tensor
    """exp(-|v_cmd - v_ball|^2 / sigma) in [0, 1], planar heading-frame velocities."""
    err2 = torch.sum(torch.square(cmd - ball_vel_h), dim=-1)
    return torch.exp(-err2 / sigma)


@torch.jit.script
def compute_dribble_near_weight(dist, d_near, d_far):
    # type: (Tensor, float, float) -> Tensor
    """1 while the ball is within d_near of the robot, linear to 0 at d_far:
    blends the ball-velocity reward (near) against the chase reward (far)."""
    return 1.0 - torch.clamp((dist - d_near) / max(d_far - d_near, 1.0e-6), 0.0, 1.0)


@torch.jit.script
def compute_dribble_dist_penalty(dist, d_free, d_max):
    # type: (Tensor, float, float) -> Tensor
    """0 inside d_free, ramps to 1 at d_max (multiply by a negative weight)."""
    return torch.clamp((dist - d_free) / max(d_max - d_free, 1.0e-6), 0.0, 1.0)


@torch.jit.script
def compute_dribble_progress_reward(ball_vel_h, cmd, over_pen=0.0):
    # type: (Tensor, Tensor, float) -> Tensor
    speed = torch.linalg.norm(cmd, dim=-1)
    direction = cmd / torch.clamp(speed.unsqueeze(-1), min=1.0e-6)
    ratio = torch.sum(ball_vel_h * direction, dim=-1) / torch.clamp(speed, min=0.1)
    r = torch.clamp(ratio, -1.0, 1.0) - over_pen * torch.clamp(ratio - 1.0, 0.0, 1.0)
    return torch.where(speed > 0.05, r, torch.zeros_like(r))