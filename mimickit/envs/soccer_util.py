"""Soccer task building blocks (arXiv:2511.03996, Table 3).

Pure-torch, batch-first functions shared by the soccer envs. All positions are
world-frame unless noted; planar quantities use (x, y). The task rewards follow
the paper: potential-based shaping terms (potential = Euclidean planar
distance) for robot->ball and ball->goal, a terminal goal reward, and shaping
components for the arch/inside-foot kick. No simulator dependencies.
"""

import numpy as np
import torch
from typing import Tuple

import util.torch_util as torch_util


@torch.jit.script
def compute_soccer_observations(root_pos, root_rot, ball_pos, goal_pos, goal_dir):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    """Task observation: ball/goal in the robot's heading frame.

    ball_pos: [N, 3] world; goal_pos: [N, 2] goal-line center (x, y);
    goal_dir: [N, 2] unit normal of the goal pointing into the field.
    Returns [N, 6]: local ball (x, y), local goal center (x, y),
    local goal dir (cos, sin).
    """
    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)

    ball_rel = ball_pos - root_pos
    local_ball = torch_util.quat_rotate(heading_inv_rot, ball_rel)

    goal_rel3d = torch.cat([goal_pos - root_pos[..., 0:2],
                            torch.zeros_like(goal_pos[..., 0:1])], dim=-1)
    local_goal = torch_util.quat_rotate(heading_inv_rot, goal_rel3d)

    goal_dir3d = torch.cat([goal_dir, torch.zeros_like(goal_dir[..., 0:1])], dim=-1)
    local_goal_dir = torch_util.quat_rotate(heading_inv_rot, goal_dir3d)

    obs = torch.cat([local_ball[..., 0:2], local_goal[..., 0:2],
                     local_goal_dir[..., 0:2]], dim=-1)
    return obs


@torch.jit.script
def compute_ball_steer_command(root_pos, ball_pos, stop_dist, speed_max):
    # type: (Tensor, Tensor, float, float) -> Tensor
    """Auto steering command toward the ball (T1 kicking-env style).

    The command fills the steering-task obs slots of a policy warm-started
    from a steering checkpoint, so the pretrained velocity tracking drags the
    robot to the ball without relying on exploration. Speed ramps linearly
    with distance beyond ``stop_dist`` (command is zeroed near the ball so
    the kick rewards take over) and saturates at ``speed_max``.

    Returns [N, 3]: world-frame unit target dir (x, y) and target speed.
    """
    delta = ball_pos[..., 0:2] - root_pos[..., 0:2]
    dist = torch.norm(delta, dim=-1)
    tar_dir = delta / torch.clamp_min(dist, 1e-6).unsqueeze(-1)
    tar_speed = torch.clamp(dist - stop_dist, min=0.0, max=speed_max)
    return torch.cat([tar_dir, tar_speed.unsqueeze(-1)], dim=-1)


@torch.jit.script
def compute_kick_direction_reward(ball_pos, ball_vel, goal_pos, min_vel, decay_tau,
                                  ball_moving_time, max_reward):
    # type: (Tensor, Tensor, Tensor, float, float, Tensor, float) -> Tensor
    """Ball velocity toward the goal, above a minimum speed (T1 kicking env).

    Only the planar velocity component projected on the ball->goal direction
    counts, minus ``min_vel`` (accidental touches and slow dribbles pay
    nothing). The exponential decay over ``ball_moving_time`` concentrates
    the credit at the impact instead of paying while the ball coasts.
    """
    to_goal = goal_pos - ball_pos[..., 0:2]
    to_goal = to_goal / torch.clamp_min(torch.norm(to_goal, dim=-1, keepdim=True), 1e-6)
    v_dir = torch.sum(ball_vel[..., 0:2] * to_goal, dim=-1)
    above = torch.clamp_min(v_dir - min_vel, 0.0)
    decay = torch.exp(-ball_moving_time / decay_tau)
    return torch.clamp(above * decay, min=0.0, max=max_reward)


@torch.jit.script
def compute_ball_approach_reward(root_pos, prev_root_pos, ball_pos, prev_ball_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    """Potential-based robot->ball shaping: r = d_prev - d_curr (planar)."""
    curr_d = torch.linalg.norm(ball_pos[..., 0:2] - root_pos[..., 0:2], dim=-1)
    prev_d = torch.linalg.norm(prev_ball_pos[..., 0:2] - prev_root_pos[..., 0:2], dim=-1)
    return prev_d - curr_d


@torch.jit.script
def compute_goal_progress_reward(ball_pos, prev_ball_pos, goal_pos):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """Potential-based ball->goal shaping: r = d_prev - d_curr (planar)."""
    curr_d = torch.linalg.norm(goal_pos - ball_pos[..., 0:2], dim=-1)
    prev_d = torch.linalg.norm(goal_pos - prev_ball_pos[..., 0:2], dim=-1)
    return prev_d - curr_d


@torch.jit.script
def compute_goal_scored_flags(ball_pos, goal_pos, goal_dir, goal_width, ball_radius):
    # type: (Tensor, Tensor, Tensor, float, float) -> Tensor
    """Ball fully crossed the goal line within the goal mouth.

    goal_dir points into the field, so "behind the line" is a negative
    projection onto goal_dir beyond the ball radius.
    """
    rel = ball_pos[..., 0:2] - goal_pos
    depth = torch.sum(rel * goal_dir, dim=-1)
    lateral = rel[..., 0] * (-goal_dir[..., 1]) + rel[..., 1] * goal_dir[..., 0]

    crossed = depth < -ball_radius
    in_mouth = torch.abs(lateral) < 0.5 * goal_width
    return torch.logical_and(crossed, in_mouth)


@torch.jit.script
def compute_out_of_bounds_flags(ball_pos, field_length, field_width):
    # type: (Tensor, float, float) -> Tensor
    """Ball center left the field centered at the origin (length: x, width: y)."""
    out_x = torch.abs(ball_pos[..., 0]) > 0.5 * field_length
    out_y = torch.abs(ball_pos[..., 1]) > 0.5 * field_width
    return torch.logical_or(out_x, out_y)


@torch.jit.script
def compute_ball_out_flags(ball_pos, field_length, field_width, goal_pos, goal_dir,
                           goal_width, ball_radius):
    # type: (Tensor, float, float, Tensor, Tensor, float, float) -> Tensor
    """Out-of-bounds that does not fire inside the goal mouth corridor.

    The center-based OOB test trips on the goal line before the
    fully-across goal test (depth < -ball_radius) can fire. Exempt the thin
    corridor behind the goal mouth so a scoring ball is flagged as a goal,
    never as out.
    """
    oob = compute_out_of_bounds_flags(ball_pos, field_length, field_width)

    rel = ball_pos[..., 0:2] - goal_pos
    depth = torch.sum(rel * goal_dir, dim=-1)
    lateral = rel[..., 0] * (-goal_dir[..., 1]) + rel[..., 1] * goal_dir[..., 0]

    slack = 0.05
    in_corridor = torch.abs(lateral) < 0.5 * goal_width + ball_radius
    crossing_band = depth >= -(ball_radius + slack)
    exempt = torch.logical_and(in_corridor, crossing_band)

    return torch.logical_and(oob, torch.logical_not(exempt))


@torch.jit.script
def apply_ball_event_dones(done, goal_scored, ball_oob, null_val, succ_val, fail_val):
    # type: (Tensor, Tensor, Tensor, int, int, int) -> Tuple[Tensor, Tensor]
    """Paper 4.1: goal and ball-out terminate the episode (bootstrap cut).

    Dones already decided by the caller (fall, timeout) take precedence;
    only NULL envs are updated. Returns (done, soft_mask); soft envs keep
    the robot state on reset and only the ball is repositioned.
    """
    undecided = done == null_val
    goal_done = torch.logical_and(goal_scored, undecided)
    oob_done = torch.logical_and(ball_oob, undecided)
    done = done.clone()
    done[goal_done] = succ_val
    done[oob_done] = fail_val
    soft = torch.logical_or(goal_done, oob_done)
    return done, soft


@torch.jit.script
def compute_stagnation_flags(root_pos, window_root_pos, move_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    """Robot displaced less than move_threshold (planar) since window_root_pos
    (the root position recorded ~1 s ago)."""
    disp = torch.linalg.norm(root_pos[..., 0:2] - window_root_pos[..., 0:2], dim=-1)
    return disp < move_threshold


@torch.jit.script
def compute_kick_components(root_rot, foot_vel, ball_contact):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """Sideways/forward foot-velocity magnitudes while touching the ball.

    foot_vel: [N, 3] world velocity of the kicking foot; ball_contact: [N]
    bool/float mask. Returns [N, 2]: (sideways speed |v_y|, forward speed
    max(v_x, 0)) in the heading frame, zeroed without contact. The env weighs
    these (+w sideways, -w forward) to shape the arch/inside-foot kick.
    """
    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
    local_vel = torch_util.quat_rotate(heading_inv_rot, foot_vel)

    sideways = torch.abs(local_vel[..., 1])
    forward = torch.clamp_min(local_vel[..., 0], 0.0)

    contact = ball_contact.type_as(sideways)
    components = torch.stack([sideways * contact, forward * contact], dim=-1)
    return components


@torch.jit.script
def compute_foot_proximity_penalty(left_foot_pos, right_foot_pos, min_dist):
    # type: (Tensor, Tensor, float) -> Tensor
    """Positive magnitude when the feet are closer than min_dist (planar)."""
    d = torch.linalg.norm(left_foot_pos[..., 0:2] - right_foot_pos[..., 0:2], dim=-1)
    return torch.clamp_min(min_dist - d, 0.0)


@torch.jit.script
def compute_action_rate_penalty(action, prev_action):
    # type: (Tensor, Tensor) -> Tensor
    """Squared action change per step: ||a_t - a_{t-1}||^2."""
    diff = action - prev_action
    return torch.sum(torch.square(diff), dim=-1)


@torch.jit.script
def compute_joint_limit_penalty(dof_pos, dof_low, dof_high):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """Total excursion beyond the joint limits (0 when inside)."""
    below = torch.clamp_min(dof_low - dof_pos, 0.0)
    above = torch.clamp_min(dof_pos - dof_high, 0.0)
    return torch.sum(below + above, dim=-1)


@torch.jit.script
def compute_ball_contact_flags(foot_pos, ball_pos, contact_dist):
    # type: (Tensor, Tensor, float) -> Tensor
    """Foot within contact_dist of the ball center (3D)."""
    d = torch.linalg.norm(foot_pos - ball_pos, dim=-1)
    return d < contact_dist


@torch.jit.script
def compute_base_accel_penalty(root_vel, prev_root_vel, dt):
    # type: (Tensor, Tensor, float) -> Tensor
    """Squared base acceleration: ||(v_t - v_{t-1}) / dt||^2."""
    accel = (root_vel - prev_root_vel) / dt
    return torch.sum(torch.square(accel), dim=-1)


def build_field_line_segments(field_length, field_width, goal_width,
                              goal_area_length=1.0, goal_area_width=4.0,
                              penalty_area_length=3.0, penalty_area_width=6.0,
                              penalty_mark_dist=2.1, center_circle_radius=1.5,
                              line_z=0.02, goal_post_height=1.25,
                              circle_segments=24):
    """Viewer-only field markings (RoboCup AdultSize layout by default).

    Numpy, field-local frame (field center at the origin, active goal on the
    +x edge). Returns (starts [S, 3], ends [S, 3], colors [S, 4]) float32 for
    ``engine.draw_lines``. Purely cosmetic: no physics, obs or reward use.
    """
    import numpy as np

    hl = 0.5 * field_length
    hw = 0.5 * field_width
    z = line_z
    white = (1.0, 1.0, 1.0, 1.0)
    green = (0.1, 0.9, 0.2, 1.0)

    segs = []  # (x0, y0, z0, x1, y1, z1, color)

    def add(p0, p1, col=white):
        segs.append((p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], col))

    # touch lines + goal lines (perimeter)
    add((-hl, -hw, z), (hl, -hw, z))
    add((-hl, hw, z), (hl, hw, z))
    add((-hl, -hw, z), (-hl, hw, z))
    add((hl, -hw, z), (hl, hw, z))

    # halfway line + center circle
    add((0.0, -hw, z), (0.0, hw, z))
    ang = np.linspace(0.0, 2.0 * np.pi, circle_segments + 1)
    cx = center_circle_radius * np.cos(ang)
    cy = center_circle_radius * np.sin(ang)
    for i in range(circle_segments):
        add((cx[i], cy[i], z), (cx[i + 1], cy[i + 1], z))

    # goal/penalty areas + penalty mark, both ends (sign = goal-line side)
    for sign in (1.0, -1.0):
        for depth, width in ((goal_area_length, goal_area_width),
                             (penalty_area_length, penalty_area_width)):
            xg = sign * hl                # goal line
            xf = sign * (hl - depth)      # front edge of the area
            hw_a = 0.5 * width
            add((xg, -hw_a, z), (xf, -hw_a, z))
            add((xg, hw_a, z), (xf, hw_a, z))
            add((xf, -hw_a, z), (xf, hw_a, z))
        xm = sign * (hl - penalty_mark_dist)
        add((xm - 0.1, 0.0, z), (xm + 0.1, 0.0, z))
        add((xm, -0.1, z), (xm, 0.1, z))

    # active goal mouth on +x: highlighted line + two vertical posts
    hg = 0.5 * goal_width
    add((hl, -hg, z), (hl, hg, z), green)
    add((hl, -hg, z), (hl, -hg, goal_post_height), green)
    add((hl, hg, z), (hl, hg, goal_post_height), green)
    add((hl, -hg, goal_post_height), (hl, hg, goal_post_height), green)

    arr = np.array([s[:6] for s in segs], dtype=np.float32)
    cols = np.array([s[6] for s in segs], dtype=np.float32)
    return arr[:, 0:3].copy(), arr[:, 3:6].copy(), cols


def compute_field_offsets(num_envs, field_length, field_width, field_sep):
    """Per-env field centers on a centered grid (world frame).

    Single source of truth for the field layout: the env places fields with
    these offsets and the engine sizes the uneven-ground mesh from the same
    grid (see ``compute_field_grid_extent``). Returns float32 [N, 2].
    """
    n_cols = int(np.ceil(np.sqrt(num_envs)))
    n_rows = int(np.ceil(num_envs / n_cols))
    pitch_x = field_length + 2.0 * field_sep
    pitch_y = field_width + 2.0 * field_sep
    idx = np.arange(num_envs)
    col = (idx % n_cols).astype(np.float32)
    row = (idx // n_cols).astype(np.float32)
    offsets = np.zeros([num_envs, 2], dtype=np.float32)
    offsets[:, 0] = (col - 0.5 * (n_cols - 1)) * pitch_x
    offsets[:, 1] = (row - 0.5 * (n_rows - 1)) * pitch_y
    return offsets


def compute_field_grid_extent(num_envs, field_length, field_width, field_sep):
    """Total (size_x, size_y) in meters of the field grid, centered at the
    origin. Covers every field of ``compute_field_offsets`` including the
    separation strip around each one."""
    n_cols = int(np.ceil(np.sqrt(num_envs)))
    n_rows = int(np.ceil(num_envs / n_cols))
    size_x = n_cols * (field_length + 2.0 * field_sep)
    size_y = n_rows * (field_width + 2.0 * field_sep)
    return size_x, size_y


def compute_anneal_scale(samples, start_samples, end_samples):
    """Linear 1 -> 0 anneal factor over a sample budget.

    start_samples < 0 disables the anneal (always 1). end_samples <=
    start_samples makes the schedule a step: 1 before start, 0 at/after it
    (used by eval configs to zero the steering crutch outright).
    """
    if (start_samples < 0):
        return 1.0
    if (samples < start_samples):
        return 1.0
    if (end_samples <= start_samples):
        return 0.0
    frac = (samples - start_samples) / float(end_samples - start_samples)
    return float(np.clip(1.0 - frac, 0.0, 1.0))


@torch.jit.script
def compute_perception_noise_std(dist, dist_coef: float, base_std: float):
    # type: (Tensor, float, float) -> Tensor
    """Ball-position noise std as a function of distance (paper section 9):
    sigma = dist_coef * d + base_std (paper: 0.124 * d + 0.149)."""
    return dist_coef * dist + base_std


@torch.jit.script
def compute_ball_detection_prob(dist, in_fov, base_prob: float,
                                full_range: float, decay_range: float):
    # type: (Tensor, Tensor, float, float, float) -> Tensor
    """Detection probability of the ball (paper section 9).

    base_prob inside the FOV up to full_range meters, decaying linearly to 0
    over the next decay_range meters; 0 outside the FOV.
    dist: [N] planar robot->ball distance; in_fov: [N] bool.
    """
    decay = 1.0 - (dist - full_range) / decay_range
    prob = base_prob * torch.clamp(decay, min=0.0, max=1.0)
    prob = torch.where(dist <= full_range,
                       torch.full_like(prob, base_prob), prob)
    prob = prob * in_fov.float()
    return prob


@torch.jit.script
def compute_ball_in_fov(root_pos, root_rot, ball_pos, fov_half_rad: float):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """Whether the ball bearing is within +-fov_half_rad of the robot heading.

    fov_half_rad <= 0 disables the check (always True). Uses the heading
    (yaw-only) frame; returns [N] bool.
    """
    if (fov_half_rad <= 0.0):
        return torch.ones_like(root_pos[..., 0], dtype=torch.bool)
    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
    ball_rel = ball_pos - root_pos
    local_ball = torch_util.quat_rotate(heading_inv_rot, ball_rel)
    bearing = torch.atan2(local_ball[..., 1], local_ball[..., 0])
    return torch.abs(bearing) <= fov_half_rad
