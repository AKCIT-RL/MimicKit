"""Soccer task building blocks (arXiv:2511.03996, Table 3).

Pure-torch, batch-first functions shared by the soccer envs. All positions are
world-frame unless noted; planar quantities use (x, y). The task rewards follow
the paper: potential-based shaping terms (potential = Euclidean planar
distance) for robot->ball and ball->goal, a terminal goal reward, and shaping
components for the arch/inside-foot kick. No simulator dependencies.
"""

import torch

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
