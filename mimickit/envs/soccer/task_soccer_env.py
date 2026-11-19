import os
import numpy as np
import torch

import gymnasium.spaces as spaces
import engines.engine as engine
import envs.base_env as base_env
import envs.diag_util as diag_util
import envs.smp_env as smp_env
import envs.soccer.cow2_util as cow2_util
import envs.soccer.cow_util as cow_util
import envs.soccer.dribble_util as dribble_util
import envs.soccer.percep_emulator as percep_emulator
import envs.soccer.soccer_util as soccer_util
import envs.steering.steering_disc as steering_disc
import envs.steering.steering_dr as steering_dr
import envs.steering.steering_reward as steering_reward
import envs.steering.steering_util as steering_util
import envs.steering.task_steering_env as task_steering_env
import util.terrain_util as terrain_util
import util.torch_util as torch_util


class TaskSoccerEnv(smp_env.SMPEnv):
    """Soccer task env (arXiv:2511.03996): one robot, one ball, one goal.

    Each env owns its own field region of the shared world frame, laid out on
    a spatial grid (see ``_field_offset``) so envs never overlap in the
    broadphase. The goal is a virtual line segment on the +x edge of each
    field; there are no physical goal posts.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._field_length = float(env_config.get("field_length", 14.0))
        self._field_width = float(env_config.get("field_width", 9.0))
        self._goal_width = float(env_config.get("goal_width", 2.6))
        self._ball_radius = float(env_config.get("ball_radius", 0.11))
        # gap between neighboring fields; every env gets its own field region
        # of the world so thousands of robots+balls never share one 14x9 box
        # (a single shared field blows up the PhysX broadphase pair count)
        self._field_sep = float(env_config.get("field_sep", 2.0))
        # keep spawns away from the field border so the ball does not
        # immediately go out of bounds
        self._spawn_margin = float(env_config.get("spawn_margin", 1.0))
        # ball spawns may reach closer to the border than the robot's (the Fig. 4A grid has balls up to 0.5 m from the goal line);
        # None keeps the old behaviour (same margin as the robot)
        _bsm = env_config.get("ball_spawn_margin", None)
        self._ball_spawn_margin = self._spawn_margin if _bsm is None else float(_bsm)
        # k64b: share of ball spawns drawn in the strip next to the goal line (field-local x in [strip_min, half_x]), any y; 0 = off
        self._ball_goal_strip_prob = float(env_config.get("ball_goal_strip_prob", 0.0))
        self._ball_goal_strip_min = float(env_config.get("ball_goal_strip_min", 4.8))
        # k64c: half-width in y of that strip (<= 0: the whole field width, as k64b)
        self._ball_goal_strip_y = float(env_config.get("ball_goal_strip_y", 0.0))
        # k64d: x = half_x - (half_x - strip_min) * u**pow; pow > 1 concentrates the strip next to the goal line (1 = uniform)
        self._ball_goal_strip_pow = float(env_config.get("ball_goal_strip_pow", 1.0))
        # k64e: second, narrow strip (the "core"): share of spawns with x in [core_min, half_x], |y| <= core_y; 0 = off
        self._ball_goal_core_prob = float(env_config.get("ball_goal_core_prob", 0.0))
        self._ball_goal_core_min = float(env_config.get("ball_goal_core_min", 5.6))
        self._ball_goal_core_y = float(env_config.get("ball_goal_core_y", 1.5))
        self._ball_spawn_min_dist = float(env_config.get("ball_spawn_min_dist", 0.5))

        # auto steering command toward the ball (T1 kicking-env style); fills
        # the steering-task obs slots preserved by the warm-start transplant
        self._steer_speed_max = float(env_config.get("steer_speed_max", 1.5))
        self._steer_stop_dist = float(env_config.get("steer_stop_dist", 0.45))

        # velocity-command (navigation) mode (doc 26, MK1): a fraction of the
        # episodes follow a (vx, vy, wz) command written into the steering
        # slots instead of playing the ball; the ball is parked far away and
        # hidden (zero slots, mask 0). Off by default (nav_cmd_prob 0).
        self._nav_cmd_prob = float(env_config.get("nav_cmd_prob", 0.0))
        self._nav_vx_range = [float(v) for v in env_config.get("nav_vx_range", [-0.6, 2.0])]
        self._nav_vy_range = [float(v) for v in env_config.get("nav_vy_range", [-0.5, 0.5])]
        self._nav_yaw_range = [float(v) for v in env_config.get("nav_yaw_range", [-1.0, 1.0])]
        self._nav_still_prob = float(env_config.get("nav_still_prob", 0.1))
        self._nav_resample_s = [float(v) for v in env_config.get("nav_resample_s", [2.0, 5.0])]
        self._nav_face_horizon_s = float(env_config.get("nav_face_horizon_s", 1.0))
        self._nav_switch_prob = float(env_config.get("nav_switch_prob", 0.0))
        self._kick_to_nav_rate = float(env_config.get("kick_to_nav_rate", 0.0))
        self._kick_to_nav_dist = float(env_config.get("kick_to_nav_dist", 0.0))
        # event-triggered kick -> nav: after a real kick (foot near ball, ball fast) with prob
        # k2n_post_prob after a delay in k2n_delay_s; abandoning the chase with prob
        # k2n_abandon_prob at a time in k2n_abandon_s after the kick episode starts
        self._k2n_post_prob = float(env_config.get("k2n_post_prob", 0.0))
        self._k2n_delay_s = env_config.get("k2n_delay_s", [0.0, 1.0])
        self._k2n_abandon_prob = float(env_config.get("k2n_abandon_prob", 0.0))
        self._k2n_abandon_s = env_config.get("k2n_abandon_s", [1.0, 6.0])
        self._k2n_kick_dist = float(env_config.get("k2n_kick_dist", 0.3))
        self._k2n_kick_speed = float(env_config.get("k2n_kick_speed", 1.0))
        self._nav_cmd_scale = float(env_config.get("nav_cmd_scale", 1.0))
        # AMP style gain in nav episodes (1.0 = unchanged): the mocaps have no backward/lateral gait, so the
        # style term can cancel the vy / negative-vx tracking reward.
        # fraction of nav commands drawn as a single backward or lateral axis (others 0): pure back/lateral
        # commands are rare in the uniform draw and the policy stays standing for them (k94k/k94l probes).
        self._nav_axis_prob = float(env_config.get("nav_axis_prob", 0.0))
        # axis commands also forward and pure yaw (4 axes instead of back/lateral): fast forward and in-place
        # turns are as rare in the uniform draw (u2/u3 students: fast_2.0 0.31, turn_right 0.21 vs nave5 0.10).
        self._nav_axis_all = bool(env_config.get("nav_axis_all", False))
        self._nav_style_gain = float(env_config.get("nav_style_gain", 1.0))
        self._reward_nav_lin_w = float(env_config.get("reward_nav_lin_w", 0.0))
        self._reward_nav_yaw_w = float(env_config.get("reward_nav_yaw_w", 0.0))
        self._nav_lin_sigma = float(env_config.get("nav_lin_sigma", 0.25))
        self._nav_yaw_sigma = float(env_config.get("nav_yaw_sigma", 0.25))
        # low-pass (EMA, time constant in s) of the tracked velocity; 0 = raw.
        # With action noise the raw yaw rate is mostly jitter (train err ~1.3
        # vs ~0.3 deterministic), which flattens the yaw reward (k94a-c)
        self._nav_vel_filter_s = float(env_config.get("nav_vel_filter_s", 0.0))
        # stop-command penalty (Olives, steering env 7e525b5; doc 43): on a zero nav command,
        # ||v_base_xy|| + still_foot_w * mean ||v_foot_xy|| (raw, unfiltered), after still_settle_s
        # seconds since the command was drawn. L1 on purpose: exp(-e^2/sigma) is flat near 0. 0 = off.
        self._reward_still_w = float(env_config.get("reward_still_w", 0.0))
        self._still_foot_w = float(env_config.get("still_foot_w", 0.5))
        self._still_settle_s = float(env_config.get("still_settle_s", 0.0))
        # evaluation hooks: None = sample; True/False = force nav/kick on
        # every (re)draw; _nav_fixed_cmd [3] replaces the sampled command
        self._nav_force = None
        self._nav_fixed_cmd = None

        # dribble mode (doc 34): third episode mode, command = desired ball
        # velocity in the heading frame, ball stays visible. All off by default.
        self._drib_cmd_prob = float(env_config.get("drib_cmd_prob", 0.0))
        self._drib_speed_range = [float(v) for v in env_config.get("drib_speed_range", [0.2, 1.0])]
        self._drib_dir_max_deg = float(env_config.get("drib_dir_max_deg", 45.0))
        self._drib_still_prob = float(env_config.get("drib_still_prob", 0.1))
        self._drib_resample_s = [float(v) for v in env_config.get("drib_resample_s", [3.0, 5.0])]
        self._drib_far_prob = float(env_config.get("drib_far_prob", 0.0))
        self._drib_style_gain = float(env_config.get("drib_style_gain", 1.0))
        self._reward_drib_vel_w = float(env_config.get("reward_drib_vel_w", 0.0))
        self._drib_vel_sigma = float(env_config.get("drib_vel_sigma", 0.1))
        self._reward_drib_prog_w = float(env_config.get("reward_drib_prog_w", 0.0))
        self._drib_prog_over_pen = float(env_config.get("drib_prog_over_pen", 0.0))
        self._reward_drib_chase_w = float(env_config.get("reward_drib_chase_w", 0.0))
        self._reward_drib_dist_w = float(env_config.get("reward_drib_dist_w", 0.0))
        self._drib_near_dist = float(env_config.get("drib_near_dist", 0.8))
        self._drib_far_dist = float(env_config.get("drib_far_dist", 2.0))
        self._drib_free_dist = float(env_config.get("drib_free_dist", 1.5))
        self._drib_max_dist = float(env_config.get("drib_max_dist", 3.0))
        # doc 36 (goals G): the command is a WORLD direction toward a field waypoint (what the BT sends), converted to the
        # heading frame every step; the robot must turn to follow it. Defaults keep the old behavior exactly.
        self._drib_world_dir = bool(env_config.get("drib_world_dir", False))
        self._drib_world_clip_deg = float(env_config.get("drib_world_dir_max_deg", 180.0))
        self._drib_wp_margin = float(env_config.get("drib_wp_margin", 1.5))
        self._reward_drib_turn_w = float(env_config.get("reward_drib_turn_w", 0.0))
        self._drib_turn_sigma = float(env_config.get("drib_turn_sigma", 0.0))
        self._reward_drib_yaw_w = float(env_config.get("reward_drib_yaw_w", 0.0))
        self._drib_oob_enable = bool(env_config.get("drib_oob_enable", False))
        # evaluation hooks: None = sample; True/False = force dribble on/off; fixed [2] command
        self._drib_force = None
        self._drib_fixed_cmd = None

        # cow dribble (doc 40): subset of the dribble episodes with a virtual opponent. All off by default.
        g = env_config.get
        self._cow_prob = float(g("cow_prob", 0.0))
        self._cow_radius = float(g("cow_radius", 0.40))
        self._cow_margin = float(g("cow_margin", 0.25))
        self._cow_robot_pad = float(g("cow_robot_pad", 0.25))
        self._cow_d_range = [float(v) for v in g("cow_d_range", [1.6, 2.6])]
        self._cow_ang_deg = float(g("cow_ang_deg", 0.0))
        self._cow_goal_blend = float(g("cow_goal_blend", 0.0))
        self._cow_opp_speed = float(g("cow_opp_speed", 0.0))
        self._cow_ball_decel = float(g("cow_ball_decel", 0.0))  # m/s^2 rolling resistance on the ball, cow envs only (0 = off, doc 40 (46))
        self._cow_phase_p = {
            "ctrl_max": float(g("cow_ctrl_max", 0.7)), "v_stab": float(g("cow_v_stab", 0.4)),
            "hold_s": float(g("cow_hold_s", 0.5)), "touch_speed": float(g("cow_touch_speed", 0.8)),
            "past_margin": float(g("cow_past_margin", 0.5)), "rec_dist": float(g("cow_rec_dist", 1.0)),
            "rec_speed": float(g("cow_rec_speed", 1.0)), "lost_dist": float(g("cow_lost_dist", 3.0)),
            "ctrl_s": float(g("cow_ctrl_s", 8.0)), "wait_s": float(g("cow_wait_s", 3.0)),
            "rec_s": float(g("cow_rec_s", 6.0)),
            "radius": self._cow_radius, "margin": self._cow_margin, "robot_pad": self._cow_robot_pad}
        self._reward_cow_ctrl_w = float(g("reward_cow_ctrl_w", 0.0))
        self._reward_cow_chase_w = float(g("reward_cow_chase_w", 0.0))
        self._cow_ctrl2_past = float(g("cow_ctrl2_past", 0.0))  # 1: ctrl2 only once the ball is beyond the opponent
        self._reward_cow_ctrl2_w = float(g("reward_cow_ctrl2_w", 0.0))  # keep/regain control of a slow ball after the touch (phase 2)
        self._reward_cow_chase2_w = float(g("reward_cow_chase2_w", 0.0))  # chase the ball after the touch (phase 2)
        self._reward_cow_touch_w = float(g("reward_cow_touch_w", 0.0))
        self._reward_cow_idle_w = float(g("reward_cow_idle_w", 0.0))
        self._reward_cow_rec_w = float(g("reward_cow_rec_w", 0.0))
        self._reward_cow_goal_w = float(g("reward_cow_goal_w", 0.0))
        self._reward_cow_succ_w = float(g("reward_cow_succ_w", 0.0))
        self._reward_cow_fail_w = float(g("reward_cow_fail_w", 0.0))
        self._reward_cow_fall_w = float(g("reward_cow_fall_w", 0.0))  # penalty on the step the root drops below 0.45 m
        self._cow_dstar = float(g("cow_dstar", 0.4))
        self._cow_sigma_d = float(g("cow_sigma_d", 0.05))
        self._cow_sigma_v = float(g("cow_sigma_v", 0.1))
        self._cow_v_target = float(g("cow_v_target", 2.0))
        # >0: the touch reward is paid only in the first cow_touch_window_s of the flight (event-like); 0 = whole flight
        self._cow_touch_window_s = float(g("cow_touch_window_s", 0.0))
        self._cow_rec_sigma = float(g("cow_rec_sigma", 1.0))
        self._cow_obs_scale = float(g("cow_obs_scale", 4.0))
        self._cow_obs_noise_pos = float(g("cow_obs_noise_pos", 0.0))
        self._cow_obs_noise_vel = float(g("cow_obs_noise_vel", 0.0))
        self._cow_vis_dropout_p = float(g("cow_vis_dropout_p", 0.0))
        # evaluation hook: None = sample; True/False = force cow on/off on every (re)draw
        self._cow_force = None
        # cow teacher v2 (doc 40, staged curriculum): 0 = v1 behaviour; 1 = stage A (capture a still/rolling ball)
        self._cow2_stage = int(g("cow2_stage", 0))
        # stage 2+: fraction of respawns that run a stage-A scenario instead (keeps A while B trains; doc 40 v2-58)
        self._cow2_mix_a = float(g("cow2_mix_a", 0.0))
        self._cow2_ball_dist = [float(v) for v in g("cow2_ball_dist", [0.3, 1.5])]
        self._cow2_ball_ang_deg = float(g("cow2_ball_ang_deg", 180.0))
        self._cow2_ball_speed = [float(v) for v in g("cow2_ball_speed", [0.0, 0.0])]
        # Stage 3 incoming ball (doc 40 stage F): fraction of episodes where the ball starts far and rolls toward the robot (0 = off).
        self._cow2_d_in_prob = float(g("cow2_d_in_prob", 0.0))
        self._cow2_d_in_dist = [float(v) for v in g("cow2_d_in_dist", [2.0, 4.0])]
        self._cow2_d_in_ang_deg = float(g("cow2_d_in_ang_deg", 60.0))
        self._cow2_d_in_speed = [float(v) for v in g("cow2_d_in_speed", [0.5, 1.5])]
        self._cow2_d_in_jit_deg = float(g("cow2_d_in_jit_deg", 15.0))
        self._cow2_ctrl_cmd = [float(v) for v in g("cow2_ctrl_cmd", [0.0, 0.0])]  # heading-frame dribble command shown in stage A
        self._cow2_cap_dist = float(g("cow2_cap_dist", 0.7))
        # stage C carry cap: a TOUCH phase longer than touch_max_s with the ball still at the foot is a carry;
        # always counted in the diag (cow2c_carry), a failure (respawn, timeout penalty) only when touch_max_fail
        self._cow2_touch_max_s = float(g("cow2_touch_max_s", 1.0))
        self._cow2_touch_max_fail = bool(g("cow2_touch_max_fail", False))
        self._cow2_cyc_T = float(g("cow2_cyc_T", 4.0))  # stage C cycle window (diag only for now)
        self._cow2_cap_speed = float(g("cow2_cap_speed", 0.5))
        self._cow2_cap_hold = float(g("cow2_cap_hold", 0.5))
        self._cow2_seek_s = float(g("cow2_seek_s", 8.0))
        self._cow2_hold_s = float(g("cow2_hold_s", 4.0))
        self._cow2_lost_dist = float(g("cow2_lost_dist", 3.0))
        self._cow2_still_speed = float(g("cow2_still_speed", 0.1))
        self._reward_cow2_ctrl_w = float(g("reward_cow2_ctrl_w", 0.0))
        self._reward_cow2_chase_w = float(g("reward_cow2_chase_w", 0.0))
        self._reward_cow2_cap_w = float(g("reward_cow2_cap_w", 0.0))
        self._reward_cow2_idle_w = float(g("reward_cow2_idle_w", 0.0))
        self._reward_cow2_lost_w = float(g("reward_cow2_lost_w", 0.0))
        self._reward_cow2_timeout_w = float(g("reward_cow2_timeout_w", 0.0))
        self._reward_cow2_fall_w = float(g("reward_cow2_fall_w", 0.0))
        # stage B pre-touch shaping (doc 40 v2-14): potential w * delta cos(robot->ball, ball->P) in CTRL within
        # cow2_setup_dist of the ball ("get behind the ball on the line to P"); potential-based, cannot be farmed
        self._reward_cow2_setup_w = float(g("reward_cow2_setup_w", 0.0))
        self._cow2_setup_dist = float(g("cow2_setup_dist", 1.5))
        # stage B time cost (doc 40 v2-17): -w on every CTRL step (idle only covers the robot far from the ball;
        # standing next to it without touching cost nothing but the one-off timeout)
        self._reward_cow2_wait_w = float(g("reward_cow2_wait_w", 0.0))
        self._reward_cow2_cyc_w = float(g("reward_cow2_cyc_w", 0.0))  # stage C: bonus per cycle completed within cow2_cyc_T
        # fall termination (FAIL) penalty in the TASK stream of cow envs (doc 40 v2-5): the aux stream nets ~-7/step
        # alive, so its -200 alone makes an early fall pay off when the task stream has no per-step reward (B)
        self._reward_cow2_term_w = float(g("reward_cow2_term_w", 0.0))
        self._cow2_replace_reward = bool(g("cow2_replace_reward", True))
        # stage B (cow2_stage 2): touch the ball to a target P (doc 40 v2); P shown as a dribble command
        self._cow2_p_fwd = [float(v) for v in g("cow2_p_fwd", [1.0, 2.0])]
        self._cow2_p_lat = float(g("cow2_p_lat", 0.5))
        self._cow2_p_ang_deg = float(g("cow2_p_ang_deg", 0.0))  # > 0: P on a ring within +-ang (B3)
        self._cow2_p_scale = float(g("cow2_p_scale", 2.0))
        # doc 40 v2-28: command = launch speed sqrt(2 decel d) instead of d / p_scale (the dribble cmd is a ball velocity)
        self._cow2_p_cmd_speed = bool(g("cow2_p_cmd_speed", False))
        self._cow2_p_sigma = float(g("cow2_p_sigma", 1.0))
        self._cow2_hit_dist = float(g("cow2_hit_dist", 0.5))
        self._cow2_touch_speed = float(g("cow2_touch_speed", 0.6))
        self._cow2_touch_dist = float(g("cow2_touch_dist", 0.35))  # foot-ball centre distance of a touch (m)
        self._cow2_roll_zero_cmd = bool(g("cow2_roll_zero_cmd", False))  # command 0 (stand) while the ball rolls
        self._cow2_p_approach = bool(g("cow2_p_approach", True))  # draw P in the robot->ball frame (else heading)
        self._cow2_touch_end = float(g("cow2_touch_end", 0.4))
        self._cow2_rest_speed = float(g("cow2_rest_speed", 0.1))
        self._cow2_rest_hold = float(g("cow2_rest_hold", 0.3))
        self._cow2_roll_s = float(g("cow2_roll_s", 8.0))
        # doc 43 S2b power: arrival speed v_T at T drawn in U[cow2_pow_v] m/s with prob cow2_pow_prob; under constant
        # decel a, passing T at v_T == resting at T + u * v_T^2 / (2a) (same launch command), so P becomes that point
        # and T is kept for the pass metric (closest approach to T, speed there). Off by default.
        _pv = g("cow2_pow_v", [0.0, 0.0])
        self._cow2_pow_v = [float(_pv[0]), float(_pv[1])] if isinstance(_pv, (list, tuple)) else [0.0, float(_pv)]
        self._cow2_pow_prob = float(g("cow2_pow_prob", 0.0))
        self._cow2_pow_tol_v = float(g("cow2_pow_tol_v", 0.5))
        self._cow2_pow_tol_d = float(g("cow2_pow_tol_d", 0.6))
        self._reward_cow2_touch_w = float(g("reward_cow2_touch_w", 0.0))
        # stage C release shaping (doc 40 v2-57): at the 1st touch end, ball velocity vs the one that stops at P
        # (v_req = sqrt(2 a d) toward P); exp(-|v - v_req|^2 / rel_sigma^2). Diag always, reward only if w > 0
        self._reward_cow2_rel_w = float(g("reward_cow2_rel_w", 0.0))
        self._reward_cow2_pass_w = float(g("reward_cow2_pass_w", 0.0))  # doc 43 S2b: ball passes T at v_T (event)
        self._reward_cow2_sep_w = float(g("reward_cow2_sep_w", 0.0))  # c2c47: pay foot-ball separation (m/s) during TOUCH
        self._cow2_rel_sigma = float(g("cow2_rel_sigma", 0.5))
        self._reward_cow2_hit_w = float(g("reward_cow2_hit_w", 0.0))
        # "area" scenario (doc 43): stage-B touch to P, wide P, area radius growing with the distance d0 = |P - ball0|
        # (hit_dist + hit_k d0, sigma + sigma_k d0), success = ball rests in the area AND the robot ends still and upright
        # (area_still), pays a success bonus, penalises extra touches; after a rest the scenario is respawned around the
        # robot (area_respawn). All off by default: other envs are unchanged.
        self._cow2_hit_k = float(g("cow2_hit_k", 0.0))
        self._cow2_p_sigma_k = float(g("cow2_p_sigma_k", 0.0))
        self._cow2_area_still = bool(g("cow2_area_still", False))
        self._cow2_area_respawn = bool(g("cow2_area_respawn", False))
        self._cow2_area_ball_vmax = float(g("cow2_area_ball_vmax", 0.0))  # >0: ball may still roll slower than this inside the area
        self._cow2_area_tilt_cos = float(g("cow2_area_tilt_cos", 0.99))  # torso up-vector z >= this (about 8 deg)
        self._cow2_area_min_h = float(g("cow2_area_min_h", 0.55))
        self._cow2_area_wmax = float(g("cow2_area_wmax", 0.3))  # trunk angular speed limit (rad/s)
        self._cow2_still_sigma = float(g("cow2_still_sigma", 0.3))
        self._cow2_still_tilt_sigma = float(g("cow2_still_tilt_sigma", 0.0))  # >0: pose factor of the still reward
        self._reward_cow2_still_w = float(g("reward_cow2_still_w", 0.0))  # ROLL phase: exp(-(v/sigma)^2) of the robot speed
        self._reward_cow2_retouch_w = float(g("reward_cow2_retouch_w", 0.0))  # penalty per extra touch while rolling
        self._reward_cow2_succ_w = float(g("reward_cow2_succ_w", 0.0))  # bonus at area success
        # one touch, then the robot STAYS where it kicked while the ball rolls (doc 43): max drift from the touch-end spot
        # allowed for a success (0 = off), penalty weight on the drift (m, capped at 1) during the roll
        self._cow2_stay_r = float(g("cow2_stay_r", 0.0))
        self._cow2_stay_first = bool(g("cow2_stay_from_first", False))  # drift measured from the 1st touch (not the last touch end)
        self._cow2_stay_clamp = float(g("cow2_stay_clamp", 1.0))  # saturation of the drift penalty (m)
        self._cow2_stay_touch = bool(g("cow2_stay_in_touch", False))  # drift penalty also while the ball is at the foot
        self._cow2_max_touches = float(g("cow2_max_touches", 0.0))  # >0: success needs at most this many touches
        self._reward_cow2_stay_w = float(g("reward_cow2_stay_w", 0.0))
        self._reward_cow2_prog_w = float(g("reward_cow2_prog_w", 0.0))
        self._cow2_prog_roll_only = bool(g("cow2_prog_roll_only", False))  # c2c44: prog only while the ball rolls (no pay for carry)
        # stage D (cow2_stage 3, doc 40 sec 4b): ball at the robot's feet, still virtual opponent ahead, region beside/behind it;
        # success = robot and ball inside the region with the ball controlled; bonus decays with time and extra touches.
        # The opponent goes in the goal slots of the task obs (same dims, B3 warm-start), the region centre in the dribble slot.
        self._cow2_d_opp_fwd = [float(v) for v in g("cow2_d_opp_fwd", [1.5, 2.5])]
        self._cow2_d_opp_lat = float(g("cow2_d_opp_lat", 0.3))
        self._cow2_d_reg_dist = [float(v) for v in g("cow2_d_reg_dist", [1.0, 2.0])]  # region centre distance from the opponent
        self._cow2_d_reg_ang_deg = float(g("cow2_d_reg_ang_deg", 75.0))  # +- around "straight past the opponent"
        self._cow2_d_reg_r = float(g("cow2_d_reg_r", 1.0))
        self._cow2_d_tmax = float(g("cow2_d_tmax", 10.0))
        self._cow2_d_fast_s = float(g("cow2_d_fast_s", 6.0))  # gate: median time to the region (diag)
        self._cow2_d_opp_r = float(g("cow2_d_opp_r", 0.45))  # robot-opponent centre distance = contact
        self._cow2_d_bypass_clear = float(g("cow2_d_bypass_clear", 0.0))  # >0: teacher cmd aims beside the opponent while it blocks ball->R
        self._cow2_d_ball_opp_r = float(g("cow2_d_ball_opp_r", 0.4))  # ball-opponent centre distance = contact (terminal)
        self._cow2_d_v_arr = float(g("cow2_d_v_arr", 0.0))  # stage D teacher: ball passes P at v_arr m/s (0 = rests at P)
        self._cow2_d_ball_vmax = float(g("cow2_d_ball_vmax", 0.0))  # stage D success also needs ball speed <= vmax (0 = off)
        self._cow2_d_prog_clip = float(g("cow2_d_prog_clip", 1.0))  # stage D: |ball/robot progress speed| clip in m/s (1.0 = original; the reward saturates at 1 m/s)
        self._cow2_d_free_run = bool(g("cow2_d_free_run", False))  # stage D toque-longo: success drops the robot-ball distance (fd<=cap) and the bonus is paid on the ROBOT arrival time; use with cow2_d_recover_s as the window X
        self._cow2_d_recover_s = float(g("cow2_d_recover_s", 0.0))  # stage D option C: ball reaches the region first, robot must recover within this many s (0 = off = original rule); bonus paid on the ball-arrival time
        self._cow2_d_cmd_gain = float(g("cow2_d_cmd_gain", 1.0))  # stage D: teacher ball-speed command x gain (1 = calibrated launch speed)
        self._cow2_d_min_ball_opp = float(g("cow2_d_min_ball_opp", 0.0))  # doc 43: >0 pushes the opponent so it spawns at least this far from the ball (0 = off; ~8% of d_in spawns land inside the 0.4 m contact radius)
        self._cow2_d_opp_speed = float(g("cow2_d_opp_speed", 0.0))  # D3: >0 the opponent walks toward the ball (m/s); 0 = still
        self._reward_cow2_d_succ_w = float(g("reward_cow2_d_succ_w", 0.0))
        self._cow2_d_touch_k = float(g("cow2_d_touch_k", 0.0))  # bonus fraction lost per touch beyond the first
        self._reward_cow2_d_prog_w = float(g("reward_cow2_d_prog_w", 0.0))  # ball toward the region (m/s, clamped 1)
        self._reward_cow2_d_rprog_w = float(g("reward_cow2_d_rprog_w", 0.0))  # robot toward the region
        self._reward_cow2_d_ocont_w = float(g("reward_cow2_d_ocont_w", 0.0))  # per step of robot-opponent contact
        self._reward_cow2_d_verr_w = float(g("reward_cow2_d_verr_w", 0.0))  # stage D: cost on |teacher cmd - ball vel| (heading frame, clamped 2 m/s); always <= 0
        self._reward_cow2_d_xtouch_w = float(g("reward_cow2_d_xtouch_w", 0.0))  # per touch beyond the first, paid at the touch
        self._reward_cow2_d_bcont_w = float(g("reward_cow2_d_bcont_w", 0.0))  # ball hits the opponent (ends the scenario)
        self._reward_cow2_d_oob_w = float(g("reward_cow2_d_oob_w", 0.0))  # ball out of the field (ends the scenario)

        # spawn the ball in front of the robot (T1: "closer ball so robot can
        # actually reach and kick it") instead of uniformly over the field
        self._ball_spawn_near = bool(env_config.get("ball_spawn_near", True))
        self._ball_spawn_front_min = float(env_config.get("ball_spawn_front_min", 0.5))
        self._ball_spawn_front_max = float(env_config.get("ball_spawn_front_max", 2.0))
        # optional [min, max] used before the ball curriculum starts and blended into
        # the range above over the curriculum window (stage-0 standing kick -> walking)
        early = env_config.get("ball_spawn_front_early", None)
        self._ball_spawn_front_early = None if early is None else [float(early[0]), float(early[1])]
        self._ball_spawn_lateral = float(env_config.get("ball_spawn_lateral", 0.5))
        # soft resets (goal / out of bounds / perturb teleport) respawn the
        # ball uniformly over the field so kicking it out costs a walk
        self._ball_soft_reset_far = bool(env_config.get("ball_soft_reset_far", True))

        # random ball perturbations (teleport or velocity push)
        self._ball_perturb_time_min = float(env_config.get("ball_perturb_time_min", 4.0))
        self._ball_perturb_time_max = float(env_config.get("ball_perturb_time_max", 8.0))
        self._ball_perturb_teleport_prob = float(env_config.get("ball_perturb_teleport_prob", 0.5))
        self._ball_perturb_speed_min = float(env_config.get("ball_perturb_speed_min", 1.0))
        self._ball_perturb_speed_max = float(env_config.get("ball_perturb_speed_max", 4.0))

        # ball-event Gaussian curriculum (c4): probability that a random ball
        # event (perturbation, goal/OOB soft respawn) draws a Gaussian near the
        # robot (sigma = ball_event_gauss_sigma) instead of the baseline draw.
        # Ramps 0 -> ball_event_gauss_prob over the sample clock. Only the
        # spawn distribution of random events changes; falls never relocate
        # the ball, so there is no free-return loop across hard resets.
        self._ball_event_gauss_start_samples = \
            float(env_config.get("ball_event_gauss_start_samples", -1.0))
        self._ball_event_gauss_end_samples = \
            float(env_config.get("ball_event_gauss_end_samples", -1.0))
        self._ball_event_gauss_prob = float(env_config.get("ball_event_gauss_prob", 0.0))
        self._ball_event_gauss_sigma = float(env_config.get("ball_event_gauss_sigma", 1.0))

        # c5: freeze the approach/progress potentials while the ball is
        # unperceived. A hidden teleport paid the potential gap for free in
        # c3/c4 (free_return_step_mean +2.7/step): the robot idled on respawn
        # spots instead of searching. With the potential frozen, a hidden
        # teleport moves the ANCHOR, so no reward is collected until the ball
        # is seen again; the only way to profit is to look at the new ball.
        # prev_ball_pos always tracks the TRUE ball so reacquisition is
        # reward-neutral (no spike from the anchor jump).
        self._percep_potential_freeze = bool(env_config.get("percep_potential_freeze", False))
        # perturbations off entirely (evals already override the times)
        self._ball_perturb_enable = bool(env_config.get("ball_perturb_enable", True))

        # c7 active search: while the ball is unperceived, reward yaw sweep
        # rate of the head (finite difference of heading angle). No
        # privileged state: the bonus is earned by moving the head, which the
        # policy measures. Paper fig. 4D/E: turn toward where the ball left
        # the FOV and re-acquire it.
        self._search_sweep_w = float(env_config.get("search_sweep_w", 0.0))
        self._search_sweep_cap = float(env_config.get("search_sweep_cap", 1.5))  # rad/s

        # c10 search curriculum: growing share of spawns in the REAR cone of
        # the robot (starts outside the camera FOV). Head-only sweeps cannot
        # find a ball behind the robot; this forces the body-turn search the
        # paper shows in fig. 3B/C. Distribution change, not reward.
        self._ball_behind_prob = float(env_config.get("ball_behind_prob", 0.0))
        # k14: share of resets that start from the standing default pose at rest
        # (gait initiation, the deployment start) instead of a mid-motion RSI state
        self._stand_start_prob = float(env_config.get("stand_start_prob", 0.0))
        # doc 25 M6: optional pose of the stand starts (the deploy config's default_joint_pos, dof order of the
        # asset) and root height; None = init_pose. Eval-only by default (deploy spawn check).
        sdp = env_config.get("stand_start_dof_pos", None)
        self._stand_start_dof_pos = None if sdp is None else torch.tensor(sdp, device=device, dtype=torch.float)
        sz = env_config.get("stand_start_root_z", None)
        self._stand_start_root_z = None if sz is None else float(sz)
        self._ball_behind_start_samples = \
            float(env_config.get("ball_behind_start_samples", -1.0))
        self._ball_behind_end_samples = \
            float(env_config.get("ball_behind_end_samples", -1.0))
        self._ball_behind_dist_min = float(env_config.get("ball_behind_dist_min", 1.0))
        self._ball_behind_dist_max = float(env_config.get("ball_behind_dist_max", 4.0))
        self._ball_behind_half_cone = \
            float(env_config.get("ball_behind_half_cone_deg", 60.0)) * np.pi / 180.0


        # per-env static domain randomization (Frente C; ranges follow the
        # HTWK T1 deploy stack, which is validated on real hardware)
        self._rand_ball_props = bool(env_config.get("rand_ball_props", False))
        self._rand_ball_mass_scale = list(env_config.get("rand_ball_mass_scale", [0.7, 1.3]))
        # absolute ball mass (paper Table 2: U(0.4, 0.45) kg); overrides the scale.
        # Converted with the mass of the asset sphere (density 80, r 0.11 = 0.446 kg)
        mass_kg = env_config.get("rand_ball_mass_kg", None)
        if (mass_kg is not None):
            r = float(env_config.get("ball_radius", 0.11))
            default_kg = 80.0 * 4.0 / 3.0 * np.pi * r ** 3
            self._rand_ball_mass_scale = [float(mass_kg[0]) / default_kg, float(mass_kg[1]) / default_kg]
        self._rand_ball_friction = list(env_config.get("rand_ball_friction", [0.2, 1.2]))
        self._rand_ball_restitution = list(env_config.get("rand_ball_restitution", [0.1, 0.9]))

        self._rand_char_props = bool(env_config.get("rand_char_props", False))
        self._rand_char_friction = list(env_config.get("rand_char_friction", [0.1, 2.0]))
        self._rand_char_base_mass_scale = list(env_config.get("rand_char_base_mass_scale", [0.8, 1.2]))
        self._rand_char_base_com = list(env_config.get("rand_char_base_com", [-0.1, 0.1]))
        self._rand_char_other_mass_scale = list(env_config.get("rand_char_other_mass_scale", [0.98, 1.02]))

        # Table 2 domain randomization, shared with the steering env (same
        # sampler, same ranges). Replaces rand_char_props when on: the two draw
        # the same friction/mass/CoM with different ranges and must not stack.
        self._dr_enabled = bool(env_config.get("domain_randomization", False))
        # evaluation only: one fixed action delay (ms) for every env, with no other DR (doc 25, goal M4)
        self._fixed_action_delay_ms = env_config.get("fixed_action_delay_ms", None)
        self._dr_static_pending = []
        self._ball_props_pending = []
        self._dr_static_deferred = False  # set once the engine is known (see the post-init props loop)
        assert not (self._dr_enabled and self._rand_char_props), \
            "domain_randomization supersedes rand_char_props; set only one"
        self._dr_ranges = steering_dr.load_ranges(env_config) if self._dr_enabled else None
        self._dr_seed = env_config.get("dr_seed", None)
        self._dr_foot_bodies = list(env_config.get(
            "dr_foot_bodies", ["left_ankle_roll_link", "right_ankle_roll_link"]))
        # the critic sees torso mass/CoM (Table 3) only if asked: it changes the
        # critic input size, which breaks warm-starting from a checkpoint
        self._dr_critic_torso_obs = bool(env_config.get("dr_critic_torso_obs", False))
        self._dr_samples = []

        # paper Table 3 observation spaces (all off = the previous env, bit for bit).
        # disc_obs_table3: the Disc. column (gravity, ang vel, joints, joint vel,
        # base lin vel, feet in the base frame) instead of the generic AMP obs.
        # critic_paper_obs: the Critic column (adds the ball friction; the base
        # mass/CoM need dr_critic_torso_obs). recon_paper_targets: the Recon.
        # column (ball, base lin vel, base height, base mass/CoM, ball vel, ball friction).
        self._disc_table3 = bool(env_config.get("disc_obs_table3", False))
        self._disc_foot_bodies = list(env_config.get(
            "disc_foot_bodies", ["left_ankle_roll_link", "right_ankle_roll_link"]))
        self._critic_paper_obs = bool(env_config.get("critic_paper_obs", False))
        self._recon_paper_targets = bool(env_config.get("recon_paper_targets", False))
        # per-env ball friction coefficient, filled by _randomize_env_props
        self._ball_friction_samples = []

        # goal estimate from odometry (Table 3 goal position/direction, appendix F):
        # the actor sees the goal through a robot pose estimate that drifts. The
        # position error grows ~goal_odom_drift m/s along a heading that is redrawn
        # every few seconds and is capped at goal_odom_max (vision correction);
        # the yaw error is a bounded random walk. Privileged obs stay exact.
        self._goal_odom_noise = bool(env_config.get("goal_odom_noise", False))
        self._goal_odom_drift = float(env_config.get("goal_odom_drift", 0.04))      # m/s
        self._goal_odom_max = float(env_config.get("goal_odom_max", 1.0))           # m
        self._goal_odom_init_std = float(env_config.get("goal_odom_init_std", 0.1)) # m
        self._goal_odom_redraw = float(env_config.get("goal_odom_redraw", 5.0))     # s
        self._goal_odom_yaw_init_std = float(env_config.get("goal_odom_yaw_init_std_deg", 2.0)) * np.pi / 180.0
        self._goal_odom_yaw_drift = float(env_config.get("goal_odom_yaw_drift_deg", 0.3)) * np.pi / 180.0  # rad/sqrt(s)
        self._goal_odom_yaw_max = float(env_config.get("goal_odom_yaw_max_deg", 8.0)) * np.pi / 180.0

        # random velocity pushes on the robot (paper: physical confrontation;
        # magnitude matches the HTWK T1 push: ~10 N x 1 s / ~30 kg)
        self._char_push_enable = bool(env_config.get("char_push_enable", False))
        # Booster Gym world perturbations, as Olives' e80tonn1 (task_steering_meas_env, envs/t1.py _kick_robots /
        # _push_robots): ONE step counter shared by all envs; a velocity kick every kick_interval_s and a base-frame
        # force + torque held for push_duration_s every push_interval_s. push_impulse_scale 0.1 = Booster's pulse of
        # F x 0.002 s every 0.02 s held continuously over the substeps (scripts/probe_force_persistence.py).
        # Off unless an interval is set; switched on with the pushes (char_push_start_samples). doc 25 item 3.
        self._kick_interval_s = float(env_config.get("kick_interval_s", 0.0))
        self._kick_lin_vel_std = float(env_config.get("kick_lin_vel_std", 0.1))
        self._kick_ang_vel_std = float(env_config.get("kick_ang_vel_std", 0.02))
        self._push_interval_s = float(env_config.get("push_interval_s", 0.0))
        self._push_duration_s = float(env_config.get("push_duration_s", 1.0))
        self._push_force_std = float(env_config.get("push_force_std", 10.0))
        self._push_torque_std = float(env_config.get("push_torque_std", 2.0))
        self._push_impulse_scale = float(env_config.get("push_impulse_scale", 0.1))
        # Booster Gym draws foot friction / compliance / restitution per foot shape (doc 25 item 2)
        self._dr_foot_props_per_foot = bool(env_config.get("dr_foot_props_per_foot", False))
        # Table 3/4 joint-limit penalty on the commanded target instead of dof_pos (dof_pos is clamped by the
        # sim and almost never exceeds the limit), as Olives' base P (doc 25 item 4)
        self._joint_limit_on_action = bool(env_config.get("joint_limit_on_action", False))
        self._char_push_time_min = float(env_config.get("char_push_time_min", 5.0))
        self._char_push_time_max = float(env_config.get("char_push_time_max", 10.0))
        self._char_push_vel_std = float(env_config.get("char_push_vel_std", 0.3))
        self._char_push_ang_vel_std = float(env_config.get("char_push_ang_vel_std", 0.1))

        # virtual perception (Frente E, paper section 9): the actor's ball obs
        # go through a simulated camera pipeline (noise + latency + frame rate
        # + detection dropout); rewards and events keep the true state. The
        # obs CONTRACT is unchanged: the same ball slots + mask now carry the
        # perceived values instead of ground truth.
        self._virtual_perception = bool(env_config.get("virtual_perception", False))
        self._percep_freq_mean = float(env_config.get("percep_freq_mean", 25.36))     # Hz
        self._percep_freq_std = float(env_config.get("percep_freq_std", 1.06))        # Hz
        self._percep_latency_mean = float(env_config.get("percep_latency_mean", 0.116))  # s
        self._percep_latency_std = float(env_config.get("percep_latency_std", 0.018))   # s
        self._percep_noise_dist_coef = float(env_config.get("percep_noise_dist_coef", 0.124))
        self._percep_noise_base = float(env_config.get("percep_noise_base", 0.149))     # m
        self._percep_detect_prob = float(env_config.get("percep_detect_prob", 0.9))
        self._env_config_percep_ab = env_config.get("percep_ab_filter", None)  # [alpha, beta]
        # RT-01 deploy-perception emulator (docs/t1/37): estimated pose + 10 Hz ball with holes/outliers.
        # Off by default. scale multiplies the curriculum progress (1 = measured deploy, 1.5 = stress).
        # percep_rt01_filter [alpha, beta]: alpha-beta filter on the held field-frame ball (variant p3a).
        self._percep_rt01 = bool(env_config.get("percep_rt01", False))
        self._percep_rt01_scale = float(env_config.get("percep_rt01_scale", 1.0))
        # per-episode scale multipliers drawn at reset (e.g. [0, 0.5, 1, 1.5]); None = one global scale
        self._percep_rt01_scale_mix = env_config.get("percep_rt01_scale_mix", None)
        self._percep_rt01_filter = env_config.get("percep_rt01_filter", None)
        self._percep_rt01_params = env_config.get("percep_rt01_params", None)
        self._percep_rt01_pose_mult = float(env_config.get("percep_rt01_pose_mult", 1.0))
        self._percep_rt01_ball_mult = float(env_config.get("percep_rt01_ball_mult", 1.0))
        # subtract the known mean detection bias at the measured distance (deploy calibration, goes to policy_io)
        self._percep_rt01_bias_correct = bool(env_config.get("percep_rt01_bias_correct", False))
        # expanding mean: gain max(alpha, 1/(n+1)) over the n measurements fused this episode, so the first delivered
        # measurement replaces the clean reset value and a static ball converges to the running mean
        self._percep_ab_expand = bool(env_config.get("percep_ab_expand", False))
        # Kalman position filter (doc 35, B1): constant-position model, process noise q m^2/s, measurement variance
        # from the noise model at the measured distance; a measurement whose normalized innovation exceeds the gate is
        # dropped, and a second one in a row resets the filter to it (the ball was kicked). q <= 0 disables.
        self._percep_kf_q = float(env_config.get("percep_kf_q", 0.0))
        self._percep_kf_gate = float(env_config.get("percep_kf_gate", 0.0))
        # constant-velocity state (pos + vel, q = white-noise accel) with the
        # same latency look-ahead as the alpha-beta path; false = position-only KF
        self._percep_kf_cv = bool(env_config.get("percep_kf_cv", False))
        self._percep_detect_full_range = float(env_config.get("percep_detect_full_range", 7.0))  # m
        self._percep_detect_decay_range = float(env_config.get("percep_detect_decay_range", 3.0))  # m
        # full FOV angle in deg; <= 0 disables the FOV check. G1 has no
        # actuated neck (Frente G pending), so the FOV is fixed to the heading.
        self._percep_fov_deg = float(env_config.get("percep_fov_deg", 120.0))
        # camera body for the FOV (Frente G): when set, the FOV cone follows
        # this body's +x axis (T1 actuated head) instead of the root heading;
        # empty string keeps the G1 heading-fixed behaviour
        self._percep_fov_body = str(env_config.get("percep_fov_body", ""))

        # task-obs history (Frente F-lite): the actor also receives the last
        # H task blocks (steer 5 + soccer 6 + mask 1 = 12 dims) it saw, so it
        # can filter perception noise and estimate ball velocity / latency.
        # 0 disables (contract stays 249 dims). Paper uses 1 s of history
        # through an encoder; 10 frames (~333 ms) cover latency (~3.5 steps)
        # + velocity estimation.
        self._task_hist_steps = int(env_config.get("task_obs_history_steps", 0))

        # measurable actor obs (Frente F, paper Table 2): the actor obs become
        # [measurable frame | H past frames] where a frame is projected
        # gravity + base angular velocity + joint offsets + joint velocities +
        # previous action + the 12-dim task block. The full-state char obs
        # move to the privileged critic_obs. CONTRACT BREAK: no warm start
        # from full-state checkpoints. Default 30 frames = 1 s at 30 Hz
        # (paper: 50 frames at 50 Hz).
        self._measurable_obs = bool(env_config.get("measurable_obs", False))
        self._meas_hist_steps = int(env_config.get("measurable_hist_steps", 30))
        # sensor noise on the actor's measurable frame (docs/t1/13); every std
        # defaults to 0, which leaves the env bit-identical
        self._obs_noise_cfg = {k: env_config[k]
                               for k in steering_util.OBS_NOISE_KEYS + ("obs_noise_scale",)
                               if k in env_config}
        assert not (self._measurable_obs and self._task_hist_steps > 0), \
            "measurable_obs already carries the task block in its history; " \
            "task_obs_history_steps must be 0"
        assert (not self._measurable_obs) or self._meas_hist_steps > 0, \
            "measurable_obs requires measurable_hist_steps > 0 (the encoder " \
            "needs a history)"

        # steering-crutch anneal (Frente D): the steering obs block is scaled
        # 1 -> 0 between start and end samples (the paper's actor receives no
        # steering command). start < 0 disables; start == end == 0 zeroes it
        # from the first step (eval configs).
        self._steer_anneal_start_samples = float(env_config.get("steer_anneal_start_samples", -1.0))
        self._steer_anneal_end_samples = float(env_config.get("steer_anneal_end_samples", -1.0))

        # perception curriculum (A): ramp from perfect perception (360 deg
        # FOV, no misses/latency/noise) to the configured realistic camera
        # over a sample budget. start < 0 disables (realistic from step 0).
        self._percep_curriculum_start_samples = \
            float(env_config.get("percep_curriculum_start_samples", -1.0))
        self._percep_curriculum_end_samples = \
            float(env_config.get("percep_curriculum_end_samples", -1.0))

        # ball-spawn curriculum (C): ramp the probability of a uniform
        # field-wide spawn from 0 to ball_curriculum_far_prob, keeping the
        # near frontal spawn as the base. start < 0 disables.
        self._ball_curriculum_start_samples = \
            float(env_config.get("ball_curriculum_start_samples", -1.0))
        self._ball_curriculum_end_samples = \
            float(env_config.get("ball_curriculum_end_samples", -1.0))
        self._ball_curriculum_far_prob = float(env_config.get("ball_curriculum_far_prob", 0.5))

        # schedule clock offset: samples already consumed by a run this one
        # resumes from, so steer anneal and curricula continue where they were
        self._curriculum_sample_offset = float(env_config.get("curriculum_sample_offset", 0.0))
        # performance gates (see soccer_util.GateClock): the schedule clock parks at each
        # gate until the stage metrics pass; no gates = the plain sample count
        # kick-heavy motion sampling until the gated clock reaches `motion_kick_weight_end_samples`
        # (stage-0 standing kick prior), then the dataset's own weights; off when the key is absent
        self._motion_kick_w_early = env_config.get("motion_kick_weight_early", None)
        self._motion_kick_w_end = float(env_config.get("motion_kick_weight_end_samples", -1.0))
        self._motion_kick_w_ramp = float(env_config.get("motion_kick_weight_ramp_samples", 0.0))
        self._motion_weights_true = None
        self._motion_weights_early = None
        # per-stage scalar overrides on the gated clock: [{at: clock, key: value, ...}], applied once each
        self._stage_params = sorted([dict(d) for d in env_config.get("stage_params", [])],
                                    key=lambda d: float(d["at"]))
        for d in self._stage_params:
            assert set(d) - {"at"} <= self._STAGE_PARAM_KEYS, "unknown stage_params key in {}".format(d)
        self._stage_params_applied = 0
        self._gate_clock = soccer_util.GateClock(
            soccer_util.parse_curriculum_gates(env_config.get("curriculum_gates", [])),
            self._curriculum_sample_offset)
        # switches on the schedule clock (negative = always on, the old behavior)
        self._char_push_start_samples = float(env_config.get("char_push_start_samples", -1.0))
        self._ball_perturb_start_samples = float(env_config.get("ball_perturb_start_samples", -1.0))
        self._reward_stagnation_start_samples = float(env_config.get("reward_stagnation_start_samples", -1.0))
        self._goal_odom_start_samples = float(env_config.get("goal_odom_start_samples", -1.0))
        self._goal_odom_end_samples = float(env_config.get("goal_odom_end_samples", -1.0))
        self._dr_runtime_start_samples = float(env_config.get("dr_runtime_start_samples", -1.0))
        self._dr_runtime_pending = False
        self._episode_length_late = env_config.get("episode_length_late", None)
        self._episode_length_late_start_samples = float(env_config.get("episode_length_late_start_samples", -1.0))

        # uneven ground (engine-side): inject one tile per field so the
        # engine's ground meshes and the env's field grid cannot diverge;
        # spawns are raised by the bump amplitude
        self._ground_z_offset = 0.0
        # per-env spawn lift [num_envs] when ground.tile_levels mixes amplitudes (None = the scalar above)
        self._ground_z_offset_env = None
        self._build_field_offsets = None
        ground_config = engine_config.get("ground", None)
        if (ground_config is not None and ground_config.get("type", "plane") == "uneven"):
            offsets = soccer_util.compute_field_offsets(
                num_envs, self._field_length, self._field_width, self._field_sep)
            if ("tile_centers" not in ground_config):
                ground_config["tile_centers"] = offsets.tolist()
                ground_config["tile_size"] = [self._field_length + 2.0 * self._field_sep,
                                              self._field_width + 2.0 * self._field_sep]
            self._ground_z_offset = float(ground_config.get("random_height", 0.02))
            if (ground_config.get("tile_levels", None) or ground_config.get("step_height_range", None)):
                # each field is lifted by its own tile's bound (bumps + tallest step), same
                # round-robin as the engine (env i <-> tile i), so a 1 mm field does not drop
                # the robot and the ball from the 5 cm one's height
                tile_cfgs = terrain_util.resolve_tile_configs(ground_config, num_envs)
                tile_z = [terrain_util.tile_max_height(c) for c in tile_cfgs]
                self._ground_z_offset = max(tile_z)
                self._ground_z_offset_env = torch.tensor(tile_z, device=device, dtype=torch.float)  # self._device is set by super().__init__ below
            # uneven ground uses env_spacing 0 (one global mesh region per
            # field); spread actors over their fields already at creation so
            # they are not all piled at the origin, which explodes the PhysX
            # GPU broadphase pair count before the first reset
            self._build_field_offsets = offsets

        # goal-stream reward weights (Table 3)
        self._reward_goal_scored_w = float(env_config.get("reward_goal_scored_w", 15.0))
        self._reward_ball_oob_w = float(env_config.get("reward_ball_oob_w", 0.0))
        self._reward_ball_approach_w = float(env_config.get("reward_ball_approach_w", 50.0))
        self._reward_goal_progress_w = float(env_config.get("reward_goal_progress_w", 500.0))

        # dense chase reward (steering velocity tracking toward the ball); off by default
        self._reward_ball_chase_w = float(env_config.get("reward_ball_chase_w", 0.0))
        self._ball_chase_vel_scale = float(env_config.get("ball_chase_vel_scale", 1.0))
        # facing-the-ball reward (turn toward a ball behind the robot); off by default
        self._reward_ball_align_w = float(env_config.get("reward_ball_align_w", 0.0))
        self._ball_align_angle_scale = float(env_config.get("ball_align_angle_scale", 1.0))
        # subtracted from the align reward where it is active: 1.0 turns it into a
        # penalty that is 0 when facing the ball (k17 paid +w for standing still facing it)
        self._ball_align_baseline = float(env_config.get("ball_align_baseline", 0.0))

        # directional kick reward (T1 kicking env): ball velocity toward the
        # goal above a threshold, credit concentrated at the impact
        self._reward_kick_direction_w = float(env_config.get("reward_kick_direction_w", 25.0))
        self._kick_direction_min_vel = float(env_config.get("kick_direction_min_vel", 1.0))
        self._kick_direction_decay = float(env_config.get("kick_direction_decay", 0.2))
        self._kick_direction_max = float(env_config.get("kick_direction_max", 30.0))
        # kick target T (docs/t1/43): 0 = always the goal (unchanged behaviour)
        self._kick_tgt_prob = float(env_config.get("kick_tgt_prob", 0.0))
        self._kick_tgt_dist = [float(v) for v in env_config.get("kick_tgt_dist", [3.0, 10.0])]
        self._kick_tgt_ang_deg = float(env_config.get("kick_tgt_ang_deg", 60.0))
        self._kick_tgt_ang_deg_start = float(env_config.get("kick_tgt_ang_deg_start", 0.0))
        self._kick_tgt_ang_start_samples = int(env_config.get("kick_tgt_ang_start_samples", -1))
        self._kick_tgt_ang_end_samples = int(env_config.get("kick_tgt_ang_end_samples", -1))
        # kick power (docs/t1/44): on the kick-target episodes (kick_tgt_prob) a power p in [0, 1] is drawn and
        # shown in the steering slots as [0, 0, p, -1, 0]; p commands the ball launch speed vmin + p*(vmax-vmin).
        # Off by default (kick_pow_enable false leaves every output unchanged).
        self._kick_pow_enable = bool(env_config.get("kick_pow_enable", False))
        self._kick_pow_vmin = float(env_config.get("kick_pow_vmin", 0.8))
        self._kick_pow_vmax = float(env_config.get("kick_pow_vmax", 7.0))
        self._kick_pow_p_lo = float(env_config.get("kick_pow_p_lo", 0.0))
        self._kick_pow_p_slot4 = bool(env_config.get("kick_pow_p_slot4", False))  # kpx5: p also in steering slot 4
        self._kick_pow_p_lo_start = float(env_config.get("kick_pow_p_lo_start", self._kick_pow_p_lo))
        self._kick_pow_p_lo_start_samples = int(env_config.get("kick_pow_p_lo_start_samples", -1))
        self._kick_pow_p_lo_end_samples = int(env_config.get("kick_pow_p_lo_end_samples", -1))
        self._kick_pow_p_fixed = float(env_config.get("kick_pow_p_fixed", -1.0))  # >= 0: eval at a fixed p
        # kpx6: adaptive top-down curriculum on p (soccer_util.update_kick_pow_adapt); off by default
        self._kick_pow_adapt = bool(env_config.get("kick_pow_adapt", False))
        self._kick_pow_adapt_lo = float(env_config.get("kick_pow_adapt_lo_start", 0.68))
        self._kick_pow_adapt_lo_min = float(env_config.get("kick_pow_adapt_lo_min", 0.0))
        self._kick_pow_adapt_win = float(env_config.get("kick_pow_adapt_win", 0.15))
        self._kick_pow_adapt_front_prob = float(env_config.get("kick_pow_adapt_front_prob", 0.5))
        self._kick_pow_adapt_thr = float(env_config.get("kick_pow_adapt_thr", 0.5))
        self._kick_pow_adapt_step = float(env_config.get("kick_pow_adapt_step", 0.05))
        self._kick_pow_adapt_min_events = int(env_config.get("kick_pow_adapt_min_events", 300))
        self._kick_pow_adapt_every = int(env_config.get("kick_pow_adapt_every", 30))
        # kpf2: >= 0 switches the ladder criterion to |mean signed error| <= bias_tol (update_kick_pow_adapt_bias)
        self._kick_pow_adapt_bias_tol = float(env_config.get("kick_pow_adapt_bias_tol", -1.0))
        self._kick_pow_adapt_s_err = 0.0
        # kpf3: count only the deterministic test-phase kicks (the exploration noise biases training kicks low)
        self._kick_pow_adapt_test_only = bool(env_config.get("kick_pow_adapt_test_only", False))
        self._kick_pow_adapt_tick = 0
        self._kick_pow_adapt_n_ev = 0
        self._kick_pow_adapt_n_ok = 0
        self._kick_pow_window_steps = int(env_config.get("kick_pow_window_steps", 6))
        self._kick_pow_sigma_a = float(env_config.get("kick_pow_sigma_a", 0.3))
        self._kick_pow_sigma_b = float(env_config.get("kick_pow_sigma_b", 0.1))
        self._kick_pow_tail_w = float(env_config.get("kick_pow_tail_w", 0.0))
        self._kick_pow_tail_s = float(env_config.get("kick_pow_tail_s", 2.0))
        self._kick_pow_dir_sigma_deg = float(env_config.get("kick_pow_dir_sigma_deg", 10.0))
        self._kick_pow_retouch_s = float(env_config.get("kick_pow_retouch_s", 1.5))
        self._kick_pow_end_s = float(env_config.get("kick_pow_end_s", -1.0))  # >= 0: SUCC (soft) this long after pay
        self._reward_kick_pow_speed_w = float(env_config.get("reward_kick_pow_speed_w", 0.0))
        self._reward_kick_pow_dir_w = float(env_config.get("reward_kick_pow_dir_w", 0.0))
        self._reward_kick_pow_retouch_w = float(env_config.get("reward_kick_pow_retouch_w", 0.0))

        # ball-state reward gating (T1 ball_rolling_scale): while the ball
        # rolls, the robot->ball approach potential is zeroed so chasing the
        # ball it just kicked cannot be farmed
        self._ball_rolling_speed = float(env_config.get("ball_rolling_speed", 0.1))
        self._gate_approach_when_rolling = bool(env_config.get("gate_approach_when_rolling", True))
        # terrain: speed under which the ball still counts as kickable for the kick shaping only
        # (foot_ball potential, side swing); bumps keep the ball rolling slowly and starve them.
        # None = ball_rolling_speed (identical to before); the approach gate, the waiting timer and
        # the kick-direction timer keep using ball_rolling_speed
        self._kick_still_speed = env_config.get("kick_still_speed", None)
        if (self._kick_still_speed is not None):
            self._kick_still_speed = float(self._kick_still_speed)

        # quadratic waiting penalty on ball-still time (T1); saturates at
        # waiting_time_max so 60 s episodes cannot blow it up
        self._reward_waiting_w = float(env_config.get("reward_waiting_w", -3.0))
        self._waiting_time_max = float(env_config.get("waiting_time_max", 3.0))

        # aux-stream reward weights (Table 3, signed). Head terms and the
        # non-foot collision penalty are omitted for the G1 (no actuated
        # head; falls already terminate the episode).
        self._reward_survival_w = float(env_config.get("reward_survival_w", 3.0))
        self._reward_termination_w = float(env_config.get("reward_termination_w", -1000.0))
        self._reward_stagnation_w = float(env_config.get("reward_stagnation_w", -100.0))
        self._reward_kick_sideways_w = float(env_config.get("reward_kick_sideways_w", 20.0))
        self._reward_kick_forward_w = float(env_config.get("reward_kick_forward_w", -20.0))
        self._reward_kick_alignment_w = float(env_config.get("reward_kick_alignment_w", 0.0))
        self._kick_alignment_power = float(env_config.get("kick_alignment_power", 4.0))
        # c9: dense shaping for setting up behind the ball (the aiming
        # bottleneck is WHERE the kick starts, not the impact)
        self._reward_kick_position_w = float(env_config.get("reward_kick_position_w", 0.0))
        self._kick_position_near_dist = float(env_config.get("kick_position_near_dist", 1.5))  # m
        self._reward_foot_proximity_w = float(env_config.get("reward_foot_proximity_w", -5.0))
        self._reward_action_rate_w = float(env_config.get("reward_action_rate_w", -1.0))
        self._reward_joint_limit_w = float(env_config.get("reward_joint_limit_w", -100.0))
        # knee-bend shortfall penalty (diagnostic crutch for the straight-leg gait); off by default
        self._reward_knee_bend_w = float(env_config.get("reward_knee_bend_w", 0.0))
        self._knee_bend_min = float(env_config.get("knee_bend_min", 0.25))
        # forward torso-lean penalty (diagnostic crutch, like the knee one); off by default
        self._reward_torso_lean_w = float(env_config.get("reward_torso_lean_w", 0.0))
        self._torso_lean_max = float(env_config.get("torso_lean_max", 0.15))
        # near-ball multiplier on the lean/arm penalties: 1 + extra*(1 - d/radius), d = distance to the ball
        self._near_ball_pen_extra = float(env_config.get("near_ball_penalty_extra", 0.0))
        self._near_ball_pen_radius = float(env_config.get("near_ball_penalty_radius", 1.5))
        # arm-pose penalty (diagnostic crutch): shoulder pitch/roll held near the omni data's pose
        self._reward_arm_pose_w = float(env_config.get("reward_arm_pose_w", 0.0))
        self._arm_pose_tol = float(env_config.get("arm_pose_tol", 0.3))
        self._arm_dof_bodies = list(env_config.get("arm_dof_bodies", [
            "left_shoulder_pitch_link", "left_shoulder_roll_link",
            "right_shoulder_pitch_link", "right_shoulder_roll_link"]))
        self._arm_pose_target_cfg = list(env_config.get("arm_pose_target", [0.05, -1.25, 0.08, 1.27]))
        self._knee_dof_bodies = list(env_config.get(
            "knee_dof_bodies", ["left_knee_pitch_link", "right_knee_pitch_link"]))
        self._reward_base_accel_w = float(env_config.get("reward_base_accel_w", -0.001))
        # head gaze reward (Frente G, paper Table 3 head term): cosine of the
        # angle between the camera body boresight and the ball; requires
        # percep_fov_body. 0 disables (G1 baseline parity).
        self._reward_head_gaze_w = float(env_config.get("reward_head_gaze_w", 0.0))
        # k1 stage-0 shaping (task stream, both 0 = identical env): potential on the
        # nearest-foot -> ball distance while the ball is still, and a bonus when a
        # foot within kick_contact_dist raises the ball speed by >= kick_contact_min_gain
        self._reward_foot_ball_w = float(env_config.get("reward_foot_ball_w", 0.0))
        self._reward_kick_contact_w = float(env_config.get("reward_kick_contact_w", 0.0))
        self._kick_contact_dist = float(env_config.get("kick_contact_dist", 0.12))  # m
        self._kick_contact_min_gain = float(env_config.get("kick_contact_min_gain", 0.3))  # m/s
        self._kick_contact_gain_cap = float(env_config.get("kick_contact_gain_cap", 2.0))  # m/s
        self._kick_contact_min_height = float(env_config.get("kick_contact_min_height", 0.4))  # m, root z
        # >0: the kick bonus is paid only after the robot stays up this many control steps
        # after the contact (a fall in between cancels it); 0 = paid at the contact
        self._kick_contact_hold_steps = int(env_config.get("kick_contact_hold_steps", 0))
        # penalty (task stream) on the ball-speed gain above kick_overspeed_limit at a foot contact
        # (violent lunge kicks that end in a fall); 0 = off
        self._reward_kick_overspeed_w = float(env_config.get("reward_kick_overspeed_w", 0.0))
        self._kick_overspeed_limit = float(env_config.get("kick_overspeed_limit", 3.5))  # m/s
        # k64f: no overspeed penalty while the ball is closer than this to the goal point (m; 0 = always penalised).
        # Near the goal the ball crosses in 1-3 steps and the goal step zeroes the shaping, so a normal kick pays
        # less than the penalty and the policy learned to park instead of kicking (grid cells x>=5.5, |y|<=1).
        self._kick_overspeed_goal_exempt = float(env_config.get("kick_overspeed_goal_exempt", 0.0))
        # True: penalize the ball speed itself while a foot is near (the gain per step misses a hit that
        # spreads over several control steps); False: per-step gain (k5 behaviour)
        self._kick_overspeed_use_speed = bool(env_config.get("kick_overspeed_use_speed", False))
        # impulsive kick (user 2026-10-21 "chapa mais forte, menos empurrao"): at a kick event pays
        # aim_gate * min(gain, kick_impulse_cap)^2, convex in the one-step ball-speed gain, so one sharp
        # strike pays more than the same speed added over several pushing steps; 0 = off
        self._reward_kick_impulse_w = float(env_config.get("reward_kick_impulse_w", 0.0))
        self._kick_impulse_cap = float(env_config.get("kick_impulse_cap", 8.0))  # m/s
        # >0: the contact/impulse caps and the overspeed limit grow with the ball-goal distance d,
        # cap(d) = clamp(kick_gain_cap_per_m * d, cap, kick_gain_cap_max) and limit(d) = limit + cap(d) - cap,
        # so a short shot is paid only for a soft strike and a long one still for the speed it needs
        # (ks4: fixed 4 m/s caps cut the 6-99 m bucket to 20-30%); 0 = fixed caps
        self._kick_gain_cap_per_m = float(env_config.get("kick_gain_cap_per_m", 0.0))  # (m/s)/m
        self._kick_gain_cap_max = float(env_config.get("kick_gain_cap_max", 8.0))  # m/s
        # EMA on the PD target: 1.0 is off; the first step after a reset takes the raw action
        self._action_filter_alpha = float(env_config.get("action_filter_alpha", 1.0))
        # filter only in kick mode (nav/drib take the raw action; the EMA state tracks it, so a switch is smooth)
        self._action_filter_kick_only = bool(env_config.get("action_filter_kick_only", False))
        # with kick_only: alpha ramps linearly from 1 (raw) to action_filter_alpha over this many
        # seconds after the episode enters kick mode (nav->kick switch); 0 = off (full filter at once)
        self._action_filter_kick_ramp_s = float(env_config.get("action_filter_kick_ramp_s", 0.0))
        # inside-foot (side-foot) kick: the contact bonus is scaled by (1 - mix) + mix * side, with
        # side = |sin| of the angle between the nearest foot's horizontal long axis and the ball launch
        # direction (0 = toe poke, 1 = inside of the foot); 0 = off, the bonus is unchanged
        self._kick_side_foot_mix = float(env_config.get("kick_side_foot_mix", 0.0))
        # foot-pitch gate on the kick bonus (doc 25, goal M13): the bonus is scaled by (1 - mix) + mix * g,
        # g = exp(-(excess / sigma)^2), excess = max(|pitch| - tol, 0) in degrees of the nearest foot's long axis
        # above the horizontal at the kick step (toe pointing down = toe poke; the dataset's kicks are within 35 deg).
        # 0 = off, the bonus is unchanged
        self._kick_pitch_mix = float(env_config.get("kick_pitch_mix", 0.0))
        self._kick_pitch_tol_deg = float(env_config.get("kick_pitch_tol_deg", 35.0))
        # aim gate on the kick bonus (doc 25, M1): the bonus is scaled by (1 - mix) + mix * g, g = exp(-(excess / sigma)^2),
        # excess = max(err - half, 0) in degrees, err = angle between the ball velocity at the kick step and ball->goal,
        # half = atan((goal_width / 2 - margin) / dist_to_goal) (the shot's angular tolerance; ~5 deg at 12 m).
        # The contact bonus is otherwise blind to direction. 0 = off, the bonus is unchanged
        self._kick_aim_mix = float(env_config.get("kick_aim_mix", 0.0))
        self._kick_aim_margin = float(env_config.get("kick_aim_margin", 0.3))
        self._kick_aim_sigma_deg = float(env_config.get("kick_aim_sigma_deg", 3.0))
        # the positive kick-style terms (pitch target, toe out, contact side) pay only for an aimed launch
        self._kick_style_aim_mix = float(env_config.get("kick_style_aim_mix", 0.0))
        self._kick_pitch_sigma_deg = float(env_config.get("kick_pitch_sigma_deg", 25.0))
        self._kick_pitch_gate_prev = bool(env_config.get("kick_pitch_gate_prev", False))
        # dense foot-pitch shaping (doc 25, M13): the gate above only acts at the kick step and the k74T
        # policies almost never reach it (|pitch| ~65 deg), so it gives no direction. This pays
        # w * (excess / 45 deg) on every step the nearest foot is within kick_pitch_dense_dist of a still
        # ball, excess as in the gate. w <= 0; 0 = off (the raw value is logged anyway)
        self._reward_kick_pitch_dense_w = float(env_config.get("reward_kick_pitch_dense_w", 0.0))
        self._kick_pitch_dense_dist = float(env_config.get("kick_pitch_dense_dist", 0.4))
        # doc 25 C(b): style gain near the ball. The agent multiplies the disc reward of a step by
        # info["style_gain"] = style_near_ball_gain while the nearest foot is within style_near_ball_dist of the
        # ball (C0: k75b style score 0.16-0.38 there vs 0.93 far away), else 1. 1.0 = off
        self._style_near_ball_gain = float(env_config.get("style_near_ball_gain", 1.0))
        self._style_near_ball_dist = float(env_config.get("style_near_ball_dist", 0.4))
        # doc 25 C(c): demo windows of the kick critic (agent kick_disc_weight). {clip basename: strike frame}
        # (scripts/lab/kick_foot_profile.py peak_frame); the window's last frame is drawn in
        # [strike - kick_disc_pre_s, strike + kick_disc_post_s]. Only read by fetch_kick_disc_obs_demo
        self._kick_disc_peak_frames = dict(env_config.get("kick_disc_peak_frames", {}))
        self._kick_disc_pre_s = float(env_config.get("kick_disc_pre_s", 0.1))
        self._kick_disc_post_s = float(env_config.get("kick_disc_post_s", 0.3))
        self._kick_disc_ids = None
        # k75: side-foot ("chapa") bonus at kick events, from the foot state of the step BEFORE the event:
        # |sin| between the foot's planar velocity and its planar long axis, times the pitch gate of that
        # step's pitch (at the event step the foot is already flattened: k74b -29 deg there vs -65 deg before)
        self._reward_kick_chapa_w = float(env_config.get("reward_kick_chapa_w", 0.0))
        # doc 25 C(a): foot-pitch target (dataset kicks: instep, pitch ~ -15 deg). g = exp(-((pitch - target) / sigma)^2);
        # event term pays g of the nearest foot's pitch on the step BEFORE each kick event, dense term pays g every
        # step the nearest foot is within kick_pitch_dense_dist of a still ball. 0 = off (raw values logged anyway)
        self._reward_kick_pitch_target_w = float(env_config.get("reward_kick_pitch_target_w", 0.0))
        self._reward_kick_pitch_target_dense_w = float(env_config.get("reward_kick_pitch_target_dense_w", 0.0))
        self._kick_pitch_target_deg = float(env_config.get("kick_pitch_target_deg", -15.0))
        self._kick_pitch_target_sigma_deg = float(env_config.get("kick_pitch_target_sigma_deg", 20.0))
        # doc 25 M13 (2026-10-19): kick like the dataset clips, no toe poke. From the previous step's nearest foot
        # and ball, at kick events: toe-face indicator (the point of the foot collision box nearest the ball centre
        # lies on the front face, MuJoCo foot_face rule) and toe-out g = exp(-((toe_out - target) / sigma)^2), toe_out = foot yaw - root
        # heading, outward positive (dataset kicks +9..+38, ~+20 deg; scripts/lab/kick_foot_profile.py). The dense
        # term pays g of the nearest foot every step it is within kick_pitch_dense_dist of a still ball. 0 = off
        self._reward_kick_toe_face_w = float(env_config.get("reward_kick_toe_face_w", 0.0))
        self._reward_kick_toe_out_w = float(env_config.get("reward_kick_toe_out_w", 0.0))
        self._reward_kick_toe_out_dense_w = float(env_config.get("reward_kick_toe_out_dense_w", 0.0))
        self._kick_toe_out_target_deg = float(env_config.get("kick_toe_out_target_deg", 20.0))
        self._kick_toe_out_sigma_deg = float(env_config.get("kick_toe_out_sigma_deg", 12.0))
        # strike direction (doc 25 M13, 2026-10-19): at kick events, |cos| between the nearest foot's previous-step
        # horizontal velocity and its long axis; the dataset strikes at median 0.85 (0.53..1.0; 1.0 = straight
        # along the foot = toe poke). Pays exp(-((|cos| - target) / sigma)^2). 0 = off
        self._reward_kick_dir_w = float(env_config.get("reward_kick_dir_w", 0.0))
        self._kick_dir_cos_target = float(env_config.get("kick_dir_cos_target", 0.85))
        self._kick_dir_cos_sigma = float(env_config.get("kick_dir_cos_sigma", 0.15))
        # contact side (doc 25 M13): the k85b already strikes at the dataset |cos| (0.85, k90s smoke) and still
        # pokes with the toe (84%), so the face comes from where the ball sits on the foot. medial = cosine between
        # the horizontal foot -> ball direction (foot frame, from the box centre) and the foot's inner side axis,
        # clamped at 0 (1 = ball on the inner side, chapa; 0 = ahead of the toe or outside). Event term at kick
        # events (previous step), dense term every step within kick_pitch_dense_dist of a still ball. 0 = off
        self._reward_kick_medial_w = float(env_config.get("reward_kick_medial_w", 0.0))
        self._reward_kick_medial_dense_w = float(env_config.get("reward_kick_medial_dense_w", 0.0))
        # ankle_roll_link collision box of data/assets/t1_deploy/t1.xml (centre, half sizes; foot frame)
        self._kick_foot_box_center = list(env_config.get("kick_foot_box_center", [0.0101079, 0.0, -0.0214208]))
        self._kick_foot_box_half = list(env_config.get("kick_foot_box_half", [0.112434, 0.05, 0.02183]))
        # arm-speed penalty (aux stream, doc 25, goal M12): mean over the arm joints of max(|qd| - tol, 0)^2;
        # tol 1.0 rad/s leaves the dataset's own arm motion (RMS 0.78 walking) free. 0 = off
        self._reward_arm_vel_w = float(env_config.get("reward_arm_vel_w", 0.0))
        # joint-acceleration penalty (aux stream, doc 35 goal C3, soft kick). 0 = off; diag always logged
        self._reward_joint_acc_w = float(env_config.get("reward_joint_acc_w", 0.0))
        self._joint_acc_scale = float(env_config.get("joint_acc_scale", 1.0e-4))
        # torque-saturation penalty (aux stream, doc 25 goal M7): mean over the joints of max(|tau|/limit - frac, 0)^2,
        # tau = the applied (clipped) drive torque. k73b on the firmware limits sits >= 95% of the hip-pitch/knee
        # limit ~30% of the steps. 0 = off; the raw value and the >=95% fraction are logged anyway.
        self._reward_torque_sat_w = float(env_config.get("reward_torque_sat_w", 0.0))
        self._torque_sat_frac = float(env_config.get("torque_sat_frac", 0.9))
        # optional joint subset for the penalty, by body name (doc 25, M7: the saturation sits on the hip pitch;
        # the mean over all 23 joints dilutes it ~11x). None = all joints (the k79b..k81 behaviour)
        self._torque_sat_dof_bodies = env_config.get("torque_sat_dof_bodies", None)
        self._arm_vel_tol = float(env_config.get("arm_vel_tol", 1.0))
        self._arm_vel_dof_bodies = list(env_config.get("arm_vel_dof_bodies", [
            "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_elbow_pitch_link", "left_elbow_yaw_link",
            "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_elbow_pitch_link", "right_elbow_yaw_link"]))
        # dense version (task stream): while a still ball is within kick_side_swing_dist of the nearest foot,
        # pay side * min(foot speed, kick_side_swing_speed) / kick_side_swing_speed, with side = |sin| of the
        # angle between the foot's horizontal long axis and its horizontal velocity (the inside of the foot
        # leads); gives gradient before any contact happens. 0 = off
        self._reward_kick_side_swing_w = float(env_config.get("reward_kick_side_swing_w", 0.0))
        self._kick_side_swing_dist = float(env_config.get("kick_side_swing_dist", 0.3))  # m
        self._kick_side_swing_speed = float(env_config.get("kick_side_swing_speed", 1.5))  # m/s
        # Table 4 variants, all 0 = bit-identical to the runs before them:
        #  - action rate split by joint group (head -15, legs -1); the whole-body
        #    reward_action_rate_w stays for the older configs and should be 0
        #    when these are on
        #  - head pitch/yaw penalty (-0.5 each) as the paper words the head term
        #  - collision on every body but the feet (-100)
        self._reward_head_action_rate_w = float(env_config.get("reward_head_action_rate_w", 0.0))
        self._reward_leg_action_rate_w = float(env_config.get("reward_leg_action_rate_w", 0.0))
        self._reward_head_pitch_w = float(env_config.get("reward_head_pitch_w", 0.0))
        self._reward_head_yaw_w = float(env_config.get("reward_head_yaw_w", 0.0))
        self._reward_collision_w = float(env_config.get("reward_collision_w", 0.0))
        self._collision_force_thresh = float(env_config.get("collision_force_thresh", 1.0))
        self._head_dof_bodies = list(env_config.get("head_dof_bodies", ["aahead_yaw_link", "aahead_pitch_link"]))
        self._leg_dof_bodies = list(env_config.get(
            "leg_dof_bodies", ["{}_{}_link".format(side, joint)
                               for side in ("left", "right")
                               for joint in ("hip_pitch", "hip_roll", "hip_yaw",
                                             "knee_pitch", "ankle_pitch", "ankle_roll")]))
        self._collision_exempt_bodies = list(env_config.get(
            "collision_exempt_bodies", ["left_ankle_roll_link", "right_ankle_roll_link"]))

        self._stagnation_window = float(env_config.get("stagnation_window", 1.0))
        self._stagnation_move_threshold = float(env_config.get("stagnation_move_threshold", 0.1))
        self._ball_contact_dist = float(env_config.get("ball_contact_dist", 0.25))
        self._foot_min_dist = float(env_config.get("foot_min_dist", 0.2))
        self._kick_feet_bodies = env_config.get(
            "kick_feet_bodies", ["left_ankle_roll_link", "right_ankle_roll_link"])

        # opt-in viewer extras (default off: stock MimicKit behavior). Purely
        # cosmetic — no physics, obs or reward impact.
        self._visualize_field = bool(env_config.get("visualize_field", False))
        self._visualize_debug_arrows = bool(env_config.get("visualize_debug_arrows", False))

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)

        if (self._motion_kick_w_early is not None):
            assert self._motion_kick_w_end >= 0.0
            lib = self._motion_lib
            is_kick = torch.tensor(["t1_kick" in f or os.path.basename(f).startswith("kick") for f in lib._motion_files], device=lib._motion_weights.device)
            assert is_kick.any() and (~is_kick).any(), "motion_kick_weight_early needs kick and non-kick clips"
            self._motion_weights_true = lib._motion_weights.clone()
            w = lib._motion_weights.clone()
            k = float(self._motion_kick_w_early)
            w[is_kick] = w[is_kick] / w[is_kick].sum() * k
            w[~is_kick] = w[~is_kick] / w[~is_kick].sum() * (1.0 - k)
            self._motion_weights_early = w
            lib._motion_weights = w

        if (self._visualize and self._visualize_field):
            self._build_field_visuals()
        return

    def _build_envs(self, config, num_envs):
        self._ball_id = None
        super()._build_envs(config, num_envs)
        return

    def _build_env(self, env_id, config):
        if (self._build_field_offsets is not None):
            # temporarily shift the char spawn to this env's field so actors
            # are created spread out (restored right after)
            offset = self._build_field_offsets[env_id]
            orig_root_pos = self._init_root_pos
            self._init_root_pos = orig_root_pos.clone()
            self._init_root_pos[0] += float(offset[0])
            self._init_root_pos[1] += float(offset[1])
            self._init_root_pos[2] += self._ground_z(env_id)
            try:
                super()._build_env(env_id, config)
            finally:
                self._init_root_pos = orig_root_pos
        else:
            super()._build_env(env_id, config)

        ball_id = self._build_ball(env_id)
        if (env_id == 0):
            self._ball_id = ball_id
        else:
            assert(ball_id == self._ball_id)

        if (self._engine.supports_build_time_props()):
            self._randomize_env_props(env_id, ball_id)
        return

    def _get_dr_rng(self):
        if (not hasattr(self, "_dr_rng")):
            seed = self._dr_seed if self._dr_seed is not None else np.random.randint(2**31 - 1)
            self._dr_rng = np.random.RandomState(int(seed))
        return self._dr_rng

    def _apply_dr_runtime(self):
        """The half of Table 2 that needs the post-initialize_sim tensors:
        motor gains, motor bias and the action delay."""
        char_id = self._get_char_id()
        self._dr_torso_params = torch.as_tensor(
            np.stack([p["torso_params"] for p in self._dr_samples]),
            device=self._device, dtype=torch.float)
        if (self._dr_runtime_start_samples >= 0.0 and self._dr_samples):
            # gated: motor gains, bias and delay switch on later (_update_schedule_switches)
            self._motor_bias = torch.zeros([len(self._dr_samples), len(self._dr_samples[0]["motor_bias"])],
                                           device=self._device, dtype=torch.float)
            self._dr_runtime_pending = True
            return
        self._apply_dr_motors()
        return

    def _apply_ball_props(self, env_id, ball_id, friction, restitution, mass_scale):
        self._engine.set_obj_shape_props(env_id, ball_id, friction=friction, restitution=restitution)
        num_bodies = self._engine.get_obj_num_bodies(ball_id)
        self._engine.scale_obj_masses(env_id, ball_id, np.full([num_bodies], mass_scale))

    def _apply_dr_static(self, env_id, params):
        char_id = self._get_char_id()
        self._engine.scale_obj_masses(env_id, char_id, params["mass_scales"], params["com_offsets"])
        foot_ids = [self._engine.find_obj_body_id(char_id, n) for n in self._dr_foot_bodies]
        if ("foot_props_per_foot" in params):
            for foot_id, (fr, co, re) in zip(foot_ids, params["foot_props_per_foot"]):
                self._engine.set_obj_shape_props(env_id, char_id, friction=fr, restitution=re,
                                                 compliance=co, body_ids=[foot_id])
        else:
            self._engine.set_obj_shape_props(env_id, char_id,
                                             friction=params["foot_friction"],
                                             restitution=params["foot_restitution"],
                                             compliance=params["foot_compliance"],
                                             body_ids=foot_ids)

    def _apply_dr_motors(self):
        char_id = self._get_char_id()
        for env_id, params in enumerate(self._dr_samples):
            self._engine.scale_obj_pd_gains(env_id, char_id, params["kp_scale"],
                                            params["kd_scale"])
        self._motor_bias = torch.as_tensor(
            np.stack([p["motor_bias"] for p in self._dr_samples]),
            device=self._device, dtype=torch.float)
        control_dt = self._engine.get_timestep()
        num_substeps = self._engine.get_num_sim_steps()
        weights = np.stack([
            steering_dr.substep_action_blend(p["action_delay_ms"], num_substeps, control_dt)
            for p in self._dr_samples])
        self._engine.set_action_delay(weights)
        return

    _STAGE_PARAM_KEYS = {"episode_length", "stand_start_prob", "reward_ball_chase_w", "reward_ball_align_w",
                         "ball_curriculum_far_prob", "ball_goal_strip_prob", "ball_goal_core_prob",
                         "nav_cmd_prob", "nav_switch_prob", "nav_cmd_scale",
                         "reward_nav_lin_w", "reward_nav_yaw_w",
                         "drib_cmd_prob", "drib_dir_max_deg", "drib_far_prob", "drib_vel_sigma",
                         "reward_drib_vel_w", "reward_drib_prog_w", "reward_drib_chase_w", "reward_drib_dist_w",
                         "reward_drib_turn_w", "reward_drib_yaw_w",
                         "cow_ctrl2_past", "cow_prob", "cow_ang_deg", "cow_goal_blend", "cow_opp_speed", "cow_ball_decel",
                         "reward_cow_ctrl_w", "reward_cow_chase_w", "reward_cow_chase2_w", "reward_cow_ctrl2_w", "reward_cow_touch_w", "reward_cow_idle_w",
                         "reward_cow_rec_w", "reward_cow_goal_w", "reward_cow_succ_w", "reward_cow_fail_w", "reward_cow_fall_w",
                         "cow_obs_noise_pos", "cow_obs_noise_vel", "cow_vis_dropout_p"}

    def _ground_z(self, env_ids, keepdim=False):
        """Spawn lift of the ground under env_ids (None = all envs): the scalar _ground_z_offset unless
        ground.tile_levels gives every field its own amplitude."""
        if (self._ground_z_offset_env is None):
            return self._ground_z_offset
        z = self._ground_z_offset_env if (env_ids is None) else self._ground_z_offset_env[env_ids]
        return z.unsqueeze(-1) if keepdim else z

    def _switch_on(self, start_samples):
        """True when a clock switch is open (negative start = always on)."""
        return start_samples < 0.0 or self._get_schedule_samples() >= start_samples

    def _schedule_progress(self, start_samples, end_samples):
        if (start_samples < 0.0):
            return 1.0
        return soccer_util.compute_curriculum_progress(
            self._get_schedule_samples(), start_samples, end_samples)

    def _stagnation_w_now(self):
        return self._reward_stagnation_w if self._switch_on(self._reward_stagnation_start_samples) else 0.0

    def _update_schedule_switches(self):
        if (self._dr_runtime_pending and self._switch_on(self._dr_runtime_start_samples)):
            self._dr_runtime_pending = False
            self._apply_dr_motors()
            for env_id in self._dr_static_pending:
                self._apply_dr_static(env_id, self._dr_samples[env_id])
            self._dr_static_pending = []
            for args in self._ball_props_pending:
                self._apply_ball_props(*args)
            self._ball_props_pending = []
            print("[GATE] DR runtime (motor gains/bias/delay) ON at clock {:.0f}".format(
                self._get_schedule_samples()))
        if (self._motion_weights_true is not None):
            q = self._schedule_progress(self._motion_kick_w_end, self._motion_kick_w_end + max(self._motion_kick_w_ramp, 1.0))
            if (q >= 1.0):
                self._motion_lib._motion_weights = self._motion_weights_true
                self._motion_weights_true = None
                print("[GATE] motion weights back to the dataset's own at clock {:.0f}".format(
                    self._get_schedule_samples()))
            else:
                w = (1.0 - q) * self._motion_weights_early + q * self._motion_weights_true
                self._motion_lib._motion_weights = w / w.sum()
        while (self._stage_params_applied < len(self._stage_params)
               and self._get_schedule_samples() >= float(self._stage_params[self._stage_params_applied]["at"])):
            d = self._stage_params[self._stage_params_applied]
            for k, v in d.items():
                if (k != "at"):
                    setattr(self, "_" + k, float(v))
            self._stage_params_applied += 1
            print("[GATE] stage params {} at clock {:.0f}".format(
                {k: v for k, v in d.items() if k != "at"}, self._get_schedule_samples()))
        if (self._episode_length_late is not None
                and self._switch_on(self._episode_length_late_start_samples)):
            self._episode_length = float(self._episode_length_late)
        return

    def _apply_action(self, actions):
        if (self._action_filter_alpha < 1.0):
            a = self._action_filter_alpha
            filt = torch.where(self._filt_fresh.unsqueeze(-1), actions,
                               a * actions + (1.0 - a) * self._filt_action)
            if (self._action_filter_kick_only):
                nonkick = self._nonkick_mode()
                if (self._action_filter_kick_ramp_s > 0.0):
                    ramp_steps = max(1.0, self._action_filter_kick_ramp_s / self._engine.get_timestep())
                    self._filt_kick_steps[:] = torch.where(
                        torch.logical_or(nonkick, self._filt_fresh), 0, self._filt_kick_steps + 1)
                    r = torch.clamp(1.0 - self._filt_kick_steps.float() / ramp_steps, 0.0, 1.0)
                    # an episode born in kick mode keeps the plain filter (no ramp)
                    r = torch.where(self._filt_kick_born, 0.0, r)
                    ae = (a + (1.0 - a) * r).unsqueeze(-1)
                    filt = torch.where(self._filt_fresh.unsqueeze(-1), actions,
                                       ae * actions + (1.0 - ae) * self._filt_action)
                    self._filt_kick_born[:] = torch.logical_and(
                        self._filt_kick_born | self._filt_fresh, torch.logical_not(nonkick))
                filt = torch.where(nonkick.unsqueeze(-1), actions, filt)
            self._filt_action[:] = filt
            self._filt_fresh[:] = False
            actions = filt
        if (not self._dr_enabled):
            super()._apply_action(actions)
            return
        # motor bias lands on the target AFTER the clip (see the steering env)
        char_id = self._get_char_id()
        clip_action = torch.minimum(torch.maximum(actions, self._action_bound_low),
                                    self._action_bound_high)
        self._engine.set_cmd(char_id, clip_action + self._motor_bias)
        return

    def _randomize_env_props(self, env_id, ball_id):
        """Per-env static randomization, applied at build time (before the sim
        is initialized, which is required by Isaac Gym's GPU pipeline)."""
        if (self._rand_ball_props):
            friction = np.random.uniform(self._rand_ball_friction[0], self._rand_ball_friction[1])
            restitution = np.random.uniform(self._rand_ball_restitution[0],
                                            self._rand_ball_restitution[1])
            self._ball_friction_samples.append(friction)
            mass_scale = np.random.uniform(self._rand_ball_mass_scale[0],
                                           self._rand_ball_mass_scale[1])
            if (self._dr_static_deferred):
                self._ball_props_pending.append((env_id, ball_id, friction, restitution, mass_scale))
            else:
                self._apply_ball_props(env_id, ball_id, friction, restitution, mass_scale)

        if (self._dr_enabled):
            char_id = self._get_char_id()
            num_bodies = self._engine.get_obj_num_bodies(char_id)
            num_dofs = self._engine.get_obj_num_dofs(char_id)
            params = steering_dr.sample_env_params(self._get_dr_rng(), self._dr_ranges,
                                                   num_bodies, num_dofs)
            if (self._dr_foot_props_per_foot):
                # one draw per foot, Booster order (friction, compliance, restitution), after the env draw
                rng = self._get_dr_rng()
                params["foot_props_per_foot"] = [
                    (float(rng.uniform(*self._dr_ranges["foot_friction"])),
                     float(rng.uniform(*self._dr_ranges["foot_compliance"])),
                     float(rng.uniform(*self._dr_ranges["foot_restitution"])))
                    for _ in self._dr_foot_bodies]
            self._dr_samples.append(params)
            if (self._dr_static_deferred):
                # gated run: the draw happens now (same RNG order), the sim props change when the DR gate opens
                self._dr_static_pending.append(env_id)
            else:
                self._apply_dr_static(env_id, params)

        if (self._rand_char_props):
            char_id = self._get_char_id()
            friction = np.random.uniform(self._rand_char_friction[0], self._rand_char_friction[1])
            self._engine.set_obj_shape_props(env_id, char_id, friction=friction)

            num_bodies = self._engine.get_obj_num_bodies(char_id)
            mass_scales = np.random.uniform(self._rand_char_other_mass_scale[0],
                                            self._rand_char_other_mass_scale[1],
                                            size=num_bodies)
            mass_scales[0] = np.random.uniform(self._rand_char_base_mass_scale[0],
                                               self._rand_char_base_mass_scale[1])
            com_offsets = np.zeros([num_bodies, 3])
            com_offsets[0] = np.random.uniform(self._rand_char_base_com[0],
                                               self._rand_char_base_com[1], size=3)
            self._engine.scale_obj_masses(env_id, char_id, mass_scales, com_offsets)
        return

    def _build_ball(self, env_id):
        ball_asset_file = "data/assets/objects/soccer_ball.xml"
        start_pos = np.array([2.0, 0.0, self._ball_radius + float(self._ground_z(env_id))],
                             dtype=np.float32)
        if (self._build_field_offsets is not None):
            start_pos[0:2] += self._build_field_offsets[env_id]
        start_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        ball_id = self._engine.create_obj(env_id=env_id,
                                          obj_type=engine.ObjType.rigid,
                                          asset_file=ball_asset_file,
                                          name="ball",
                                          start_pos=start_pos,
                                          start_rot=start_rot,
                                          color=[0.9, 0.9, 0.9])
        return ball_id

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)

        num_envs = self.get_num_envs()

        # per-env field centers on a grid (world frame); the same helper
        # sizes the uneven-ground mesh, so layout and terrain cannot diverge
        offsets = soccer_util.compute_field_offsets(num_envs, self._field_length,
                                                    self._field_width, self._field_sep)
        self._field_offset = torch.tensor(offsets, device=self._device, dtype=torch.float)

        self._goal_pos = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._goal_pos[:, 0] = 0.5 * self._field_length
        self._goal_pos += self._field_offset
        self._goal_dir = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._goal_dir[:, 0] = -1.0
        # kick target T (docs/t1/43): what the kick obs/rewards aim at. Equal to the goal centre unless
        # kick_tgt_prob > 0 draws a point on the field per episode (goal scoring keeps using _goal_pos).
        self._kick_tgt = self._goal_pos.clone()
        self._kick_tgt_on = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._kick_pow_on = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._kick_pow_p = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._kick_pow_n = torch.full([num_envs], -1, device=self._device, dtype=torch.long)
        self._kick_pow_peak = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._kick_pow_derr = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._kick_pow_paid = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._kick_pow_paid_t = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._kick_pow_b0 = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)

        self._prev_root_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._prev_ball_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)

        self._task_reward_buf = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._goal_scored_buf = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._ball_oob_buf = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        # envs whose episode ended by a ball event (goal/out): the reset
        # keeps the robot state and only repositions the ball (paper 4.1)
        self._soft_done_buf = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._ball_perturb_times = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._char_push_times = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        # global control-step counter driving the steering anneal (samples
        # seen by the agent ~= steps * num_envs)
        self._total_env_steps = 0

        # c4 free-return probe: reward collected on continuation steps where
        # the ball just respawned via a random event but perception has not
        # seen it yet (see _update_done)
        self._diag_free_ep_time = torch.zeros([1], device=self._device, dtype=torch.float)
        self._diag_free_ep_count = torch.zeros([1], device=self._device, dtype=torch.float)
        # goal-then-wait probe: reward on the last step before a goal (usually
        # ~0) vs the first steps of the continuation after the soft respawn;
        # if the latter catches up, the policy idles through goals
        self._diag_pre_goal_time = torch.zeros([1], device=self._device, dtype=torch.float)
        self._diag_pre_goal_count = torch.zeros([1], device=self._device, dtype=torch.float)
        self._diag_post_goal_time = torch.zeros([1], device=self._device, dtype=torch.float)
        self._diag_post_goal_count = torch.zeros([1], device=self._device, dtype=torch.float)
        self._prev_ep_free = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._prev_ball_event_step = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._post_goal_steps_left = torch.zeros([num_envs], device=self._device, dtype=torch.long)
        self._head_yaw_prev = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._nav_mode = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._nav_cmd = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._nav_cmd_time = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._nav_cmd_set_t = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._k2n_decided = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._k2n_left = torch.full([num_envs], float("inf"), device=self._device)
        self._k2n_abandon_left = torch.full([num_envs], float("inf"), device=self._device)
        self._nav_vel_filt = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._drib_mode = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._drib_cmd = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._drib_wp = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._drib_spd = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._drib_cmd_time = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._cow_mode = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._cow_opp = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._cow_opp_vel = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._cow_est = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._cow_est_vel = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._cow_vis = torch.ones([num_envs], device=self._device, dtype=torch.float)
        self._cow_phase = torch.zeros([num_envs], device=self._device, dtype=torch.long)
        self._cow_hold = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._cow_since = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._cow_min_bo = torch.full([num_envs], 10.0, device=self._device, dtype=torch.float)
        self._cow_min_ro = torch.full([num_envs], 10.0, device=self._device, dtype=torch.float)
        self._cow_axis = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._cow_goal0 = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._cow_passed = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._cow2_phase = torch.zeros([num_envs], device=self._device, dtype=torch.long)
        self._cow2_hold = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._cow2_since = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._cow2_htime = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._cow2_P = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._cow2_T = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)  # S2b: target T (P = T when v_T = 0)
        self._cow2_vT = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._cow2_tmin = torch.full([num_envs], 1.0e6, device=self._device, dtype=torch.float)
        self._cow2_tspd = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._cow2_tdone = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._cow2_ball0 = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)  # ball when P was drawn
        self._cow2_pside = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._cow2_pang = torch.zeros([num_envs], device=self._device, dtype=torch.float)  # P angle (rad) in the draw frame, eval B3 sectors
        # stage C cycle (doc 40): capture -> touch -> hit rest -> recapture (= next cycle's capture), clock from the capture;
        # -1 = no cycle started (ball not yet captured in this scenario); pend = a hit rest waits for the recapture
        self._cow2_cyc_clock = -torch.ones([num_envs], device=self._device, dtype=torch.float)
        self._cow2_cyc_pend = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._cow2_isA = torch.zeros([num_envs], device=self._device, dtype=torch.bool)  # stage-A episode (cow2_mix_a)
        self._cow2_tdur = torch.zeros([num_envs], device=self._device, dtype=torch.float)  # time in the TOUCH phase
        self._cow2_pred = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._cow2_stay_pos = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._cow2_touches = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._cow2_prev_fd = torch.full([num_envs], 10.0, device=self._device, dtype=torch.float)
        self._cow2_prev_speed = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._cow2_prev_pd = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._cow2d_prev_rd = torch.zeros([num_envs], device=self._device, dtype=torch.float)  # stage D robot-region dist
        self._cow2d_tball = torch.full([num_envs], -1.0, device=self._device)  # option C: time the ball first entered the region (-1 = not yet)
        self._cow2d_ocont = torch.zeros([num_envs], device=self._device, dtype=torch.bool)  # robot touched the opponent this scenario

        # ball motion timers (T1): drive the kick-direction decay, the
        # approach gating and the waiting penalty
        self._ball_moving_time = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._ball_still_time = torch.zeros([num_envs], device=self._device, dtype=torch.float)

        # aux-stream state
        self._aux_reward_buf = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._style_gain_buf = torch.ones([num_envs], device=self._device, dtype=torch.float)
        self._kick_near_buf = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._prev_root_vel = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._prev_dof_vel = None  # [num_envs, dof], set on the first _record_prev_states
        self._stagnation_anchor_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._stagnation_anchor_time = torch.zeros([num_envs], device=self._device, dtype=torch.float)

        # virtual-perception pipeline state (Frente E). Measurements are
        # captured at the camera frame rate and delivered ~latency later;
        # since latency (~116 ms) > frame period (~40 ms) several frames are
        # in flight at once -> ring buffer of K slots per env. "percep" is
        # what the actor currently sees (zero-order hold).
        K = 8
        self._percep_buf_slots = K
        self._percep_period = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._percep_next_capture = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._percep_buf_pos = torch.zeros([num_envs, K, 2], device=self._device, dtype=torch.float)
        self._percep_buf_valid = torch.zeros([num_envs, K], device=self._device, dtype=torch.bool)
        self._percep_buf_deliver = torch.full([num_envs, K], float("inf"),
                                              device=self._device, dtype=torch.float)
        self._percep_buf_capture = torch.zeros([num_envs, K], device=self._device, dtype=torch.float)
        self._percep_buf_head = torch.zeros([num_envs], device=self._device, dtype=torch.long)
        # optional alpha-beta filter (goal B1): fuses the delayed noisy
        # measurements at their capture time into a ball position/velocity
        # estimate and extrapolates it to the current time. None = raw
        # measurements (training default).
        ab = self._env_config_percep_ab
        self._percep_ab = None if ab is None else (float(ab[0]), float(ab[1]))
        self._percep_ab_pos = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._percep_ab_vel = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._percep_ab_t = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._percep_ab_n = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._percep_kf_P = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._percep_kf_out = torch.zeros([num_envs], device=self._device, dtype=torch.long)
        self._percep_kf_P01 = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._percep_kf_P11 = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._rt01 = None
        if (self._percep_rt01):
            assert self._virtual_perception, "percep_rt01 needs virtual_perception"
            prm = None if self._percep_rt01_params is None \
                else percep_emulator.load_params(self._percep_rt01_params)
            self._rt01 = percep_emulator.PercepEmulator(num_envs, self._device, params=prm)
            # diagnostic only (docs/t1/39): split the pose error from the ball error, default 1 = unchanged
            self._rt01.pose_mult = self._percep_rt01_pose_mult
            self._rt01.ball_mult = self._percep_rt01_ball_mult
            self._rt01.bias_correct = self._percep_rt01_bias_correct
            self._rt01_t = 0.0
            self._rt01_dpos = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
            self._rt01_dyaw = torch.zeros([num_envs], device=self._device, dtype=torch.float)
            self._rt01_scale_now = 0.0
            self._rt01_mult = torch.ones([num_envs], device=self._device, dtype=torch.float)
        self._percep_ball_pos = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._percep_ball_valid = torch.ones([num_envs], device=self._device, dtype=torch.bool)
        # evaluation hook (occlusion experiment): captures are all misses until
        # this time. -inf = never blacked out, so training is unaffected.
        self._percep_blackout_until = torch.full([num_envs], -float("inf"), device=self._device,
                                                 dtype=torch.float)

        # per-iteration diagnostics window (reward-term decomposition +
        # episode event rates); popped and reset by record_diagnostics()
        self._diag = diag_util.DiagWindow(self._device)
        self._prev_ball_touch = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        # k1: previous nearest-foot/ball distance and ball speed (potential, kick bonus) and
        # per-episode kick statistics (live diag, no reward effect)
        self._prev_foot_ball_dist = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._prev_ball_speed = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._kick_pending_r = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._kick_pending_t = torch.zeros([num_envs], device=self._device, dtype=torch.long)
        self._prev_foot_pos = None  # [N, feet, 3], [N, feet, 4], [N, feet, 3]; set on reset
        self._prev_foot_rot = None
        self._prev_foot_vel = None
        self._ep_peak_ball_speed = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._ep_peak_ball_angle = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._ep_min_foot_dist = torch.full([num_envs], 10.0, device=self._device, dtype=torch.float)
        self._prev_percep_valid = torch.ones([num_envs], device=self._device, dtype=torch.bool)
        self._percep_lost_since = torch.full([num_envs], -1.0, device=self._device,
                                             dtype=torch.float)

        # task-obs history buffer (Frente F-lite); 12 = steer 5 + soccer 6 + mask 1
        if (self._task_hist_steps > 0):
            self._task_hist_buf = torch.zeros([num_envs, self._task_hist_steps, 12],
                                              device=self._device, dtype=torch.float)
            if (self._virtual_perception):
                # privileged twin (true ball) for the asymmetric critic
                self._critic_task_hist_buf = torch.zeros(
                    [num_envs, self._task_hist_steps, 12],
                    device=self._device, dtype=torch.float)

        action_dim = self._action_space.shape[0]
        self._prev_action = torch.zeros([num_envs, action_dim], device=self._device, dtype=torch.float)
        self._filt_action = torch.zeros([num_envs, action_dim], device=self._device, dtype=torch.float)
        self._filt_fresh = torch.ones([num_envs], device=self._device, dtype=torch.bool)
        self._filt_kick_steps = torch.zeros([num_envs], device=self._device, dtype=torch.long)
        self._filt_kick_born = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._world_step = 0
        self._push_active = False
        self._push_force_b = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._push_torque_b = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._action_rate_buf = torch.zeros([num_envs], device=self._device, dtype=torch.float)

        # measurable-frame history (Frente F): 6 proprio + 3 * D + 12 task
        if (self._measurable_obs):
            self._meas_frame_dim = 6 + 3 * action_dim + 12
            self._obs_noise_std = steering_util.build_obs_noise_std(
                self._obs_noise_cfg, action_dim, 12).to(self._device)
            assert self._obs_noise_std.shape[0] == self._meas_frame_dim
            self._obs_noise_on = bool(torch.any(self._obs_noise_std > 0))
            self._meas_hist_buf = torch.zeros(
                [num_envs, self._meas_hist_steps, self._meas_frame_dim],
                device=self._device, dtype=torch.float)
            # privileged twin of the task-block columns of the history (true ball, mask 1): the
            # distillation teacher reads the same history with these columns swapped in (get_teacher_obs)
            self._meas_priv_blk_hist = torch.zeros(
                [num_envs, self._meas_hist_steps, 12], device=self._device, dtype=torch.float)

        char_id = self._get_char_id()
        dof_low, dof_high = self._engine.get_obj_dof_limits(0, char_id)
        self._dof_limits_low = torch.tensor(np.asarray(dof_low), device=self._device,
                                            dtype=torch.float)
        self._dof_limits_high = torch.tensor(np.asarray(dof_high), device=self._device,
                                             dtype=torch.float)

        self._head_act_rate_buf = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._leg_act_rate_buf = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._head_dof_ids = torch.tensor(
            steering_util.build_dof_group_ids(self._kin_char_model, self._head_dof_bodies),
            device=self._device, dtype=torch.long)
        self._leg_dof_ids = torch.tensor(
            steering_util.build_dof_group_ids(self._kin_char_model, self._leg_dof_bodies),
            device=self._device, dtype=torch.long)
        self._arm_dof_ids = torch.tensor(
            steering_util.build_dof_group_ids(self._kin_char_model, self._arm_dof_bodies),
            device=self._device, dtype=torch.long)
        self._arm_pose_target = torch.tensor(self._arm_pose_target_cfg, device=self._device,
                                             dtype=torch.float)
        self._torque_lim = torch.as_tensor(self._engine.get_obj_torque_limits(0, self._get_char_id()),
                                           device=self._device, dtype=torch.float).clamp_min(1e-3)
        self._torque_sat_dof_ids = None
        if (self._torque_sat_dof_bodies is not None):
            self._torque_sat_dof_ids = torch.tensor(
                steering_util.build_dof_group_ids(self._kin_char_model, list(self._torque_sat_dof_bodies)),
                device=self._device, dtype=torch.long)
        self._arm_vel_dof_ids = torch.tensor(
            steering_util.build_dof_group_ids(self._kin_char_model, self._arm_vel_dof_bodies),
            device=self._device, dtype=torch.long)
        self._knee_dof_ids = torch.tensor(
            steering_util.build_dof_group_ids(self._kin_char_model, self._knee_dof_bodies),
            device=self._device, dtype=torch.long)
        exempt = set(self._engine.find_obj_body_id(self._get_char_id(), n)
                     for n in self._collision_exempt_bodies)
        self._collision_body_ids = torch.tensor(
            [b for b in range(self._engine.get_obj_num_bodies(self._get_char_id()))
             if b not in exempt], device=self._device, dtype=torch.long)
        if (not self._engine.supports_build_time_props()):
            # engines whose per-env mass/shape properties need the initialized
            # sim (Isaac Lab): same per-env order, so the RNG draws are identical.
            # With a DR gate the draws happen now and the sim props change when it opens.
            self._dr_static_deferred = (self._dr_runtime_start_samples >= 0.0
                                        and (self._dr_enabled or self._rand_ball_props))
            assert not self._dr_static_deferred or self._dr_enabled, \
                "deferred static props are applied by the DR gate: needs domain_randomization"
            for env_id in range(num_envs):
                self._randomize_env_props(env_id, self._ball_id)
        if (self._dr_enabled):
            self._apply_dr_runtime()
        if (self._fixed_action_delay_ms is not None):
            assert not self._dr_enabled, "fixed_action_delay_ms is for runs without domain_randomization"
            blend = steering_dr.substep_action_blend(float(self._fixed_action_delay_ms),
                                                     self._engine.get_num_sim_steps(),
                                                     self._engine.get_timestep())
            self._engine.set_action_delay(np.tile(blend[None, :], [num_envs, 1]))

        self._foot_body_ids = self._build_body_ids_tensor(self._kick_feet_bodies)
        if (self._percep_fov_body != ""):
            self._fov_body_id = int(self._build_body_ids_tensor([self._percep_fov_body])[0].item())
        else:
            self._fov_body_id = None
        assert ((self._reward_head_pitch_w == 0.0 and self._reward_head_yaw_w == 0.0)
                or self._fov_body_id is not None), \
            "reward_head_pitch_w/yaw_w need percep_fov_body (the head frame)"
        assert (self._reward_head_gaze_w == 0.0 or self._fov_body_id is not None), \
            "reward_head_gaze_w requires percep_fov_body (the gaze is measured " \
            "about the camera body's boresight)"

        # before _build_data_buffers, which sizes the disc obs from a demo
        if (self._disc_table3):
            self._disc_foot_body_ids = [
                self._engine.find_obj_body_id(char_id, name) for name in self._disc_foot_bodies]
        if (self._critic_paper_obs or self._recon_paper_targets):
            # without randomization (evaluation overrides) the dims stay and carry the nominal values
            if (self._rand_ball_props and len(self._ball_friction_samples) == num_envs):
                fr = np.asarray(self._ball_friction_samples)
            else:
                fr = np.full([num_envs], 0.5 * sum(self._rand_ball_friction))
            self._ball_friction_obs = torch.as_tensor(
                fr, device=self._device, dtype=torch.float).unsqueeze(-1)
            if (not self._dr_enabled):
                self._dr_torso_params = torch.zeros([num_envs, 4], device=self._device, dtype=torch.float)
                self._dr_torso_params[:, 0] = 1.0
        # goal odometry error state (world frame position, yaw) and its drift heading
        self._goal_odom_pos = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._goal_odom_yaw = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._goal_odom_dir = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._goal_odom_next_redraw = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        return

    def _get_ball_id(self):
        return self._ball_id

    def _get_ball_pos(self):
        return self._engine.get_root_pos(self._get_ball_id())

    def _build_field_visuals(self):
        starts, ends, cols = soccer_util.build_field_line_segments(
            self._field_length, self._field_width, self._goal_width)
        offsets = self._field_offset.cpu().numpy()  # [N, 2]
        num_envs = self.get_num_envs()
        off3 = np.zeros([num_envs, 1, 3], dtype=np.float32)
        off3[:, 0, 0:2] = offsets
        # [N, S, 3]: field-local segments shifted onto each env's field
        self._field_line_starts = starts[np.newaxis] + off3
        self._field_line_ends = ends[np.newaxis] + off3
        self._field_line_cols = cols
        return

    def _render_scene(self):
        super()._render_scene()
        if (self._visualize_field):
            self._render_field_lines()
        if (self._visualize_debug_arrows):
            self._render_debug_arrows()
        return

    def _render_field_lines(self):
        num_envs = self.get_num_envs()
        for i in range(num_envs):
            self._engine.draw_lines(i, self._field_line_starts[i],
                                    self._field_line_ends[i],
                                    self._field_line_cols, 2.0)
        return

    def _render_debug_arrows(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        ball_pos = self._get_ball_pos()
        ball_vel = self._engine.get_root_vel(self._get_ball_id())

        # blue: ball -> goal (the direction the kick-direction reward pays)
        to_goal = self._kick_tgt - ball_pos[:, 0:2]
        to_goal = to_goal / torch.clamp_min(torch.linalg.norm(to_goal, dim=-1, keepdim=True), 1e-6)
        # red: steering command the env feeds into the policy's obs
        steer_cmd = soccer_util.compute_ball_steer_command(
            root_pos, ball_pos, self._steer_stop_dist, self._steer_speed_max)
        rolling = torch.linalg.norm(ball_vel[:, 0:2], dim=-1) >= self._ball_rolling_speed

        root_np = root_pos.cpu().numpy()
        ball_np = ball_pos.cpu().numpy()
        goal_dir_np = to_goal.cpu().numpy()
        steer_np = steer_cmd.cpu().numpy()
        rolling_np = rolling.cpu().numpy()

        blue = np.array([[0.1, 0.3, 1.0, 1.0]], dtype=np.float32)
        red = np.array([[1.0, 0.1, 0.1, 1.0]], dtype=np.float32)
        white = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        orange = np.array([[1.0, 0.6, 0.0, 1.0]], dtype=np.float32)

        num_envs = self.get_num_envs()
        for i in range(num_envs):
            # kick direction (1 m arrow from the ball)
            start = ball_np[i:i + 1].copy()
            start[0, 2] = 0.15
            end = start.copy()
            end[0, 0:2] += goal_dir_np[i]
            self._engine.draw_lines(i, start, end, blue, 2.0)

            # steering command (dir scaled by commanded speed)
            s = root_np[i:i + 1].copy()
            s[0, 2] = 0.15
            e = s.copy()
            e[0, 0:2] += steer_np[i, 0:2] * max(float(steer_np[i, 2]), 0.1)
            self._engine.draw_lines(i, s, e, red, 2.0)

            # ball state pole: white = still, orange = rolling
            p0 = ball_np[i:i + 1].copy()
            p1 = p0.copy()
            p1[0, 2] += 0.6
            col = orange if rolling_np[i] else white
            self._engine.draw_lines(i, p0, p1, col, 2.0)
        return

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._record_prev_states()
        self._total_env_steps += 1
        self._gate_clock.advance(self.get_num_envs())
        self._update_schedule_switches()

        self._action_rate_buf[:] = soccer_util.compute_action_rate_penalty(actions,
                                                                           self._prev_action)
        # split rates skip the first step of an episode: _prev_action was zeroed
        # and the action is an absolute target, so the "rate" would be |a_1|^2
        first = (self._timestep_buf == 0)
        head = soccer_util.compute_action_rate_penalty(
            actions.index_select(-1, self._head_dof_ids),
            self._prev_action.index_select(-1, self._head_dof_ids))
        leg = soccer_util.compute_action_rate_penalty(
            actions.index_select(-1, self._leg_dof_ids),
            self._prev_action.index_select(-1, self._leg_dof_ids))
        zero = torch.zeros_like(head)
        self._head_act_rate_buf[:] = torch.where(first, zero, head)
        self._leg_act_rate_buf[:] = torch.where(first, zero, leg)
        self._prev_action[:] = actions
        if (self._push_interval_s > 0.0):
            self._apply_world_push()
        return

    def _world_perturb_on(self):
        return self._switch_on(self._char_push_start_samples)

    def _apply_world_push(self):
        """The held push, base frame -> world with this step's base rotation (Booster applies it in LOCAL_SPACE).
        Re-issued every control step; one explicit zero when the push ends (Isaac Lab keeps a wrench)."""
        char_id = self._get_char_id()
        if (self._push_active):
            rot = self._engine.get_root_rot(char_id)
            f = torch_util.quat_rotate(rot, self._push_force_b) * self._push_impulse_scale
            t = torch_util.quat_rotate(rot, self._push_torque_b) * self._push_impulse_scale
            self._engine.set_body_wrench(None, char_id, 0, f, t)
        return

    def _every(self, interval_s):
        return max(1, int(round(interval_s / self._engine.get_timestep())))

    def _update_world_perturbations(self):
        """Booster Gym's post-step block: counter += 1, kick, push (Olives' task_steering_meas_env)."""
        self._world_step += 1
        # diag keys every step (the logger fixes its columns on the first iteration)
        on = self._world_perturb_on()
        kick_now = on and self._kick_interval_s > 0.0 and self._world_step % self._every(self._kick_interval_s) == 0
        self._diag.add_mean("world_kick_rate", torch.tensor(float(kick_now), device=self._device))
        self._diag.add_mean("world_push_force_n", torch.linalg.norm(self._push_force_b, dim=-1)
                            * (self._push_impulse_scale if self._push_active else 0.0))
        if (not on):
            return
        char_id = self._get_char_id()
        if (kick_now):
            vel = self._engine.get_root_vel(char_id)
            ang = self._engine.get_root_ang_vel(char_id)
            self._engine.set_root_vel(None, char_id, vel + self._kick_lin_vel_std * torch.randn_like(vel))
            self._engine.set_root_ang_vel(None, char_id, ang + self._kick_ang_vel_std * torch.randn_like(ang))
        if (self._push_interval_s > 0.0):
            k = self._world_step % self._every(self._push_interval_s)
            if (k == 0):
                self._push_force_b.normal_(0.0, self._push_force_std)
                self._push_torque_b.normal_(0.0, self._push_torque_std)
                self._push_active = True
            elif (k == self._every(self._push_duration_s) and self._push_active):
                self._push_force_b.zero_()
                self._push_torque_b.zero_()
                self._push_active = False
                char_id = self._get_char_id()
                z = torch.zeros([self.get_num_envs(), 3], device=self._device)
                self._engine.set_body_wrench(None, char_id, 0, z, z)
        return

    def _record_prev_states(self):
        char_id = self._get_char_id()
        self._prev_root_pos[:] = self._engine.get_root_pos(char_id)
        self._prev_root_vel[:] = self._engine.get_root_vel(char_id)
        self._prev_dof_vel = self._engine.get_dof_vel(char_id).clone()
        self._prev_ball_pos[:] = self._get_ball_pos()
        return

    def _compute_task_block(self, env_ids=None, privileged=False):
        """The 12-dim task block exactly as the actor sees it this step:
        steering obs (annealed), soccer obs (perceived ball when the virtual
        perception is on) and the ball detection mask. With privileged=True
        the block uses the TRUE ball state and mask 1 (asymmetric critic)."""
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        ball_pos = self._get_ball_pos()
        goal_pos = self._kick_tgt
        goal_dir = self._goal_dir
        if (self._cow2_stage == 3 and self._cow_active()):
            # stage D: the opponent replaces the goal centre in the task obs (same dims)
            d_on = (self._cow_mode & ~self._cow2_isA).unsqueeze(-1)
            goal_pos = torch.where(d_on, self._cow_opp, goal_pos)

        if (self._virtual_perception and not privileged):
            # the actor only ever sees the virtual camera's ball estimate
            # (noise + latency + dropout); z is irrelevant for the heading-
            # frame planar obs, keep the true one
            ball_pos = ball_pos.clone()
            ball_pos[:, 0:2] = self._percep_ball_pos
            ball_mask = self._percep_ball_valid.float().unsqueeze(-1)
        else:
            ball_mask = torch.ones([ball_pos.shape[0], 1], device=self._device,
                                   dtype=torch.float)

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            ball_pos = ball_pos[env_ids]
            goal_pos = goal_pos[env_ids]
            goal_dir = goal_dir[env_ids]
            ball_mask = ball_mask[env_ids]

        # steering command slots first, so the obs prefix (char + steering
        # task dims) matches the steering checkpoint layout column-for-column
        steer_cmd = soccer_util.compute_ball_steer_command(root_pos, ball_pos,
                                                           self._steer_stop_dist,
                                                           self._steer_speed_max)
        steer_obs = task_steering_env.compute_steering_observations(
            root_rot, steer_cmd[..., 0:2], steer_cmd[..., 2], steer_cmd[..., 0:2])
        # steering-crutch anneal (Frente D): fade the whole steering block to
        # zeros so the final policy matches the paper's command-free actor
        steer_scale = soccer_util.compute_anneal_scale(
            self._get_schedule_samples(),
            self._steer_anneal_start_samples, self._steer_anneal_end_samples)
        if (steer_scale < 1.0):
            steer_obs = steer_obs * steer_scale

        task_obs = soccer_util.compute_soccer_observations(root_pos, root_rot, ball_pos,
                                                           goal_pos, goal_dir)
        nav = None
        if (self._nav_active()):
            nav = self._nav_mode if env_ids is None else self._nav_mode[env_ids]
            if (nav.any()):
                cmd = self._nav_cmd if env_ids is None else self._nav_cmd[env_ids]
                nav_obs = compute_nav_cmd_obs(cmd, self._nav_face_horizon_s)
                steer_obs = torch.where(nav.unsqueeze(-1), nav_obs, steer_obs)
                # the ball is not part of the navigation task: hidden slots
                task_obs = task_obs.clone()
                task_obs[nav, 0:2] = 0.0
                ball_mask = torch.where(nav.unsqueeze(-1), torch.zeros_like(ball_mask), ball_mask)
        if (self._drib_active()):
            drib = self._drib_mode if env_ids is None else self._drib_mode[env_ids]
            if (drib.any()):
                # dribble: the command replaces the steering block, the ball stays visible
                cmd = self._drib_cmd if env_ids is None else self._drib_cmd[env_ids]
                steer_obs = torch.where(drib.unsqueeze(-1), dribble_util.compute_dribble_cmd_obs(cmd), steer_obs)
        if (self._cow_active()):
            cow = self._cow_mode if env_ids is None else self._cow_mode[env_ids]
            if (cow.any() and self._cow2_stage == 0):
                steer_obs = torch.where(cow.unsqueeze(-1), self._compute_cow_obs_block(root_pos, root_rot, env_ids, privileged),
                                        steer_obs)
        if ((self._goal_odom_noise or self._rt01 is not None) and not privileged):
            # the goal comes from the odometry pose estimate, not the true pose
            if (self._rt01 is not None):
                odom_pos = self._rt01_dpos
                odom_yaw = self._rt01_dyaw
            else:
                odom_scale = self._schedule_progress(self._goal_odom_start_samples, self._goal_odom_end_samples)
                odom_pos = self._goal_odom_pos * odom_scale
                odom_yaw = self._goal_odom_yaw * odom_scale
            if (env_ids is not None):
                odom_pos = odom_pos[env_ids]
                odom_yaw = odom_yaw[env_ids]
            est_pos = root_pos.clone()
            est_pos[:, 0:2] += odom_pos
            yaw_q = torch.zeros_like(root_rot)
            yaw_q[:, 2] = torch.sin(0.5 * odom_yaw)
            yaw_q[:, 3] = torch.cos(0.5 * odom_yaw)
            est_obs = soccer_util.compute_soccer_observations(
                est_pos, torch_util.quat_mul(yaw_q, root_rot), ball_pos, goal_pos, goal_dir)
            task_obs = torch.cat([task_obs[:, 0:2], est_obs[:, 2:6]], dim=-1)
            if (nav is not None and nav.any()):
                task_obs[nav, 0:2] = 0.0
        # ball detection mask (Table 2); real dropout when the virtual
        # perception pipeline is enabled, constant 1 otherwise
        if (self._kick_pow_enable):
            act = self._kick_pow_active()
            pw = self._kick_pow_p
            if (env_ids is not None):
                act = act[env_ids]
                pw = pw[env_ids]
            if (act.any()):
                steer_obs = torch.where(act.unsqueeze(-1), soccer_util.compute_kick_pow_obs(pw, self._kick_pow_p_slot4), steer_obs)
        return torch.cat([steer_obs, task_obs, ball_mask], dim=-1)

    def _compute_measurable_frame(self, env_ids=None):
        """One measurable frame: [proprio (6 + 3D) | task block (12)]."""
        char_id = self._get_char_id()
        root_rot = self._engine.get_root_rot(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        prev_action = self._prev_action
        if (env_ids is not None):
            root_rot = root_rot[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]
            prev_action = prev_action[env_ids]
        proprio = soccer_util.compute_proprio_frame(root_rot, root_ang_vel, dof_pos,
                                                    dof_vel, prev_action,
                                                    self._init_dof_pos)
        block = self._compute_task_block(env_ids)
        return torch.cat([proprio, block], dim=-1)

    def _compute_actor_frame(self, env_ids=None):
        """The frame the actor receives: the measurable frame, corrupted with
        sensor noise when configured. Called once per step (history roll) and
        once per reset (refill). The critic never goes through here."""
        return steering_util.apply_obs_noise(self._compute_measurable_frame(env_ids),
                                             self._obs_noise_std)

    def get_teacher_obs(self):
        """Actor-layout obs whose task-block columns carry the TRUE ball (mask 1) in every history frame:
        what a policy trained without virtual perception would see. Proprio columns keep the actor's noise."""
        assert self._measurable_obs
        hist = self._meas_hist_buf.clone()
        hist[..., -12:] = self._meas_priv_blk_hist
        return torch.cat([hist[:, -1], hist.flatten(start_dim=1)], dim=-1)

    def get_measurable_frame_dim(self):
        assert self._measurable_obs
        return self._meas_frame_dim

    def get_measurable_hist_steps(self):
        assert self._measurable_obs
        return self._meas_hist_steps

    def _compute_obs(self, env_ids=None):
        if (self._measurable_obs):
            # history is mutated once per step in _update_task (and refilled
            # on reset), NEVER here: _compute_obs is also used as a shape
            # probe by get_obs_space()
            hist = self._meas_hist_buf if env_ids is None else self._meas_hist_buf[env_ids]
            # with noise the frame was drawn ONCE, when it entered the history;
            # re-drawing here would differ from hist[-1] on every call
            frame = hist[:, -1] if self._obs_noise_on else self._compute_measurable_frame(env_ids)
            return torch.cat([frame, hist.flatten(start_dim=1)], dim=-1)
        obs = super()._compute_obs(env_ids)
        block = self._compute_task_block(env_ids)
        obs = torch.cat([obs, block], dim=-1)
        if (self._task_hist_steps > 0):
            # history is mutated once per step in _update_task (and refilled
            # on reset), NEVER here: _compute_obs is also used as a shape
            # probe by get_obs_space()
            hist = self._task_hist_buf if env_ids is None else self._task_hist_buf[env_ids]
            obs = torch.cat([obs, hist.flatten(start_dim=1)], dim=-1)
        return obs

    def _update_misc(self):
        super()._update_misc()
        self._update_task()
        return

    def _update_task(self):
        # 1. detect events with the pre-reset ball state (field-local frame)
        ball_pos = self._get_ball_pos()
        ball_pos_local = ball_pos.clone()
        ball_pos_local[:, 0:2] -= self._field_offset
        goal_pos_local = self._goal_pos - self._field_offset
        self._goal_scored_buf[:] = soccer_util.compute_goal_scored_flags(
            ball_pos_local, goal_pos_local, self._goal_dir, self._goal_width, self._ball_radius)
        self._ball_oob_buf[:] = soccer_util.compute_ball_out_flags(
            ball_pos_local, self._field_length, self._field_width,
            goal_pos_local, self._goal_dir, self._goal_width, self._ball_radius)
        # a scored ball is also outside the field; count it only as a goal
        self._ball_oob_buf &= ~self._goal_scored_buf
        # navigation episodes ignore the (parked) ball
        self._goal_scored_buf &= ~self._nonkick_mode()
        self._ball_oob_buf &= ~(self._nav_mode | (self._drib_mode & (not self._drib_oob_enable)))

        # ball motion timers (before reward caching: the decay and gating of
        # this step must see the up-to-date times)
        ball_vel = self._engine.get_root_vel(self._get_ball_id())
        rolling = torch.linalg.norm(ball_vel[:, 0:2], dim=-1) >= self._ball_rolling_speed
        dt = self._engine.get_timestep()
        self._ball_moving_time[:] = torch.where(rolling, self._ball_moving_time + dt,
                                                torch.zeros_like(self._ball_moving_time))
        self._ball_still_time[:] = torch.where(rolling, torch.zeros_like(self._ball_still_time),
                                               self._ball_still_time + dt)

        # 2. cache the rewards before any ball teleport corrupts the
        #    potentials or the contact geometry
        self._cache_task_reward(ball_pos, ball_vel, rolling)
        self._cache_aux_reward(ball_pos)
        if (self._nav_active()):
            self._cache_nav_reward()
        if (self._drib_active()):
            self._cache_drib_reward()
        if (self._cow_active()):
            if (self._cow2_stage > 0):
                self._update_cow2()
            else:
                self._update_cow()

        # 3. goal / out of bounds terminate the episode (paper 4.1); the
        #    flags become dones in _update_done and the reset keeps the
        #    robot in place (see _reset_envs)

        # 4. random ball perturbations + robot pushes
        self._update_ball_perturb()
        self._update_char_push()
        if (self._kick_interval_s > 0.0 or self._push_interval_s > 0.0):
            self._update_world_perturbations()
        self._update_goal_odom()
        if (self._virtual_perception):
            self._update_perception()
        self._post_goal_steps_left = torch.clamp(self._post_goal_steps_left - 1, min=0)
        if (self._nav_active()):
            self._update_nav_cmd()
        if (self._drib_active()):
            self._update_drib_cmd()
        if (self._task_hist_steps > 0):
            # exactly one roll per physics step, after the perception tick so
            # the newest history entry matches what the actor sees this step
            self._task_hist_buf[:, :-1] = self._task_hist_buf[:, 1:].clone()
            self._task_hist_buf[:, -1] = self._compute_task_block()
            if (self._virtual_perception):
                self._critic_task_hist_buf[:, :-1] = self._critic_task_hist_buf[:, 1:].clone()
                self._critic_task_hist_buf[:, -1] = self._compute_task_block(privileged=True)
        if (self._measurable_obs):
            self._meas_hist_buf[:, :-1] = self._meas_hist_buf[:, 1:].clone()
            self._meas_hist_buf[:, -1] = self._compute_actor_frame()
            self._meas_priv_blk_hist[:, :-1] = self._meas_priv_blk_hist[:, 1:].clone()
            self._meas_priv_blk_hist[:, -1] = self._compute_task_block(privileged=True)
        return

    def _reset_goal_odom(self, env_ids):
        n = len(env_ids)
        if (not self._goal_odom_noise or n == 0):
            return
        self._goal_odom_pos[env_ids] = self._goal_odom_init_std * torch.randn(
            n, 2, device=self._device)
        self._goal_odom_yaw[env_ids] = torch.clamp(
            self._goal_odom_yaw_init_std * torch.randn(n, device=self._device),
            -self._goal_odom_yaw_max, self._goal_odom_yaw_max)
        self._goal_odom_dir[env_ids] = 2.0 * np.pi * torch.rand(n, device=self._device)
        self._goal_odom_next_redraw[env_ids] = self._time_buf[env_ids] + self._goal_odom_redraw

    def _update_goal_odom(self):
        """One control step of the odometry error: position drifts at
        goal_odom_drift m/s along a heading redrawn every goal_odom_redraw s and
        is held inside goal_odom_max; yaw is a bounded random walk."""
        if (not self._goal_odom_noise):
            return
        dt = self._engine.get_timestep()
        redraw = self._time_buf >= self._goal_odom_next_redraw
        if (torch.any(redraw)):
            ids = redraw.nonzero(as_tuple=False).flatten()
            self._goal_odom_dir[ids] = 2.0 * np.pi * torch.rand(len(ids), device=self._device)
            self._goal_odom_next_redraw[ids] = self._time_buf[ids] + self._goal_odom_redraw
        step = self._goal_odom_drift * dt
        self._goal_odom_pos[:, 0] += step * torch.cos(self._goal_odom_dir)
        self._goal_odom_pos[:, 1] += step * torch.sin(self._goal_odom_dir)
        norm = torch.linalg.norm(self._goal_odom_pos, dim=-1, keepdim=True)
        self._goal_odom_pos *= torch.clamp(self._goal_odom_max / torch.clamp(norm, min=1e-6), max=1.0)
        self._goal_odom_yaw += self._goal_odom_yaw_drift * np.sqrt(dt) * torch.randn(
            self.get_num_envs(), device=self._device)
        self._goal_odom_yaw.clamp_(-self._goal_odom_yaw_max, self._goal_odom_yaw_max)

    # ------------------------------------------------ Table 3 discriminator

    def _build_table3_disc_obs(self, root_pos, root_rot, root_vel, root_ang_vel,
                               joint_rot, dof_vel, body_pos):
        dof_pos = self._kin_char_model.rot_to_dof(joint_rot)
        foot_pos = body_pos[..., self._disc_foot_body_ids, :]
        return steering_disc.compute_disc_obs(
            root_pos=root_pos, root_rot=root_rot, root_vel=root_vel,
            root_ang_vel=root_ang_vel, dof_pos=dof_pos, dof_vel=dof_vel,
            foot_pos=foot_pos, init_dof_pos=self._init_dof_pos)

    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        if (not self._disc_table3):
            return super()._compute_disc_obs_demo(motion_ids, motion_times0)
        data = self._fetch_disc_demo_data(motion_ids, motion_times0)
        return self._build_table3_disc_obs(*data)

    def _update_disc_obs(self, env_ids=None):
        if (not self._disc_table3):
            super()._update_disc_obs(env_ids)
            return
        tensors = [self._disc_hist_root_pos.get_all(),
                   self._disc_hist_root_rot.get_all(),
                   self._disc_hist_root_vel.get_all(),
                   self._disc_hist_root_ang_vel.get_all(),
                   self._disc_hist_joint_rot.get_all(),
                   self._disc_hist_dof_vel.get_all(),
                   self._disc_hist_body_pos.get_all()]
        if (env_ids is not None):
            tensors = [t[env_ids] for t in tensors]
        disc_obs = self._build_table3_disc_obs(*tensors)
        if (env_ids is None):
            self._disc_obs_buf[:] = disc_obs
        else:
            self._disc_obs_buf[env_ids] = disc_obs

    def _update_perception(self):
        """One tick of the virtual camera pipeline (Frente E).

        Deliver first (measurements captured ~latency ago become what the
        actor sees; if several are due, the most recent wins), then capture
        (a new noisy measurement enters the ring at the sampled frame rate)."""
        if (self._rt01 is not None):
            self._update_rt01()
            return
        # deliver: pick, per env, the due slot with the latest deliver time
        due = self._percep_buf_deliver <= self._time_buf.unsqueeze(-1)  # [N, K]
        any_due = due.any(dim=-1)
        if (any_due.any()):
            deliver_times = torch.where(due, self._percep_buf_deliver,
                                        torch.full_like(self._percep_buf_deliver, -float("inf")))
            latest = deliver_times.argmax(dim=-1)  # [N]
            env_ids = any_due.nonzero(as_tuple=False).flatten()
            slots = latest[env_ids]
            valid = self._percep_buf_valid[env_ids, slots]
            upd = env_ids[valid]
            meas = self._percep_buf_pos[env_ids, slots][valid]
            if (self._percep_kf_q > 0.0):
                self._kf_update(upd, meas, self._percep_buf_capture[env_ids, slots][valid])
            elif (self._percep_ab is None):
                self._percep_ball_pos[upd] = meas
            else:
                alpha, beta = self._percep_ab
                t_meas = self._percep_buf_capture[env_ids, slots][valid]
                dt = torch.clamp(t_meas - self._percep_ab_t[upd], min=1e-3).unsqueeze(-1)
                pred = self._percep_ab_pos[upd] + self._percep_ab_vel[upd] * dt
                r = meas - pred
                if (self._percep_ab_expand):
                    alpha = torch.clamp(1.0 / (self._percep_ab_n[upd] + 1.0), min=alpha).unsqueeze(-1)
                    self._percep_ab_n[upd] += 1.0
                self._percep_ab_pos[upd] = pred + alpha * r
                vel = self._percep_ab_vel[upd] + (beta / dt) * r
                speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
                self._percep_ab_vel[upd] = vel * torch.clamp(15.0 / (speed + 1e-6), max=1.0)
                self._percep_ab_t[upd] = t_meas
            self._percep_ball_valid[env_ids] = valid
            self._percep_buf_deliver[due] = float("inf")
        if (self._percep_kf_q > 0.0 and not self._percep_kf_cv):
            self._percep_ball_pos[:] = self._percep_ab_pos
        elif (self._percep_ab is not None or self._percep_kf_cv):
            ahead = torch.clamp(self._time_buf - self._percep_ab_t, min=0.0, max=0.5).unsqueeze(-1)
            self._percep_ball_pos[:] = self._percep_ab_pos + self._percep_ab_vel * ahead

        # capture
        capture = self._time_buf >= self._percep_next_capture
        if (capture.any()):
            # perception curriculum (A): interpolate easy -> configured camera
            p = soccer_util.compute_curriculum_progress(
                self._get_schedule_samples(),
                self._percep_curriculum_start_samples, self._percep_curriculum_end_samples)
            fov_deg = 360.0 + p * (self._percep_fov_deg - 360.0)
            detect_prob = 1.0 + p * (self._percep_detect_prob - 1.0)
            latency_mean = p * self._percep_latency_mean
            latency_std = p * self._percep_latency_std
            noise_dist_coef = p * self._percep_noise_dist_coef
            noise_base = p * self._percep_noise_base

            env_ids = capture.nonzero(as_tuple=False).flatten()
            char_id = self._get_char_id()
            root_pos = self._engine.get_root_pos(char_id)[env_ids]
            root_rot = self._engine.get_root_rot(char_id)[env_ids]
            ball_pos = self._get_ball_pos()[env_ids]

            dist = torch.linalg.norm(ball_pos[:, 0:2] - root_pos[:, 0:2], dim=-1)
            if (self._fov_body_id is not None):
                head_pos = self._engine.get_body_pos(char_id)[env_ids, self._fov_body_id]
                head_rot = self._engine.get_body_rot(char_id)[env_ids, self._fov_body_id]
                in_fov = soccer_util.compute_ball_in_fov_body(
                    head_pos, head_rot, ball_pos,
                    0.5 * fov_deg * np.pi / 180.0)
            else:
                in_fov = soccer_util.compute_ball_in_fov(
                    root_pos, root_rot, ball_pos,
                    0.5 * fov_deg * np.pi / 180.0)
            detect_prob_t = soccer_util.compute_ball_detection_prob(
                dist, in_fov, detect_prob,
                self._percep_detect_full_range, self._percep_detect_decay_range)
            detected = torch.rand_like(dist) < detect_prob_t
            detected = detected & (self._time_buf[env_ids] >= self._percep_blackout_until[env_ids])

            noise_std = soccer_util.compute_perception_noise_std(
                dist, noise_dist_coef, noise_base)
            noisy_pos = ball_pos[:, 0:2] + noise_std.unsqueeze(-1) * torch.randn_like(ball_pos[:, 0:2])

            latency = latency_mean \
                + latency_std * torch.randn_like(dist)
            latency = torch.clamp(latency, min=0.0)

            head = self._percep_buf_head[env_ids]
            self._percep_buf_pos[env_ids, head] = noisy_pos
            self._percep_buf_valid[env_ids, head] = detected
            self._percep_buf_deliver[env_ids, head] = self._time_buf[env_ids] + latency
            self._percep_buf_capture[env_ids, head] = self._time_buf[env_ids]
            self._percep_buf_head[env_ids] = (head + 1) % self._percep_buf_slots
            self._percep_next_capture[env_ids] = self._percep_next_capture[env_ids] \
                + self._percep_period[env_ids]

        # diagnostics: perception availability + reacquisition time
        d = self._diag
        d.add_mean("ball_visible_frac", self._percep_ball_valid.float())
        self._percep_lost_since, reacq_sum, reacq_count = diag_util.update_reacquisition(
            self._prev_percep_valid, self._percep_ball_valid,
            self._percep_lost_since, self._time_buf)
        d.add_sum("reacq_time", reacq_sum)
        d.add_sum("reacq_count", reacq_count)
        self._prev_percep_valid[:] = self._percep_ball_valid
        return

    def _update_rt01(self):
        """RT-01 emulator tick: estimated pose + deploy-style ball (held, 10 Hz, holes) + mask. The actor's ball
        slot is rebuilt as a fake world point whose heading-frame coordinates equal the emulator's robot-frame
        ball; goal slots use the estimated pose via _rt01_dpos/_rt01_dyaw."""
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        ball_pos = self._get_ball_pos()
        yaw = torch_util.calc_heading(root_rot)
        vel = self._engine.get_root_vel(char_id)[:, 0:2]
        p = soccer_util.compute_curriculum_progress(
            self._get_schedule_samples(),
            self._percep_curriculum_start_samples, self._percep_curriculum_end_samples)
        scale = p * self._percep_rt01_scale
        self._rt01_scale_now = scale
        if (self._percep_rt01_scale_mix is not None):
            scale = scale * self._rt01_mult
        fov_deg = 360.0 + p * (self._percep_fov_deg - 360.0)
        if (self._fov_body_id is not None):
            head_pos = self._engine.get_body_pos(char_id)[:, self._fov_body_id]
            head_rot = self._engine.get_body_rot(char_id)[:, self._fov_body_id]
            in_fov = soccer_util.compute_ball_in_fov_body(head_pos, head_rot, ball_pos,
                                                         0.5 * fov_deg * np.pi / 180.0)
        else:
            in_fov = soccer_util.compute_ball_in_fov(root_pos, root_rot, ball_pos,
                                                    0.5 * fov_deg * np.pi / 180.0)
        self._rt01_t += self._rt01.dt
        o = self._rt01.step(self._rt01_t, root_pos[:, 0:2], yaw, vel, ball_pos[:, 0:2], in_fov, scale)
        mask = o["mask"] > 0.5
        ball_rel = o["ball_rel"]
        if (self._percep_rt01_filter is not None and float(self._rt01_scale_now) > 0.0):
            alpha, beta = float(self._percep_rt01_filter[0]), float(self._percep_rt01_filter[1])
            det = o["detected"]
            t = self._rt01_t
            held = self._rt01.held_field
            if (det.any()):
                ids = det.nonzero(as_tuple=False).flatten()
                first = self._percep_ab_n[ids] == 0
                dt = torch.clamp(t - self._percep_ab_t[ids], min=1e-3).unsqueeze(-1)
                pred = self._percep_ab_pos[ids] + self._percep_ab_vel[ids] * dt
                r = held[ids] - pred
                pos = pred + alpha * r
                vel_e = self._percep_ab_vel[ids] + (beta / dt) * r
                sp = torch.linalg.norm(vel_e, dim=-1, keepdim=True)
                vel_e = vel_e * torch.clamp(15.0 / (sp + 1e-6), max=1.0)
                f = first.unsqueeze(-1)
                self._percep_ab_pos[ids] = torch.where(f, held[ids], pos)
                self._percep_ab_vel[ids] = torch.where(f, torch.zeros_like(vel_e), vel_e)
                self._percep_ab_t[ids] = t
                self._percep_ab_n[ids] += 1.0
            ahead = torch.clamp(t - self._percep_ab_t, min=0.0, max=0.5).unsqueeze(-1)
            ball_rel = self._rt01.to_robot_frame(self._percep_ab_pos + self._percep_ab_vel * ahead)
        elif (self._percep_kf_q > 0.0 and float(self._rt01_scale_now) > 0.0):
            # Kalman filter on the held field-frame detections (gate + reset on kick); R from the emulator's
            # noise table at the measured distance (from the ESTIMATED pose, as the deploy can compute it)
            det = o["detected"]
            t = self._rt01_t
            if (det.any()):
                ids = det.nonzero(as_tuple=False).flatten()
                meas = self._rt01.held_field[ids]
                dist = torch.linalg.norm(meas - o["est_pos"][ids], dim=-1)
                sc_ids = scale[ids] if torch.is_tensor(scale) and scale.dim() > 0 else scale
                R = self._rt01.meas_var(dist, sc_ids)
                self._kf_update(ids, meas, torch.full_like(dist, t), R=R)
            ahead = torch.clamp(t - self._percep_ab_t, min=0.0, max=0.5).unsqueeze(-1)
            est = self._percep_ab_pos + (self._percep_ab_vel * ahead if self._percep_kf_cv else 0.0)
            ball_rel = self._rt01.to_robot_frame(est)
        # before the first detection held_field is 0 (world origin, ~100 m away in Lab): show the actor zeros
        never = (self._rt01.last_det_t < -1e8).unsqueeze(-1)
        ball_rel = torch.where(never, torch.zeros_like(ball_rel), ball_rel)
        mask = mask & (self._time_buf >= self._percep_blackout_until)
        c, s = torch.cos(yaw), torch.sin(yaw)
        self._percep_ball_pos[:] = root_pos[:, 0:2] + torch.stack(
            [c * ball_rel[:, 0] - s * ball_rel[:, 1], s * ball_rel[:, 0] + c * ball_rel[:, 1]], dim=-1)
        self._percep_ball_valid[:] = mask
        self._rt01_dpos[:] = o["est_pos"] - root_pos[:, 0:2]
        self._rt01_dyaw[:] = torch.atan2(torch.sin(o["est_yaw"] - yaw), torch.cos(o["est_yaw"] - yaw))
        d = self._diag
        d.add_mean("ball_visible_frac", self._percep_ball_valid.float())
        d.add_mean("rt01_pose_err_m", torch.linalg.norm(self._rt01_dpos, dim=-1))
        d.add_mean("rt01_yaw_err_deg", torch.abs(self._rt01_dyaw) * 180.0 / np.pi)
        d.add_mean("rt01_ball_err_m", torch.linalg.norm(
            self._percep_ball_pos - ball_pos[:, 0:2], dim=-1))
        d.add_mean("rt01_scale", torch.full_like(yaw, 1.0) * scale)
        self._percep_lost_since, reacq_sum, reacq_count = diag_util.update_reacquisition(
            self._prev_percep_valid, self._percep_ball_valid,
            self._percep_lost_since, self._time_buf)
        d.add_sum("reacq_time", reacq_sum)
        d.add_sum("reacq_count", reacq_count)
        self._prev_percep_valid[:] = self._percep_ball_valid
        return

    def set_percep_blackout(self, env_ids, duration_s):
        """Hide the ball from the actor for `duration_s` from now (evaluation
        only). Drops what is already in flight too, so the actor sees the
        mask go to 0 immediately instead of after the camera latency."""
        self._percep_blackout_until[env_ids] = self._time_buf[env_ids] + float(duration_s)
        self._percep_buf_valid[env_ids] = False
        self._percep_ball_valid[env_ids] = False
        return

    def _kf_update(self, ids, meas, t_meas, R=None):
        if (R is None):
            p = soccer_util.compute_curriculum_progress(
                self._get_schedule_samples(),
                self._percep_curriculum_start_samples, self._percep_curriculum_end_samples)
            root_pos = self._engine.get_root_pos(self._get_char_id())[ids]
            dist = torch.linalg.norm(meas - root_pos[:, 0:2], dim=-1)
            R = soccer_util.compute_perception_noise_std(
                dist, p * self._percep_noise_dist_coef, p * self._percep_noise_base) ** 2 + 1e-6
        dt = torch.clamp(t_meas - self._percep_ab_t[ids], min=0.0)
        if (self._percep_kf_cv):
            self._kf_cv_update(ids, meas, t_meas, R, dt)
            return
        P = self._percep_kf_P[ids] + self._percep_kf_q * dt
        r = meas - self._percep_ab_pos[ids]
        S = P + R
        first = self._percep_ab_n[ids] == 0
        if (self._percep_kf_gate > 0.0):
            out = (r.square().sum(-1) / S) > self._percep_kf_gate ** 2
            n_out = torch.where(out, self._percep_kf_out[ids] + 1, torch.zeros_like(self._percep_kf_out[ids]))
            reset = torch.logical_or(first, n_out >= 2)
            skip = torch.logical_and(out, ~reset)
            self._percep_kf_out[ids] = torch.where(reset, torch.zeros_like(n_out), n_out)
        else:
            reset = first
            skip = torch.zeros_like(first)
        K = torch.where(reset, torch.ones_like(P), torch.where(skip, torch.zeros_like(P), P / S))
        self._percep_ab_pos[ids] = self._percep_ab_pos[ids] + K.unsqueeze(-1) * r
        self._percep_kf_P[ids] = torch.where(reset, R, (1.0 - K) * P)
        self._percep_ab_t[ids] = t_meas
        self._percep_ab_n[ids] += 1.0
        return

    def _kf_cv_update(self, ids, meas, t_meas, R, dt):
        # isotropic R -> one 2x2 covariance [[P, P01], [P01, P11]] per env, shared by x and y
        q = self._percep_kf_q
        P00, P01, P11 = self._percep_kf_P[ids], self._percep_kf_P01[ids], self._percep_kf_P11[ids]
        P00 = P00 + 2.0 * dt * P01 + dt * dt * P11 + q * dt ** 3 / 3.0
        P01 = P01 + dt * P11 + q * dt * dt / 2.0
        P11 = P11 + q * dt
        pred = self._percep_ab_pos[ids] + self._percep_ab_vel[ids] * dt.unsqueeze(-1)
        r = meas - pred
        S = P00 + R
        first = self._percep_ab_n[ids] == 0
        if (self._percep_kf_gate > 0.0):
            out = (r.square().sum(-1) / S) > self._percep_kf_gate ** 2
            n_out = torch.where(out, self._percep_kf_out[ids] + 1, torch.zeros_like(self._percep_kf_out[ids]))
            reset = torch.logical_or(first, n_out >= 2)
            skip = torch.logical_and(out, ~reset)
            self._percep_kf_out[ids] = torch.where(reset, torch.zeros_like(n_out), n_out)
        else:
            reset = first
            skip = torch.zeros_like(first)
        K0 = torch.where(skip, torch.zeros_like(S), P00 / S)
        K1 = torch.where(skip, torch.zeros_like(S), P01 / S)
        pos = pred + K0.unsqueeze(-1) * r
        vel = self._percep_ab_vel[ids] + K1.unsqueeze(-1) * r
        speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
        vel = vel * torch.clamp(15.0 / (speed + 1e-6), max=1.0)
        n00, n01, n11 = (1.0 - K0) * P00, (1.0 - K0) * P01, P11 - K1 * P01
        rs = reset.unsqueeze(-1)
        self._percep_ab_pos[ids] = torch.where(rs, meas, pos)
        self._percep_ab_vel[ids] = torch.where(rs, torch.zeros_like(vel), vel)
        self._percep_kf_P[ids] = torch.where(reset, R, n00)
        self._percep_kf_P01[ids] = torch.where(reset, torch.zeros_like(n01), n01)
        self._percep_kf_P11[ids] = torch.where(reset, torch.full_like(n11, 4.0), n11)
        self._percep_ab_t[ids] = t_meas
        self._percep_ab_n[ids] += 1.0
        return

    def _reset_perception(self, env_ids):
        """Per-episode camera parameters + a clean first measurement (the
        true ball position, valid), so the policy never sees stale data from
        the previous episode."""
        n = len(env_ids)
        freq = self._percep_freq_mean \
            + self._percep_freq_std * torch.randn([n], device=self._device)
        freq = torch.clamp(freq, min=1.0)
        self._percep_period[env_ids] = 1.0 / freq
        # desync the first capture across envs (time_buf is 0 after reset)
        self._percep_next_capture[env_ids] = self._percep_period[env_ids] \
            * torch.rand([n], device=self._device)
        self._percep_buf_deliver[env_ids] = float("inf")
        self._percep_buf_valid[env_ids] = False
        self._percep_blackout_until[env_ids] = -float("inf")
        self._percep_buf_head[env_ids] = 0
        self._percep_ball_pos[env_ids] = self._get_ball_pos()[env_ids][:, 0:2]
        self._percep_ab_pos[env_ids] = self._percep_ball_pos[env_ids]
        self._percep_ab_vel[env_ids] = 0.0
        self._percep_ab_t[env_ids] = self._time_buf[env_ids]
        self._percep_ab_n[env_ids] = 0.0
        self._percep_kf_P[env_ids] = 0.0
        self._percep_kf_out[env_ids] = 0
        self._percep_kf_P01[env_ids] = 0.0
        self._percep_kf_P11[env_ids] = 0.0
        self._percep_ball_valid[env_ids] = True
        if (self._rt01 is not None):
            self._rt01.reset(env_ids, self._rt01_t)
            if (self._percep_rt01_scale_mix is not None):
                mix = torch.tensor(self._percep_rt01_scale_mix, device=self._device, dtype=torch.float)
                self._rt01_mult[env_ids] = mix[torch.randint(0, len(mix), (len(env_ids),), device=self._device)]
            self._rt01_dpos[env_ids] = 0.0
            self._rt01_dyaw[env_ids] = 0.0
        self._prev_percep_valid[env_ids] = True
        self._percep_lost_since[env_ids] = -1.0
        return

    def _cache_task_reward(self, ball_pos, ball_vel, rolling):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)

        approach_r = soccer_util.compute_ball_approach_reward(root_pos, self._prev_root_pos,
                                                              ball_pos, self._prev_ball_pos)
        if (self._gate_approach_when_rolling):
            # no credit for chasing a ball the robot just kicked
            approach_r = approach_r * (~rolling).float()
        progress_r = soccer_util.compute_goal_progress_reward(ball_pos, self._prev_ball_pos,
                                                              self._kick_tgt)
        chase_r = torch.zeros_like(approach_r)
        if (self._reward_ball_chase_w != 0.0):
            chase_r = soccer_util.compute_ball_chase_reward(
                root_pos, self._prev_root_pos, ball_pos, self._steer_stop_dist,
                self._steer_speed_max, self._engine.get_timestep(), self._ball_chase_vel_scale)
            if (self._gate_approach_when_rolling):
                # same gate as approach: no credit for chasing a ball just kicked
                chase_r = chase_r * (~rolling).float()
        align_r = torch.zeros_like(approach_r)
        if (self._reward_ball_align_w != 0.0):
            align_r = soccer_util.compute_ball_align_reward(
                root_pos, self._engine.get_root_rot(self._get_char_id()), ball_pos,
                self._steer_stop_dist, self._ball_align_angle_scale)
            if (self._ball_align_baseline != 0.0):
                dist_far = torch.norm(ball_pos[..., 0:2] - root_pos[..., 0:2], dim=-1) \
                    > self._steer_stop_dist
                align_r = torch.where(dist_far, align_r - self._ball_align_baseline,
                                      torch.zeros_like(align_r))
            if (self._gate_approach_when_rolling):
                align_r = align_r * (~rolling).float()
        if (self._percep_potential_freeze and self._virtual_perception):
            # no potential change while the ball is hidden: a teleport is
            # free distance otherwise, and idling on respawn spots pays
            hidden = ~self._percep_ball_valid
            approach_r = approach_r * (~hidden).float()
            progress_r = progress_r * (~hidden).float()
        dir_r = soccer_util.compute_kick_direction_reward(
            ball_pos, ball_vel, self._kick_tgt, self._kick_direction_min_vel,
            self._kick_direction_decay, self._ball_moving_time, self._kick_direction_max)
        goal_r = (self._goal_scored_buf & ~self._kick_tgt_on).float()
        position_r = torch.zeros_like(goal_r)
        if (self._reward_kick_position_w != 0.0):
            position_r = soccer_util.compute_kick_position_reward(
                root_pos, ball_pos, self._kick_tgt, self._kick_position_near_dist)

        # k1 shaping + live kick diagnostics (the foot distance is always computed)
        foot_d = self._min_foot_ball_dist(ball_pos)
        ball_speed = torch.linalg.norm(ball_vel[:, 0:2], dim=-1)
        kick_still = ~rolling
        if (self._kick_still_speed is not None):
            kick_still = ball_speed < self._kick_still_speed
        foot_ball_r = torch.clamp(self._prev_foot_ball_dist - foot_d, -0.2, 0.2) \
            * kick_still.float()
        gain = ball_speed - self._prev_ball_speed
        # the ball already left the foot at the end of the kick step: use the closer of the two steps
        near = torch.minimum(self._prev_foot_ball_dist, foot_d) < self._kick_contact_dist
        kick_event = torch.logical_and(near, gain >= self._kick_contact_min_gain)
        # a robot that fell onto the ball does not count as a kick
        kick_event = torch.logical_and(kick_event, root_pos[:, 2] > self._kick_contact_min_height)
        contact_cap = self._kick_contact_gain_cap
        impulse_cap = self._kick_impulse_cap
        over_limit = self._kick_overspeed_limit
        if (self._kick_gain_cap_per_m > 0.0):
            d_ball_goal = torch.linalg.norm(self._kick_tgt - ball_pos[:, 0:2], dim=-1)
            extra = torch.clamp(self._kick_gain_cap_per_m * d_ball_goal - contact_cap, min=0.0,
                                max=max(self._kick_gain_cap_max - contact_cap, 0.0))
            contact_cap = contact_cap + extra
            impulse_cap = impulse_cap + extra
            over_limit = over_limit + extra
        kick_contact_r = kick_event.float() * torch.minimum(gain, torch.as_tensor(contact_cap, device=gain.device))
        side = self._kick_side_factor(ball_pos, ball_vel, ball_speed)
        if (self._kick_side_foot_mix != 0.0):
            kick_contact_r = kick_contact_r * ((1.0 - self._kick_side_foot_mix)
                                               + self._kick_side_foot_mix * side)
        pitch_deg = self._nearest_foot_pitch_deg(ball_pos)
        if (self._prev_foot_pos is None):
            self._store_prev_foot_state(torch.arange(ball_pos.shape[0], device=self._device))
        chapa_score, chapa_side, prev_pitch = self._prev_foot_chapa(ball_pos)
        heading = torch_util.calc_heading(self._engine.get_root_rot(char_id))
        toe_face, toe_out, medial = self._foot_strike(self._prev_foot_pos, self._prev_foot_rot, self._prev_ball_pos,
                                                      heading)
        char_body_pos = self._engine.get_body_pos(char_id)[:, self._foot_body_ids, :]
        char_body_rot = self._engine.get_body_rot(char_id)[:, self._foot_body_ids, :]
        _, toe_out_now, medial_now = self._foot_strike(char_body_pos, char_body_rot, ball_pos, heading)
        excess = torch.clamp_min(torch.abs(pitch_deg) - self._kick_pitch_tol_deg, 0.0)
        # the foot rotates ~35 deg inside the contact step (doc 25, M13): gate on the step before it
        gate_pitch = prev_pitch if self._kick_pitch_gate_prev else pitch_deg
        excess_gate = torch.clamp_min(torch.abs(gate_pitch) - self._kick_pitch_tol_deg, 0.0)
        pitch_gate = torch.exp(-torch.square(excess_gate / self._kick_pitch_sigma_deg))
        if (self._kick_pitch_mix != 0.0):
            kick_contact_r = kick_contact_r * ((1.0 - self._kick_pitch_mix) + self._kick_pitch_mix * pitch_gate)
        aim_gate = self._kick_aim_gate(ball_pos, ball_vel, ball_speed)
        if (self._kick_aim_mix != 0.0):
            kick_contact_r = kick_contact_r * ((1.0 - self._kick_aim_mix) + self._kick_aim_mix * aim_gate)
        impulse_r = torch.zeros_like(kick_contact_r)
        if (self._reward_kick_impulse_w != 0.0):
            impulse_r = kick_event.float() * aim_gate * torch.minimum(torch.clamp_min(gain, 0.0), torch.as_tensor(impulse_cap, device=gain.device)) ** 2
        pitch_near = torch.logical_and(foot_d < self._kick_pitch_dense_dist, kick_still).float()
        pitch_dense_r = pitch_near * excess / 45.0
        chapa_r = kick_event.float() * chapa_score
        sig = self._kick_pitch_target_sigma_deg
        ptarget_r = kick_event.float() * torch.exp(-torch.square((prev_pitch - self._kick_pitch_target_deg) / sig))
        ptarget_dense_r = pitch_near * torch.exp(-torch.square((pitch_deg - self._kick_pitch_target_deg) / sig))
        tsig = self._kick_toe_out_sigma_deg
        toe_face_r = kick_event.float() * toe_face
        toe_out_r = kick_event.float() * torch.exp(-torch.square((toe_out - self._kick_toe_out_target_deg) / tsig))
        toe_out_dense_r = pitch_near * torch.exp(-torch.square((toe_out_now - self._kick_toe_out_target_deg) / tsig))
        medial_r = kick_event.float() * medial
        medial_dense_r = pitch_near * medial_now
        if (self._kick_style_aim_mix != 0.0):
            style_gate = (1.0 - self._kick_style_aim_mix) + self._kick_style_aim_mix * aim_gate
            ptarget_r = ptarget_r * style_gate
            toe_out_r = toe_out_r * style_gate
            medial_r = medial_r * style_gate
        strike_cos = torch.sqrt(torch.clamp(1.0 - chapa_side * chapa_side, min=0.0))
        strike_dir_r = kick_event.float() * torch.exp(-torch.square((strike_cos - self._kick_dir_cos_target)
                                                             / self._kick_dir_cos_sigma))
        if (self._kick_contact_hold_steps > 0):
            new = torch.logical_and(kick_event, self._kick_pending_t == 0)
            self._kick_pending_r = torch.where(new, kick_contact_r, self._kick_pending_r)
            self._kick_pending_t = torch.where(
                new, torch.full_like(self._kick_pending_t, self._kick_contact_hold_steps),
                self._kick_pending_t)
            ticking = torch.logical_and(self._kick_pending_t > 0, ~new)
            self._kick_pending_t = torch.where(ticking, self._kick_pending_t - 1, self._kick_pending_t)
            due = torch.logical_and(ticking, self._kick_pending_t == 0)
            kick_contact_r = torch.where(due, self._kick_pending_r, torch.zeros_like(kick_contact_r))
            self._kick_pending_r = torch.where(due, torch.zeros_like(self._kick_pending_r), self._kick_pending_r)
        overspeed_r = torch.zeros_like(kick_contact_r)
        if (self._reward_kick_overspeed_w != 0.0):
            over = ball_speed if self._kick_overspeed_use_speed else gain
            overspeed_r = near.float() * torch.clamp(torch.clamp_min(over - over_limit, 0.0), max=10.0)
            if (self._kick_overspeed_goal_exempt > 0.0):
                d_goal = torch.linalg.norm(self._kick_tgt - ball_pos[:, 0:2], dim=-1)
                overspeed_r = overspeed_r * (d_goal >= self._kick_overspeed_goal_exempt).float()
        swing_r = torch.zeros_like(kick_contact_r)
        if (self._reward_kick_side_swing_w != 0.0):
            fvel = self._engine.get_body_vel(self._get_char_id())[:, self._foot_body_ids, :]
            _, nearest = self._nearest_foot_side(ball_pos, ball_vel[:, 0:2], ball_speed)
            fv = fvel[torch.arange(nearest.shape[0], device=nearest.device), nearest][:, 0:2]
            fspeed = torch.linalg.norm(fv, dim=-1)
            fside, _ = self._nearest_foot_side(ball_pos, fv, fspeed)
            swing_r = (foot_d < self._kick_side_swing_dist).float() * kick_still.float() * fside \
                * torch.clamp(fspeed / self._kick_side_swing_speed, max=1.0)
        pow_speed_r = torch.zeros_like(kick_contact_r)
        pow_dir_r = torch.zeros_like(kick_contact_r)
        pow_retouch_r = torch.zeros_like(kick_contact_r)
        pow_act = None
        if (self._kick_pow_enable):
            pow_act = self._kick_pow_active()
            start = pow_act & kick_event & (self._kick_pow_n < 0) & ~self._kick_pow_paid
            # the aim is the ball->target direction at the first contact
            self._kick_pow_b0[:] = torch.where(start.unsqueeze(-1), self._prev_ball_pos[:, 0:2], self._kick_pow_b0)
            (self._kick_pow_n, self._kick_pow_peak, self._kick_pow_derr, self._kick_pow_paid, self._kick_pow_paid_t,
             pow_speed_r, pow_dir_r, pow_retouch_r, pow_pay, pow_err) = soccer_util.step_kick_pow(
                pow_act, kick_event, ball_speed, ball_vel[:, 0:2], self._kick_tgt - self._kick_pow_b0,
                self._kick_pow_p, self._kick_pow_n, self._kick_pow_peak, self._kick_pow_derr, self._kick_pow_paid,
                self._kick_pow_paid_t, self._engine.get_timestep(), self._kick_pow_window_steps,
                self._kick_pow_vmin, self._kick_pow_vmax, self._kick_pow_sigma_a, self._kick_pow_sigma_b,
                self._kick_pow_dir_sigma_deg, self._kick_pow_retouch_s,
                self._kick_pow_tail_w, self._kick_pow_tail_s)
            # power episodes: the strength and goal terms are off (the speed term replaces them)
            keep = (~pow_act).float()
            kick_contact_r = kick_contact_r * keep
            impulse_r = impulse_r * keep
            overspeed_r = overspeed_r * keep
            progress_r = progress_r * keep
            d = self._diag
            d.add_mean("kick_pow_frac", pow_act.float())
            d.add_sum("kick_pow_pay", pow_pay.float())
            d.add_sum("kick_pow_abserr", torch.abs(pow_err) * pow_pay.float())
            d.add_sum("kick_pow_err", pow_err * pow_pay.float())  # signed (peak - v0), kpf2
            d.add_sum("kick_pow_ok", (pow_pay & (torch.abs(pow_err) <= 0.5)).float())
            d.add_sum("kick_pow_derr", self._kick_pow_derr * pow_pay.float())
            d.add_sum("kick_pow_retouch_ev", pow_retouch_r)
            v0 = soccer_util.kick_pow_target_speed(self._kick_pow_p, self._kick_pow_vmin, self._kick_pow_vmax)
            okp = pow_pay & (torch.abs(pow_err) <= 0.5)
            for b, (lo_v, hi_v) in enumerate(((0.0, 2.0), (2.0, 4.0), (4.0, 1.0e9))):
                inb = (v0 >= lo_v) & (v0 < hi_v)
                d.add_sum("kick_pow_pay_b%d" % b, (pow_pay & inb).float())
                d.add_sum("kick_pow_ok_b%d" % b, (okp & inb).float())
            if (self._kick_pow_adapt and (not self._kick_pow_adapt_test_only
                                          or self._mode == base_env.EnvMode.TEST)):
                front = pow_pay & (self._kick_pow_p >= self._kick_pow_adapt_lo) \
                    & (self._kick_pow_p < self._kick_pow_adapt_lo + self._kick_pow_adapt_win)
                if (not hasattr(self, "_kick_pow_adapt_acc")):
                    self._kick_pow_adapt_acc = torch.zeros(3, device=self._device)
                self._kick_pow_adapt_acc += torch.stack([front.float().sum(), (front & okp).float().sum(),
                                                         (pow_err * front.float()).sum()])
                self._kick_pow_adapt_tick += 1
                if (self._kick_pow_adapt_tick % self._kick_pow_adapt_every == 0):
                    acc = self._kick_pow_adapt_acc.tolist()
                    self._kick_pow_adapt_acc.zero_()
                    if (self._kick_pow_adapt_bias_tol >= 0.0):
                        (self._kick_pow_adapt_lo, self._kick_pow_adapt_n_ev, self._kick_pow_adapt_s_err, _) = \
                            soccer_util.update_kick_pow_adapt_bias(
                                self._kick_pow_adapt_lo, self._kick_pow_adapt_n_ev + int(acc[0]),
                                self._kick_pow_adapt_s_err + float(acc[2]), self._kick_pow_adapt_min_events,
                                self._kick_pow_adapt_bias_tol, self._kick_pow_adapt_step, self._kick_pow_adapt_lo_min)
                    else:
                        (self._kick_pow_adapt_lo, self._kick_pow_adapt_n_ev, self._kick_pow_adapt_n_ok, _) = \
                            soccer_util.update_kick_pow_adapt(
                                self._kick_pow_adapt_lo, self._kick_pow_adapt_n_ev + int(acc[0]),
                                self._kick_pow_adapt_n_ok + int(acc[1]), self._kick_pow_adapt_min_events,
                                self._kick_pow_adapt_thr, self._kick_pow_adapt_step, self._kick_pow_adapt_lo_min)
            if (self._kick_pow_adapt):
                d.add_mean("kick_pow_adapt_lo", torch.full_like(v0, self._kick_pow_adapt_lo))
        self._update_kick_diag(ball_pos, ball_vel, ball_speed, foot_d)
        self._prev_foot_ball_dist[:] = foot_d
        self._prev_ball_speed[:] = ball_speed
        self._store_prev_foot_state(None)

        # on the goal step the ball crosses past the potential's minimum, so
        # the shaping terms would fire a large negative spike that cancels the
        # terminal reward; the event step pays only the goal reward
        shaping_mask = (~self._goal_scored_buf).float()
        oob_pen = self._ball_oob_buf.float()
        if (pow_act is not None):
            oob_pen = oob_pen * (~(pow_act & self._kick_pow_paid)).float()

        self._task_reward_buf[:] = shaping_mask * (self._reward_ball_approach_w * approach_r
                                                   + self._reward_ball_chase_w * chase_r
                                                   + self._reward_ball_align_w * align_r
                                                   + self._reward_goal_progress_w * progress_r
                                                   + self._reward_kick_direction_w * dir_r
                                                   + self._reward_kick_position_w * position_r
                                                   + self._reward_foot_ball_w * foot_ball_r
                                                   + self._reward_kick_contact_w * kick_contact_r
                                                   + self._reward_kick_overspeed_w * overspeed_r
                                                   + self._reward_kick_impulse_w * impulse_r
                                                   + self._reward_kick_side_swing_w * swing_r
                                                   + self._reward_kick_pitch_dense_w * pitch_dense_r
                                                   + self._reward_kick_chapa_w * chapa_r
                                                   + self._reward_kick_pitch_target_w * ptarget_r
                                                   + self._reward_kick_pitch_target_dense_w * ptarget_dense_r
                                                   + self._reward_kick_toe_face_w * toe_face_r
                                                   + self._reward_kick_toe_out_w * toe_out_r
                                                   + self._reward_kick_toe_out_dense_w * toe_out_dense_r
                                                   + self._reward_kick_dir_w * strike_dir_r
                                                   + self._reward_kick_medial_w * medial_r
                                                   + self._reward_kick_medial_dense_w * medial_dense_r) \
            + self._reward_goal_scored_w * goal_r \
            + self._reward_ball_oob_w * oob_pen
        if (pow_act is not None):
            pow_r = (self._reward_kick_pow_speed_w * pow_speed_r + self._reward_kick_pow_dir_w * pow_dir_r
                     + self._reward_kick_pow_retouch_w * pow_retouch_r)
            self._task_reward_buf += pow_r
            d = self._diag
            d.add_mean("reward_kick_pow_speed", self._reward_kick_pow_speed_w * pow_speed_r)
            d.add_mean("reward_kick_pow_dir", self._reward_kick_pow_dir_w * pow_dir_r)
            d.add_mean("reward_kick_pow_retouch", self._reward_kick_pow_retouch_w * pow_retouch_r)

        # diagnostics: weighted per-term means (goal stream decomposition)
        d = self._diag
        d.add_mean("reward_ball_oob", self._reward_ball_oob_w * oob_pen)
        d.add_mean("reward_ball_approach", shaping_mask * self._reward_ball_approach_w * approach_r)
        d.add_mean("reward_ball_chase", shaping_mask * self._reward_ball_chase_w * chase_r)
        d.add_mean("reward_ball_align", shaping_mask * self._reward_ball_align_w * align_r)
        d.add_mean("reward_goal_progress", shaping_mask * self._reward_goal_progress_w * progress_r)
        d.add_mean("reward_kick_direction", shaping_mask * self._reward_kick_direction_w * dir_r)
        d.add_mean("reward_kick_position", shaping_mask * self._reward_kick_position_w * position_r)
        d.add_mean("reward_goal_scored", self._reward_goal_scored_w * goal_r)
        d.add_mean("kick_tgt_frac", self._kick_tgt_on.float())
        d.add_mean("kick_tgt_dist", torch.linalg.norm(self._kick_tgt - ball_pos[:, 0:2], dim=-1))
        d.add_mean("reward_foot_ball", shaping_mask * self._reward_foot_ball_w * foot_ball_r)
        d.add_mean("reward_kick_contact", shaping_mask * self._reward_kick_contact_w * kick_contact_r)
        d.add_mean("reward_kick_overspeed", shaping_mask * self._reward_kick_overspeed_w * overspeed_r)
        d.add_mean("reward_kick_impulse", shaping_mask * self._reward_kick_impulse_w * impulse_r)
        d.add_mean("reward_kick_side_swing", shaping_mask * self._reward_kick_side_swing_w * swing_r)
        d.add_mean("foot_ball_dist", foot_d)
        d.add_mean("ball_speed", ball_speed)
        d.add_mean("ball_rolling_frac", rolling.float())
        if (self._kick_still_speed is not None):
            # rolling for the gates but still kickable for the kick shaping (the terrain tolerance at work)
            d.add_mean("ball_kick_still_frac", kick_still.float())
            d.add_mean("ball_rolling_kickable_frac", torch.logical_and(rolling, kick_still).float())
        if (self._ground_z_offset_env is not None):
            # mean tile bound (bumps + steps) under the envs: confirms the tile_levels mixture in the log
            d.add_mean("ground_tile_lift", self._ground_z_offset_env)
        d.add_sum("kick_events", kick_event.float())
        d.add_sum("kick_side_sum", kick_event.float() * side)
        # pitch of the nearest foot at kick events (diagnostic whether or not the gate is on)
        # per-step means (episode logs have no sums): events/step and, among them, the pitch statistics
        d.add_mean("kick_event_rate", kick_event.float())
        d.add_mean("kick_pitch_abs_deg_x_event", kick_event.float() * torch.abs(pitch_deg))
        d.add_mean("kick_pitch_ok_x_event", kick_event.float() * (torch.abs(pitch_deg) <= self._kick_pitch_tol_deg).float())
        d.add_mean("kick_pitch_gate_x_event", kick_event.float() * pitch_gate)
        d.add_mean("kick_aim_g_x_event", kick_event.float() * aim_gate)
        d.add_mean("kick_pitch_near_frac", pitch_near)
        d.add_mean("kick_pitch_dense_raw", pitch_dense_r)
        d.add_mean("reward_kick_pitch_dense", shaping_mask * self._reward_kick_pitch_dense_w * pitch_dense_r)
        d.add_mean("kick_chapa_side_x_event", kick_event.float() * chapa_side)
        d.add_mean("kick_pitch_prev_abs_x_event", kick_event.float() * torch.abs(prev_pitch))
        d.add_mean("kick_chapa_x_event", chapa_r)
        d.add_mean("reward_kick_chapa", shaping_mask * self._reward_kick_chapa_w * chapa_r)
        d.add_mean("kick_pitch_target_x_event", ptarget_r)
        d.add_mean("kick_pitch_target_dense_raw", ptarget_dense_r)
        d.add_mean("reward_kick_pitch_target", shaping_mask * self._reward_kick_pitch_target_w * ptarget_r)
        d.add_mean("reward_kick_pitch_target_dense",
                   shaping_mask * self._reward_kick_pitch_target_dense_w * ptarget_dense_r)
        toe_out_ok = torch.logical_and(toe_out >= 9.0, toe_out <= 38.0)
        pitch_ok = torch.logical_and(prev_pitch >= -30.0, prev_pitch <= 0.0)
        d.add_mean("kick_toe_face_x_event", toe_face_r)
        d.add_mean("kick_toe_out_x_event", kick_event.float() * toe_out)
        d.add_mean("kick_m13_pose_ok_x_event", kick_event.float() * torch.logical_and(toe_out_ok, pitch_ok).float())
        d.add_mean("kick_toe_out_g_x_event", toe_out_r)
        d.add_mean("kick_toe_out_dense_raw", toe_out_dense_r)
        d.add_mean("reward_kick_toe_face", shaping_mask * self._reward_kick_toe_face_w * toe_face_r)
        d.add_mean("reward_kick_toe_out", shaping_mask * self._reward_kick_toe_out_w * toe_out_r)
        d.add_mean("reward_kick_toe_out_dense", shaping_mask * self._reward_kick_toe_out_dense_w * toe_out_dense_r)
        d.add_mean("kick_dir_cos_x_event", kick_event.float() * strike_cos)
        d.add_mean("kick_dir_g_x_event", strike_dir_r)
        d.add_mean("reward_kick_dir", shaping_mask * self._reward_kick_dir_w * strike_dir_r)
        d.add_mean("kick_medial_x_event", medial_r)
        d.add_mean("kick_medial_dense_raw", medial_dense_r)
        d.add_mean("reward_kick_medial", shaping_mask * self._reward_kick_medial_w * medial_r)
        d.add_mean("reward_kick_medial_dense", shaping_mask * self._reward_kick_medial_dense_w * medial_dense_r)
        return

    def _foot_strike(self, foot_pos, foot_rot, ball_pos, heading):
        """(toe-face indicator, toe-out deg, medial) of the foot nearest the ball: the point of the foot collision box
        nearest the ball centre lies on the front (+x) face; foot yaw - root heading,
        outward positive (left foot +, right foot -), as kick_foot_profile.py measures the dataset; medial =
        max(cos(horizontal box centre -> ball, inner side axis), 0), inner side = -y on the left foot, +y on the right."""
        if (getattr(self, "_foot_out_sign", None) is None):
            self._foot_out_sign = torch.tensor([1.0 if "left" in b else -1.0 for b in self._kick_feet_bodies],
                                               device=self._device)
            self._foot_box_c = torch.tensor(self._kick_foot_box_center, device=self._device, dtype=torch.float)
            self._foot_box_h = torch.tensor(self._kick_foot_box_half, device=self._device, dtype=torch.float)
        nearest = torch.linalg.norm(foot_pos - ball_pos.unsqueeze(1), dim=-1).argmin(dim=1)
        idx = torch.arange(nearest.shape[0], device=nearest.device)
        rot, pos = foot_rot[idx, nearest], foot_pos[idx, nearest]
        r = torch_util.quat_rotate(torch_util.quat_conjugate(rot), ball_pos - pos) - self._foot_box_c
        # contact point estimate = point of the box nearest the ball centre; its face as scripts/mujoco_soccer_eval.py
        # foot_face (argmax |p_i| / half_i): the top-front edge where the T1 pokes the ball counts as toe
        p = torch.maximum(torch.minimum(r, self._foot_box_h), -self._foot_box_h)
        ax = (torch.abs(p) / self._foot_box_h).argmax(dim=-1)
        toe = torch.logical_and(ax == 0, p[:, 0] > 0.0).float()
        axis = torch_util.quat_rotate(rot, torch.tensor([1.0, 0.0, 0.0], device=rot.device).expand(rot.shape[0], 3))
        dyaw = torch.atan2(axis[:, 1], axis[:, 0]) - heading
        dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
        sign = self._foot_out_sign[nearest]
        medial = torch.clamp_min(-sign * r[:, 1] / torch.clamp(torch.linalg.norm(r[:, 0:2], dim=-1), min=1e-6), 0.0)
        return toe, torch.rad2deg(dyaw) * sign, medial

    def _store_prev_foot_state(self, env_ids):
        char_id = self._get_char_id()
        pos = self._engine.get_body_pos(char_id)[:, self._foot_body_ids, :]
        rot = self._engine.get_body_rot(char_id)[:, self._foot_body_ids, :]
        vel = self._engine.get_body_vel(char_id)[:, self._foot_body_ids, 0:3]
        if (self._prev_foot_pos is None):
            self._prev_foot_pos = pos.clone()
            self._prev_foot_rot = rot.clone()
            self._prev_foot_vel = vel.clone()
        elif (env_ids is None):
            self._prev_foot_pos[:] = pos
            self._prev_foot_rot[:] = rot
            self._prev_foot_vel[:] = vel
        else:
            self._prev_foot_pos[env_ids] = pos[env_ids]
            self._prev_foot_rot[env_ids] = rot[env_ids]
            self._prev_foot_vel[env_ids] = 0.0
        return

    def _prev_foot_chapa(self, ball_pos):
        """(chapa score, |sin| foot velocity vs foot long axis, pitch deg) of the foot nearest the ball,
        all from the previous step's foot state."""
        d = torch.linalg.norm(self._prev_foot_pos - ball_pos.unsqueeze(1), dim=-1)
        nearest = d.argmin(dim=1)
        idx = torch.arange(nearest.shape[0], device=nearest.device)
        rot = self._prev_foot_rot[idx, nearest]
        vel = self._prev_foot_vel[idx, nearest][:, 0:2]
        axis = torch_util.quat_rotate(rot, torch.tensor([1.0, 0.0, 0.0], device=rot.device).expand(rot.shape[0], 3))
        pitch = torch.rad2deg(torch.asin(torch.clamp(axis[:, 2], -1.0, 1.0)))
        ah = axis[:, 0:2] / torch.clamp(torch.linalg.norm(axis[:, 0:2], dim=-1, keepdim=True), min=1e-6)
        vh = vel / torch.clamp(torch.linalg.norm(vel, dim=-1, keepdim=True), min=1e-6)
        cos = torch.sum(ah * vh, dim=-1).abs().clamp(max=1.0)
        side = torch.sqrt(1.0 - cos * cos)
        excess = torch.clamp_min(torch.abs(pitch) - self._kick_pitch_tol_deg, 0.0)
        gate = torch.exp(-torch.square(excess / self._kick_pitch_sigma_deg))
        return side * gate, side, pitch

    def fetch_kick_disc_obs_demo(self, num_samples):
        """Disc obs of the dataset kick clips around the strike (doc 25 C(c))."""
        if (self._kick_disc_ids is None):
            lib = self._motion_lib
            ids, peaks = [], []
            for m in range(lib.get_num_motions()):
                name = os.path.basename(lib.get_motion_file(m))
                if (name in self._kick_disc_peak_frames):
                    ids.append(m)
                    peaks.append(float(self._kick_disc_peak_frames[name]) / float(lib._motion_fps[m]))
            assert len(ids) == len(self._kick_disc_peak_frames), "kick_disc_peak_frames clips missing from the dataset"
            self._kick_disc_ids = torch.tensor(ids, device=self._device, dtype=torch.long)
            self._kick_disc_peaks = torch.tensor(peaks, device=self._device, dtype=torch.float)
        k = torch.randint(0, len(self._kick_disc_ids), [num_samples], device=self._device)
        motion_ids = self._kick_disc_ids[k]
        t = self._kick_disc_peaks[k] + (torch.rand(num_samples, device=self._device)
                                        * (self._kick_disc_pre_s + self._kick_disc_post_s) - self._kick_disc_pre_s)
        t = torch.minimum(torch.clamp_min(t, 0.0), self._motion_lib.get_motion_length(motion_ids))
        return self._compute_disc_obs_demo(motion_ids, t)

    def _nearest_foot_pitch_deg(self, ball_pos):
        """Pitch (deg) of the nearest foot's long (x) axis above the horizontal; negative = toe down."""
        char_id = self._get_char_id()
        body_pos = self._engine.get_body_pos(char_id)[:, self._foot_body_ids, :]
        body_rot = self._engine.get_body_rot(char_id)[:, self._foot_body_ids, :]
        nearest = torch.linalg.norm(body_pos - ball_pos.unsqueeze(1), dim=-1).argmin(dim=1)
        idx = torch.arange(nearest.shape[0], device=nearest.device)
        rot = body_rot[idx, nearest]
        axis = torch_util.quat_rotate(rot, torch.tensor([1.0, 0.0, 0.0], device=rot.device).expand(rot.shape[0], 3))
        return torch.rad2deg(torch.asin(torch.clamp(axis[:, 2], -1.0, 1.0)))

    def _kick_aim_gate(self, ball_pos, ball_vel, ball_speed):
        """exp(-(excess / sigma)^2) of the ball launch direction outside the goal's angular half-width, in [0, 1]."""
        to_goal = self._kick_tgt - ball_pos[:, 0:2]
        d_goal = torch.clamp(torch.linalg.norm(to_goal, dim=-1), min=1e-6)
        cos = torch.sum(ball_vel[:, 0:2] * to_goal, dim=-1) / (torch.clamp(ball_speed, min=1e-6) * d_goal)
        err = torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))
        half = torch.rad2deg(torch.atan(torch.clamp_min(torch.full_like(d_goal, 0.5 * self._goal_width - self._kick_aim_margin), 0.05) / d_goal))
        excess = torch.clamp_min(err - half, 0.0)
        return torch.exp(-torch.square(excess / self._kick_aim_sigma_deg))

    def _kick_side_factor(self, ball_pos, ball_vel, ball_speed):
        """|sin| of the angle between the nearest foot's horizontal x axis and the ball velocity, in [0, 1]."""
        side, _ = self._nearest_foot_side(ball_pos, ball_vel[:, 0:2], ball_speed)
        return side

    def _nearest_foot_side(self, ball_pos, vec_xy, vec_speed):
        """(|sin| of the angle between the nearest foot's horizontal x axis and vec_xy, nearest foot index)."""
        char_id = self._get_char_id()
        body_pos = self._engine.get_body_pos(char_id)[:, self._foot_body_ids, :]
        body_rot = self._engine.get_body_rot(char_id)[:, self._foot_body_ids, :]
        nearest = torch.linalg.norm(body_pos - ball_pos.unsqueeze(1), dim=-1).argmin(dim=1)
        idx = torch.arange(nearest.shape[0], device=nearest.device)
        rot = body_rot[idx, nearest]
        axis = torch_util.quat_rotate(rot, torch.tensor([1.0, 0.0, 0.0], device=rot.device).expand(rot.shape[0], 3))
        axis = axis[:, 0:2] / torch.clamp(torch.linalg.norm(axis[:, 0:2], dim=-1, keepdim=True), min=1e-6)
        launch = vec_xy / torch.clamp(vec_speed, min=1e-6).unsqueeze(-1)
        cos = torch.sum(axis * launch, dim=-1).abs().clamp(max=1.0)
        return torch.sqrt(1.0 - cos * cos), nearest

    def _min_foot_ball_dist(self, ball_pos):
        body_pos = self._engine.get_body_pos(self._get_char_id())
        d = torch.linalg.norm(body_pos[:, self._foot_body_ids, :] - ball_pos.unsqueeze(1), dim=-1)
        return d.min(dim=1).values

    def _update_kick_diag(self, ball_pos, ball_vel, ball_speed, foot_d):
        """Per-episode peak ball speed, the ball direction at that peak (angle to the
        goal, deg) and the closest foot approach; summed at episode end."""
        to_goal = self._kick_tgt[:, 0:2] - ball_pos[:, 0:2]
        cosang = torch.sum(ball_vel[:, 0:2] * to_goal, dim=-1) / (
            torch.clamp(ball_speed, min=1e-6) * torch.clamp(torch.linalg.norm(to_goal, dim=-1), min=1e-6))
        ang = torch.rad2deg(torch.acos(torch.clamp(cosang, -1.0, 1.0)))
        new_peak = ball_speed > self._ep_peak_ball_speed
        self._ep_peak_ball_angle[:] = torch.where(new_peak, ang, self._ep_peak_ball_angle)
        self._ep_peak_ball_speed[:] = torch.where(new_peak, ball_speed, self._ep_peak_ball_speed)
        self._ep_min_foot_dist[:] = torch.minimum(self._ep_min_foot_dist,
                                                  torch.minimum(self._prev_foot_ball_dist, foot_d))
        return

    def _cache_aux_reward(self, ball_pos):
        """Environment part of the aux critic stream (Table 3). The AMP style
        reward is added by the agent; the fall termination penalty is added in
        _update_done once the done flags are known."""
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)

        aux_r = torch.full_like(self._aux_reward_buf, self._reward_survival_w)

        # stagnation: displaced less than the threshold over the whole window
        elapsed = self._time_buf - self._stagnation_anchor_time
        window_done = elapsed >= self._stagnation_window
        stagnant = soccer_util.compute_stagnation_flags(root_pos, self._stagnation_anchor_pos,
                                                        self._stagnation_move_threshold)
        stagnant = torch.logical_and(stagnant, window_done)
        kick_m = (~self._nonkick_mode()).float()
        aux_r += self._stagnation_w_now() * stagnant.float() * kick_m
        if (window_done.any()):
            self._stagnation_anchor_pos[window_done] = root_pos[window_done]
            self._stagnation_anchor_time[window_done] = self._time_buf[window_done]

        # kick shaping: sideways foot motion in ball contact is rewarded,
        # frontal (toe-poke) motion is penalized
        body_pos = self._engine.get_body_pos(char_id)
        body_vel = self._engine.get_body_vel(char_id)
        kick_side_r = torch.zeros_like(aux_r)
        kick_fwd_r = torch.zeros_like(aux_r)
        kick_align_r = torch.zeros_like(aux_r)
        touch_any = torch.zeros_like(self._prev_ball_touch)
        for i in range(len(self._foot_body_ids)):
            foot_id = self._foot_body_ids[i]
            foot_pos = body_pos[:, foot_id, :]
            foot_vel = body_vel[:, foot_id, :]
            contact = soccer_util.compute_ball_contact_flags(foot_pos, ball_pos,
                                                             self._ball_contact_dist)
            touch_any = torch.logical_or(touch_any, contact)
            kick = soccer_util.compute_kick_components(root_rot, foot_vel, contact)
            kick_side_r += self._reward_kick_sideways_w * kick[:, 0]
            kick_fwd_r += self._reward_kick_forward_w * kick[:, 1]
            if (self._reward_kick_alignment_w != 0.0):
                kick_align_r += self._reward_kick_alignment_w * \
                    soccer_util.compute_kick_alignment_reward(foot_vel, ball_pos,
                                                              self._kick_tgt, contact,
                                                              self._kick_alignment_power) * kick_m
        aux_r += kick_side_r + kick_fwd_r + kick_align_r
        foot_ball_d = torch.stack([torch.linalg.norm(body_pos[:, f, :] - ball_pos, dim=-1)
                                   for f in self._foot_body_ids], dim=-1).min(dim=-1).values
        style_near = (foot_ball_d < self._style_near_ball_dist).float()
        self._style_gain_buf[:] = 1.0 + (self._style_near_ball_gain - 1.0) * style_near
        if (self._nav_style_gain != 1.0):
            self._style_gain_buf[:] = torch.where(self._nav_mode, self._style_gain_buf * self._nav_style_gain,
                                                  self._style_gain_buf)
        if (self._drib_style_gain != 1.0):
            self._style_gain_buf[:] = torch.where(self._drib_mode, self._style_gain_buf * self._drib_style_gain,
                                                  self._style_gain_buf)
        self._kick_near_buf[:] = style_near
        self._diag.add_mean("style_near_ball_frac", style_near)

        # regularizations
        left_foot_pos = body_pos[:, self._foot_body_ids[0], :]
        right_foot_pos = body_pos[:, self._foot_body_ids[1], :]
        foot_prox = soccer_util.compute_foot_proximity_penalty(left_foot_pos, right_foot_pos,
                                                               self._foot_min_dist)
        aux_r += self._reward_foot_proximity_w * foot_prox

        # head gaze (Frente G): true ball state, like every other reward term
        if (self._reward_head_gaze_w != 0.0):
            body_rot = self._engine.get_body_rot(char_id)
            head_pos = body_pos[:, self._fov_body_id, :]
            head_rot = body_rot[:, self._fov_body_id, :]
            gaze_r = soccer_util.compute_head_gaze_reward(head_pos, head_rot, ball_pos)
            aux_r += self._reward_head_gaze_w * gaze_r * kick_m

        # c7 active search: while the ball is unperceived, reward head yaw
        # sweep rate (finite difference of the heading angle). No privileged
        # state: the bonus is earned by moving the head, which the policy
        # measures; the true ball position never enters. Paper fig. 4D/E.
        if (self._search_sweep_w != 0.0 and self._virtual_perception):
            if (self._reward_head_gaze_w == 0.0):
                body_rot = self._engine.get_body_rot(char_id)
            hidden = ~self._percep_ball_valid
            head_yaw = torch_util.calc_heading(body_rot[:, self._fov_body_id, :])
            dyaw = torch.atan2(torch.sin(head_yaw - self._head_yaw_prev),
                               torch.cos(head_yaw - self._head_yaw_prev))
            dt = self._engine.get_timestep()
            sweep = torch.clamp(torch.abs(dyaw) / dt, max=self._search_sweep_cap) \
                / self._search_sweep_cap
            aux_r += hidden.float() * (self._search_sweep_w * sweep) * kick_m
            self._head_yaw_prev[:] = head_yaw

        aux_r += self._reward_action_rate_w * self._action_rate_buf
        aux_r += self._reward_head_action_rate_w * self._head_act_rate_buf
        aux_r += self._reward_leg_action_rate_w * self._leg_act_rate_buf

        # head pitch/yaw penalty (Table 4): true ball state, like the gaze term
        if (self._reward_head_pitch_w != 0.0 or self._reward_head_yaw_w != 0.0):
            body_rot = self._engine.get_body_rot(char_id)
            yaw_err, pitch_err = soccer_util.compute_head_gaze_angles(
                body_pos[:, self._fov_body_id, :], body_rot[:, self._fov_body_id, :], ball_pos)
            aux_r += self._reward_head_pitch_w * pitch_err + self._reward_head_yaw_w * yaw_err

        collision = None
        if (self._reward_collision_w != 0.0):
            collision = steering_reward.compute_collision_penalty(
                self._engine.get_contact_forces(char_id), self._collision_body_ids,
                self._collision_force_thresh)
            aux_r += self._reward_collision_w * collision

        dof_pos = self._engine.get_dof_pos(char_id)
        joint_limit = soccer_util.compute_joint_limit_penalty(
            self._prev_action if self._joint_limit_on_action else dof_pos, self._dof_limits_low,
                                                              self._dof_limits_high)
        aux_r += self._reward_joint_limit_w * joint_limit
        if (self._reward_knee_bend_w != 0.0):
            knee_bend = soccer_util.compute_knee_bend_penalty(dof_pos, self._knee_dof_ids,
                                                              self._knee_bend_min)
            aux_r += self._reward_knee_bend_w * knee_bend
            self._diag.add_mean("reward_knee_bend", self._reward_knee_bend_w * knee_bend)
        near_scale = 1.0
        if (self._near_ball_pen_extra != 0.0):
            near_scale = soccer_util.compute_near_ball_scale(
                self._engine.get_root_pos(char_id), ball_pos,
                self._near_ball_pen_radius, self._near_ball_pen_extra)
        if (self._reward_arm_pose_w != 0.0):
            arm = near_scale * soccer_util.compute_arm_pose_penalty(
                dof_pos, self._arm_dof_ids, self._arm_pose_target, self._arm_pose_tol)
            aux_r += self._reward_arm_pose_w * arm
            self._diag.add_mean("reward_arm_pose", self._reward_arm_pose_w * arm)
        # arm speed: raw diagnostics always (smoke before the weight is raised), penalty only with the weight on
        arm_qd = self._engine.get_dof_vel(char_id).index_select(-1, self._arm_vel_dof_ids)
        arm_vel_pen = torch.mean(torch.square(torch.clamp_min(torch.abs(arm_qd) - self._arm_vel_tol, 0.0)), dim=-1)
        self._diag.add_mean("arm_speed_rms", torch.sqrt(torch.mean(torch.square(arm_qd), dim=-1)))
        self._diag.add_mean("arm_vel_pen_raw", arm_vel_pen)
        tau_ratio = torch.abs(self._engine.get_dof_forces(char_id)) / self._torque_lim
        self._diag.add_mean("torque_sat95_frac", torch.mean((tau_ratio >= 0.95).float(), dim=-1))
        if (self._torque_sat_dof_ids is not None):
            tau_ratio = tau_ratio.index_select(-1, self._torque_sat_dof_ids)
            self._diag.add_mean("torque_sat95_sel_frac", torch.mean((tau_ratio >= 0.95).float(), dim=-1))
        torque_sat_pen = torch.mean(torch.square(torch.clamp_min(tau_ratio - self._torque_sat_frac, 0.0)), dim=-1)
        self._diag.add_mean("torque_sat_pen_raw", torque_sat_pen)
        if (self._reward_torque_sat_w != 0.0):
            aux_r += self._reward_torque_sat_w * torque_sat_pen
            self._diag.add_mean("reward_torque_sat", self._reward_torque_sat_w * torque_sat_pen)
        if (self._reward_arm_vel_w != 0.0):
            aux_r += self._reward_arm_vel_w * arm_vel_pen
            self._diag.add_mean("reward_arm_vel", self._reward_arm_vel_w * arm_vel_pen)
        if (self._reward_torso_lean_w != 0.0):
            lean = near_scale * soccer_util.compute_torso_lean_penalty(
                self._engine.get_root_rot(char_id), self._torso_lean_max)
            aux_r += self._reward_torso_lean_w * lean
            self._diag.add_mean("reward_torso_lean", self._reward_torso_lean_w * lean)

        dt = self._engine.get_timestep()
        base_accel = soccer_util.compute_base_accel_penalty(root_vel, self._prev_root_vel, dt)
        aux_r += self._reward_base_accel_w * base_accel
        # joint acceleration (aux stream, doc 35 goal C3): mean over the joints of ((qd - qd_prev) / dt)^2,
        # scaled by joint_acc_scale; raw value and per-step peak logged always, penalty only with the weight on
        if (self._prev_dof_vel is not None):
            ddq = (self._engine.get_dof_vel(char_id) - self._prev_dof_vel) / dt
            joint_acc = torch.mean(torch.square(ddq), dim=-1) * self._joint_acc_scale
            self._diag.add_mean("joint_acc_pen_raw", joint_acc)
            self._diag.add_mean("joint_acc_peak", torch.amax(torch.abs(ddq), dim=-1))
            if (self._reward_joint_acc_w != 0.0):
                aux_r += self._reward_joint_acc_w * joint_acc
                self._diag.add_mean("reward_joint_acc", self._reward_joint_acc_w * joint_acc)

        # waiting penalty (T1): grows quadratically with ball-still time,
        # saturating at waiting_time_max; zero while the ball rolls
        wait_frac = torch.clamp(self._ball_still_time / self._waiting_time_max, max=1.0)
        waiting_r = self._reward_waiting_w * wait_frac * wait_frac * kick_m
        aux_r += waiting_r

        self._aux_reward_buf[:] = aux_r

        # diagnostics: weighted per-term means (env part of the aux stream;
        # the AMP style reward is logged by the agent, termination in
        # _update_done). One step() per control step lives here.
        d = self._diag
        d.step()
        if (self._search_sweep_w != 0.0 and self._virtual_perception):
            d.add_mean("reward_search_sweep", hidden.float() * (self._search_sweep_w * sweep))
        d.add_mean("reward_survival", self._reward_survival_w)
        d.add_mean("reward_stagnation", self._stagnation_w_now() * stagnant.float())
        d.add_mean("reward_kick_sideways", kick_side_r)
        d.add_mean("reward_kick_forward", kick_fwd_r)
        d.add_mean("reward_kick_alignment", kick_align_r)
        d.add_mean("reward_foot_proximity", self._reward_foot_proximity_w * foot_prox)
        if (self._reward_head_gaze_w != 0.0):
            d.add_mean("reward_head_gaze", self._reward_head_gaze_w * gaze_r)
        d.add_mean("reward_action_rate", self._reward_action_rate_w * self._action_rate_buf)
        if (self._reward_head_action_rate_w != 0.0 or self._reward_leg_action_rate_w != 0.0):
            d.add_mean("reward_head_action_rate",
                       self._reward_head_action_rate_w * self._head_act_rate_buf)
            d.add_mean("reward_leg_action_rate",
                       self._reward_leg_action_rate_w * self._leg_act_rate_buf)
        if (self._reward_head_pitch_w != 0.0 or self._reward_head_yaw_w != 0.0):
            d.add_mean("reward_head_pitch", self._reward_head_pitch_w * pitch_err)
            d.add_mean("reward_head_yaw", self._reward_head_yaw_w * yaw_err)
        if (collision is not None):
            d.add_mean("reward_collision", self._reward_collision_w * collision)
        d.add_mean("reward_joint_limit", self._reward_joint_limit_w * joint_limit)
        d.add_mean("head_dof_speed", torch.linalg.norm(
            self._engine.get_dof_vel(char_id).index_select(-1, self._head_dof_ids), dim=-1))  # rad/s, diag only (user: head moves too much)
        d.add_mean("reward_base_accel", self._reward_base_accel_w * base_accel)
        d.add_mean("reward_waiting", waiting_r)

        # ball-touch events: fraction of steps in contact + rising edges
        d.add_mean("ball_touch_frac", touch_any.float())
        new_touch = torch.logical_and(touch_any, ~self._prev_ball_touch)
        d.add_sum("touch_events", new_touch.float())
        self._prev_ball_touch[:] = touch_any
        return

    def _update_done(self):
        super()._update_done()
        # fall termination penalty joins the aux stream once dones are known;
        # gated BEFORE ball events also become FAIL (out-of-bounds ends the
        # episode but pays no fall penalty)
        fall_mask = (self._done_buf == base_env.DoneFlags.FAIL.value)
        self._aux_reward_buf += self._reward_termination_w * fall_mask.float()

        # diagnostics: termination term + per-episode event counters
        d = self._diag
        d.add_mean("reward_termination", self._reward_termination_w * fall_mask.float())
        if (self._cow2_stage > 0):
            cow_fall = torch.logical_and(fall_mask, self._cow_mode).float()
            self._task_reward_buf -= self._reward_cow2_term_w * cow_fall
            self._reward_buf -= self._reward_cow2_term_w * cow_fall  # copied from the task buf in _update_reward
            d.add_mean("cow2_term_fall", cow_fall)
            d.add_mean("reward_cow2_term", -self._reward_cow2_term_w * cow_fall)
        kick_ep = ~self._nonkick_mode()
        d.add_sum("ep_falls", torch.logical_and(fall_mask, kick_ep).float())
        if (self._nav_active()):
            nav_end = torch.logical_and(self._nav_mode, self._done_buf != base_env.DoneFlags.NULL.value)
            d.add_sum("nav_ep_done", nav_end.float())
            d.add_sum("nav_ep_falls", torch.logical_and(fall_mask, self._nav_mode).float())
        if (self._drib_active()):
            drib_end = torch.logical_and(self._drib_mode, self._done_buf != base_env.DoneFlags.NULL.value)
            d.add_sum("drib_ep_done", drib_end.float())
            d.add_sum("drib_ep_falls", torch.logical_and(fall_mask, self._drib_mode).float())
        d.add_sum("ep_goals", self._goal_scored_buf.float())
        d.add_sum("ep_oob", self._ball_oob_buf.float())
        d.add_sum("goal_time", self._time_buf * self._goal_scored_buf.float())

        done, soft = soccer_util.apply_ball_event_dones(
            self._done_buf, self._goal_scored_buf, self._ball_oob_buf,
            base_env.DoneFlags.NULL.value, base_env.DoneFlags.SUCC.value,
            base_env.DoneFlags.FAIL.value)
        if (self._kick_pow_enable and self._kick_pow_end_s >= 0.0):
            # power episode: success (soft, the robot keeps its state) once the paid kick had time to show
            pow_end = self._kick_pow_active() & self._kick_pow_paid & (self._kick_pow_paid_t >= self._kick_pow_end_s)
            pow_end = pow_end & (done == base_env.DoneFlags.NULL.value)
            done = torch.where(pow_end, torch.full_like(done, base_env.DoneFlags.SUCC.value), done)
            soft = soft | pow_end
            d.add_sum("kick_pow_ep_end", pow_end.float())
        if (self._kick_pow_enable):
            d.add_sum("kick_pow_ep_done", torch.logical_and(
                done != base_env.DoneFlags.NULL.value, self._kick_pow_active()).float())
        self._done_buf[:] = done
        self._soft_done_buf[:] = soft
        ended = torch.logical_and(done != base_env.DoneFlags.NULL.value, kick_ep)
        kicked = torch.logical_and(self._ep_min_foot_dist < self._kick_contact_dist,
                                   self._ep_peak_ball_speed > 1.0)
        d.add_sum("ep_kicked", torch.logical_and(ended, kicked).float())
        d.add_sum("ep_foot_contact", torch.logical_and(
            ended, self._ep_min_foot_dist < self._kick_contact_dist).float())
        d.add_sum("ep_peak_speed", ended.float() * self._ep_peak_ball_speed)
        d.add_sum("ep_min_foot_dist", ended.float() * self._ep_min_foot_dist)
        d.add_sum("ep_kicked_angle", torch.logical_and(ended, kicked).float() * self._ep_peak_ball_angle)
        d.add_sum("ep_kicked_within30", torch.logical_and(
            torch.logical_and(ended, kicked), self._ep_peak_ball_angle < 30.0).float())
        self._ep_peak_ball_speed[ended] = 0.0
        self._ep_peak_ball_angle[ended] = 0.0
        self._ep_min_foot_dist[ended] = 10.0
        d.add_sum("ep_done", ended.float())

        # free-return hack probe: robot sits on the respawn spot between a
        # random ball event (perturb/soft respawn, NOT a fall or episode
        # reset) and the first perception of the new position; the cycle then
        # pays approach/progress without locomotion. Only relevant while the
        # Gaussian event curriculum is active; the robot sees the result only
        # via the virtual camera (zero-order hold at the last valid fix).
        soft_now = torch.logical_and(self._soft_done_buf,
                                     self._done_buf != base_env.DoneFlags.NULL.value)
        event_now = torch.logical_or(soft_now, self._prev_ball_event_step)
        self._diag_free_ep_time += (
            self._prev_ep_free * (self._task_reward_buf + self._aux_reward_buf)).sum()
        self._diag_free_ep_count += self._prev_ep_free.float().sum()
        self._prev_ep_free = torch.logical_and(event_now, ~self._percep_ball_valid)
        self._prev_ball_event_step[:] = False

        step_r = self._task_reward_buf + self._aux_reward_buf
        self._diag_pre_goal_time += (self._goal_scored_buf.float() * step_r).sum()
        self._diag_pre_goal_count += self._goal_scored_buf.float().sum()
        pg = self._post_goal_steps_left > 0
        if (pg.any()):
            self._diag_post_goal_time += (pg.float() * step_r).sum()
            self._diag_post_goal_count += pg.float().sum()
        return

    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)
        self._info["aux_reward"] = self._aux_reward_buf
        self._info["style_gain"] = self._style_gain_buf
        self._info["kick_near"] = self._kick_near_buf
        if (self.has_critic_obs()):
            # asymmetric critic (paper Table 2): the critic trains on the
            # TRUE ball state while the actor only ever sees the perceived
            # one. Fresh tensor every call: no aliasing with later mutations.
            self._info["critic_obs"] = self._compute_critic_obs()
        if (self._measurable_obs):
            self._info["recon_tar"] = self._compute_recon_tar()
        return

    @staticmethod
    def _cow_event_diags(sums):
        """Cow episode rates from the window's event sums (doc 40 metrics V1-V6). Scenario = one gate/respawn
        cycle ended by success or failure; rates are per ended scenario."""
        g = lambda k: sums.get(k, 0.0)
        succ = g("cow_succ")
        fails = g("cow_fail_ball") + g("cow_fail_contact") + g("cow_fail_lost") + g("cow_fail_time")
        ended = succ + fails
        touches = g("cow_touch")
        gates = g("cow_gate")
        ratio = lambda a, b: a / b if b > 0.0 else 0.0
        side_pos = g("cow_side_pos_succ") + g("cow_side_pos_fail")
        side_neg = g("cow_side_neg_succ") + g("cow_side_neg_fail")
        return {
            "cow_ended": ended,
            "cow_succ_rate": ratio(succ, ended),
            "cow_gate_count": gates,
            "cow_gate_time_mean": ratio(g("cow_gate_time"), gates),
            "cow_touch_count": touches,
            "cow_pass_rate": ratio(g("cow_pass"), touches),
            "cow_fail_ball_rate": ratio(g("cow_fail_ball"), ended),
            "cow_fail_contact_rate": ratio(g("cow_fail_contact"), ended),
            "cow_fail_lost_rate": ratio(g("cow_fail_lost"), ended),
            "cow_fail_time_rate": ratio(g("cow_fail_time"), ended),
            "cow_fail_time_ctrl_rate": ratio(g("cow_fail_time_ctrl"), ended),
            "cow_side_pos_frac": ratio(side_pos, side_pos + side_neg),
            "cow_side_pos_succ_rate": ratio(g("cow_side_pos_succ"), side_pos),
            "cow_side_neg_succ_rate": ratio(g("cow_side_neg_succ"), side_neg),
            "cow_goal_gain_pos_frac": ratio(g("cow_goal_gain_pos"), succ),
            "cow_goal_gain_mean": ratio(g("cow_goal_gain"), succ),
            "cow_opp_dist_mean": ratio(g("cow_opp_dist"), g("cow_steps")),
        }

    def record_diagnostics(self):
        """Per-iteration diagnostics: windowed means of every weighted reward
        term plus episode event rates. The window covers every env step since
        the previous call (train + test rollouts of one iteration) and resets
        on read. Keys are stable across iterations; rate/time entries report
        0.0 when the window saw no matching events."""
        diags = dict(super().record_diagnostics())
        means, sums = self._diag.pop()
        diags.update(means)

        eps = sums.get("ep_done", 0.0)
        goals = sums.get("ep_goals", 0.0)
        reacq_count = sums.get("reacq_count", 0.0)
        diags["ep_goal_rate"] = goals / eps if eps > 0.0 else 0.0
        diags["ep_fall_rate"] = sums.get("ep_falls", 0.0) / eps if eps > 0.0 else 0.0
        diags["ep_oob_rate"] = sums.get("ep_oob", 0.0) / eps if eps > 0.0 else 0.0
        if (self._kick_pow_enable):
            pays = sums.get("kick_pow_pay", 0.0)
            pr = lambda a, b: a / b if b > 0.0 else 0.0
            diags["kick_pow_pay_rate"] = pr(pays, sums.get("kick_pow_ep_done", 0.0))
            diags["kick_pow_abserr"] = pr(sums.get("kick_pow_abserr", 0.0), pays)
            diags["kick_pow_err"] = pr(sums.get("kick_pow_err", 0.0), pays)
            diags["kick_pow_ok_rate"] = pr(sums.get("kick_pow_ok", 0.0), pays)
            diags["kick_pow_derr_deg"] = pr(sums.get("kick_pow_derr", 0.0), pays)
            diags["kick_pow_retouch_rate"] = pr(sums.get("kick_pow_retouch_ev", 0.0), pays)
            for b in range(3):
                diags["kick_pow_ok_rate_b%d" % b] = pr(sums.get("kick_pow_ok_b%d" % b, 0.0),
                                                       sums.get("kick_pow_pay_b%d" % b, 0.0))
        if (self._drib_active()):
            drib_eps = sums.get("drib_ep_done", 0.0)
            drib_steps = sums.get("drib_steps", 0.0)
            diags["drib_ep_fall_rate"] = sums.get("drib_ep_falls", 0.0) / drib_eps if drib_eps > 0.0 else 0.0
            for k in ("drib_vel_err", "drib_dir_err_deg", "drib_ball_dist", "drib_ball_lost"):
                diags[k] = sums.get(k, 0.0) / drib_steps if drib_steps > 0.0 else 0.0
        if (self._cow_active()):
            diags.update(self._cow_event_diags(sums))
            if (self._cow2_stage > 0):
                diags.update(self._cow2_event_diags(sums))
        if (self._nav_active()):
            nav_eps = sums.get("nav_ep_done", 0.0)
            nav_steps = sums.get("nav_steps", 0.0)
            diags["nav_ep_fall_rate"] = sums.get("nav_ep_falls", 0.0) / nav_eps if nav_eps > 0.0 else 0.0
            diags["nav_lin_err"] = sums.get("nav_lin_err", 0.0) / nav_steps if nav_steps > 0.0 else 0.0
            diags["nav_yaw_err"] = sums.get("nav_yaw_err", 0.0) / nav_steps if nav_steps > 0.0 else 0.0
            diags["nav_yaw_err_raw"] = sums.get("nav_yaw_err_raw", 0.0) / nav_steps if nav_steps > 0.0 else 0.0
        diags["ep_touches"] = sums.get("touch_events", 0.0) / eps if eps > 0.0 else 0.0
        kicked_n = sums.get("ep_kicked", 0.0)
        diags["kicked_ep_frac"] = kicked_n / eps if eps > 0.0 else 0.0
        diags["foot_contact_ep_frac"] = sums.get("ep_foot_contact", 0.0) / eps if eps > 0.0 else 0.0
        diags["ep_peak_ball_speed"] = sums.get("ep_peak_speed", 0.0) / eps if eps > 0.0 else 0.0
        diags["ep_min_foot_ball_dist"] = sums.get("ep_min_foot_dist", 0.0) / eps if eps > 0.0 else 0.0
        diags["kicked_angle_mean"] = sums.get("ep_kicked_angle", 0.0) / kicked_n if kicked_n > 0.0 else 0.0
        diags["kicked_within30_frac"] = \
            sums.get("ep_kicked_within30", 0.0) / kicked_n if kicked_n > 0.0 else 0.0
        diags["goal_time_mean"] = sums.get("goal_time", 0.0) / goals if goals > 0.0 else 0.0
        diags["ball_reacquire_time"] = \
            sums.get("reacq_time", 0.0) / reacq_count if reacq_count > 0.0 else 0.0
        diags["free_return_step_mean"] = \
            (self._diag_free_ep_time / torch.clamp(self._diag_free_ep_count, min=1.0)).item()
        diags["free_return_steps"] = self._diag_free_ep_count.item()
        diags["pre_goal_reward"] = \
            (self._diag_pre_goal_time / torch.clamp(self._diag_pre_goal_count, min=1.0)).item()
        diags["post_goal_reward"] = \
            (self._diag_post_goal_time / torch.clamp(self._diag_post_goal_count, min=1.0)).item()
        diags["post_goal_steps"] = self._diag_post_goal_count.item()
        self._diag_free_ep_time[:] = 0.0
        self._diag_free_ep_count[:] = 0.0
        self._diag_pre_goal_time[:] = 0.0
        self._diag_pre_goal_count[:] = 0.0
        self._diag_post_goal_time[:] = 0.0
        self._diag_post_goal_count[:] = 0.0

        if (self._gate_clock.gates):
            if (self._gate_clock.update(diags, eps)):
                opened = self._gate_clock.gates[self._gate_clock.idx - 1]
                print("[GATE] gate {} (at {:.0f}) opened: {}".format(
                    self._gate_clock.idx - 1, opened["at"],
                    {m: round(float(diags[m]), 3) for m, _, _ in opened["conds"]}))
            diags.update(self._gate_clock.info())
        samples = self._get_schedule_samples()
        diags["percep_curriculum"] = soccer_util.compute_curriculum_progress(
            samples, self._percep_curriculum_start_samples, self._percep_curriculum_end_samples)
        diags["ball_curriculum"] = soccer_util.compute_curriculum_progress(
            samples, self._ball_curriculum_start_samples, self._ball_curriculum_end_samples)
        diags["steer_scale"] = soccer_util.compute_anneal_scale(
            samples, self._steer_anneal_start_samples, self._steer_anneal_end_samples)
        diags["ball_behind_curriculum"] = soccer_util.compute_curriculum_progress(
            samples, self._ball_behind_start_samples, self._ball_behind_end_samples)
        return diags

    def record_train_diagnostics(self):
        """Like record_diagnostics(), but the window only spans the train
        rollout of the iteration. Called by the agent after _update_model()."""
        return self.record_diagnostics()

    def _get_schedule_samples(self):
        """Sample clock shared by the steer anneal and the curricula."""
        return self._gate_clock.clock

    def has_critic_obs(self):
        """Whether this env publishes a privileged critic_obs in the info."""
        return self._virtual_perception or self._measurable_obs

    def _compute_critic_obs(self):
        """Privileged critic obs (asymmetric critic, paper Table 2).

        measurable_obs mode: full-state char obs + privileged task block +
        true planar ball velocity (heading frame). DIFFERENT SHAPE from the
        actor obs; consumers must size the critic from get_critic_obs_space().
        Otherwise: actor-layout obs with the perceived ball slots replaced by
        the true ball state (current block + history), same shape as the obs."""
        obs = super()._compute_obs()
        block = self._compute_task_block(privileged=True)
        obs = torch.cat([obs, block], dim=-1)
        if (self._measurable_obs):
            char_id = self._get_char_id()
            ball_state = soccer_util.compute_ball_state_local(
                self._engine.get_root_pos(char_id),
                self._engine.get_root_rot(char_id),
                self._get_ball_pos(),
                self._engine.get_root_vel(self._get_ball_id()))
            blocks = [obs, ball_state[..., 2:4]]
            if ((self._dr_enabled and self._dr_critic_torso_obs) or self._critic_paper_obs):
                blocks.append(self._dr_torso_params)
            if (self._critic_paper_obs):
                # Table 3 critic: ball velocity in the world frame and the ball friction
                blocks.append(self._engine.get_root_vel(self._get_ball_id())[..., 0:2])
                blocks.append(self._ball_friction_obs)
            return torch.cat(blocks, dim=-1)
        if (self._task_hist_steps > 0):
            obs = torch.cat([obs, self._critic_task_hist_buf.flatten(start_dim=1)], dim=-1)
        return obs

    def get_critic_obs_space(self):
        obs = self._compute_critic_obs()
        return spaces.Box(low=-np.inf, high=np.inf, shape=list(obs.shape[1:]),
                          dtype=torch_util.torch_dtype_to_numpy(obs.dtype))

    def _compute_recon_tar(self):
        """Decoder target (paper Fig. 4B): true planar ball position and
        velocity in the heading frame. Training-only; never enters the actor."""
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        ball_vel = self._engine.get_root_vel(self._get_ball_id())
        ball_state = soccer_util.compute_ball_state_local(
            root_pos, root_rot, self._get_ball_pos(), ball_vel)
        if (not self._recon_paper_targets):
            return ball_state
        # Table 3 Recon. column: ball (x, y) + velocity, base linear velocity,
        # base height, base mass/CoM, ball velocity (world), ball friction. The
        # static and slow terms are scaled to ~unit range so the MSE weighs them
        # comparably; the ball terms stay first (the recency gate only applies to them)
        heading_inv = torch_util.calc_heading_quat_inv(root_rot)
        lin_vel = torch_util.quat_rotate(heading_inv, self._engine.get_root_vel(char_id))
        height = (root_pos[:, 2:3] - self._ground_z(None, keepdim=True) - 0.67) / 0.1
        if (self._dr_enabled):
            lo, hi = self._dr_ranges["torso_mass_scale"]
            clo, chi = self._dr_ranges["torso_com_m"]
            torso = self._dr_torso_params.clone()
            torso[:, 0] = (torso[:, 0] - 0.5 * (lo + hi)) / max(0.5 * (hi - lo), 1e-6)
            torso[:, 1:] = (torso[:, 1:] - 0.5 * (clo + chi)) / max(0.5 * (chi - clo), 1e-6)
        else:
            torso = torch.zeros_like(self._dr_torso_params)
        flo, fhi = self._rand_ball_friction
        friction = (self._ball_friction_obs - 0.5 * (flo + fhi)) / max(0.5 * (fhi - flo), 1e-6)
        return torch.cat([ball_state, lin_vel, height, torso, ball_vel[:, 0:2], friction], dim=-1)

    def get_recon_tar_size(self):
        assert self._measurable_obs
        return 15 if self._recon_paper_targets else 4

    def get_recon_gated_dims(self):
        """Leading recon dims that depend on seeing the ball (the recency gate
        applies only to them; the robot and static terms are always observable)."""
        return 4

    def _update_ball_perturb(self):
        if (not self._ball_perturb_enable or not self._switch_on(self._ball_perturb_start_samples)):
            return
        trigger_mask = torch.logical_and(self._time_buf >= self._ball_perturb_times, ~self._drib_mode)
        env_ids = trigger_mask.nonzero(as_tuple=False).flatten()
        n = len(env_ids)
        if (n > 0):
            ball_id = self._get_ball_id()
            teleport_mask = torch.rand(n, device=self._device) < self._ball_perturb_teleport_prob

            teleport_ids = env_ids[teleport_mask]
            if (len(teleport_ids) > 0):
                self._reset_ball(teleport_ids, near=not self._ball_soft_reset_far,
                                 gauss=self._ball_event_gauss())
                self._prev_ball_event_step[teleport_ids] = True

            push_ids = env_ids[~teleport_mask]
            m = len(push_ids)
            if (m > 0):
                theta = 2.0 * np.pi * torch.rand(m, device=self._device)
                speed = (self._ball_perturb_speed_max - self._ball_perturb_speed_min) \
                    * torch.rand(m, device=self._device) + self._ball_perturb_speed_min
                push_vel = torch.zeros([m, 3], device=self._device, dtype=torch.float)
                push_vel[:, 0] = speed * torch.cos(theta)
                push_vel[:, 1] = speed * torch.sin(theta)
                self._engine.set_root_vel(push_ids, ball_id, push_vel)

            self._resample_ball_perturb_times(env_ids)
        return

    def _resample_ball_perturb_times(self, env_ids):
        n = len(env_ids)
        rand_dt = (self._ball_perturb_time_max - self._ball_perturb_time_min) \
            * torch.rand(n, device=self._device) + self._ball_perturb_time_min
        self._ball_perturb_times[env_ids] = self._time_buf[env_ids] + rand_dt
        return

    def _update_char_push(self):
        if (not self._char_push_enable or not self._switch_on(self._char_push_start_samples)):
            return
        trigger_mask = self._time_buf >= self._char_push_times
        env_ids = trigger_mask.nonzero(as_tuple=False).flatten()
        n = len(env_ids)
        if (n > 0):
            char_id = self._get_char_id()
            vel = self._engine.get_root_vel(char_id)[env_ids].clone()
            vel[:, 0:2] += self._char_push_vel_std * torch.randn([n, 2], device=self._device)
            self._engine.set_root_vel(env_ids, char_id, vel)

            ang_vel = self._engine.get_root_ang_vel(char_id)[env_ids].clone()
            ang_vel[:, 2] += self._char_push_ang_vel_std * torch.randn([n], device=self._device)
            self._engine.set_root_ang_vel(env_ids, char_id, ang_vel)

            self._resample_char_push_times(env_ids)
        return

    def _resample_char_push_times(self, env_ids):
        n = len(env_ids)
        rand_dt = (self._char_push_time_max - self._char_push_time_min) \
            * torch.rand(n, device=self._device) + self._char_push_time_min
        self._char_push_times[env_ids] = self._time_buf[env_ids] + rand_dt
        return

    def _update_reward(self):
        self._reward_buf[:] = self._task_reward_buf
        return

    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            # ball-event dones keep the robot: split them off the full path
            soft_mask = self._soft_done_buf[env_ids]
            soft_ids = env_ids[soft_mask]
            env_ids = env_ids[~soft_mask]
            self._soft_done_buf[soft_ids] = False
            if (len(soft_ids) > 0):
                self._soft_reset_envs(soft_ids)

        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._reset_goal_odom(env_ids)
            self._reset_char_placement(env_ids)
            # the placement moved/rotated the root, so the rigid body state
            # written by char_env's reset is stale; recompute it with FK or
            # key-body obs will mix the old pose with the new root position
            self._reset_char_rigid_body_state(env_ids)
            self._reset_ball(env_ids)
            self._reset_nav_mode(env_ids)
            self._reset_drib_mode(env_ids)
            if (self._virtual_perception):
                self._reset_perception(env_ids)
            if (self._task_hist_steps > 0):
                # replicate the post-reset block: no stale data, no zero mixing
                self._task_hist_buf[env_ids] = self._compute_task_block(env_ids).unsqueeze(1)
                if (self._virtual_perception):
                    self._critic_task_hist_buf[env_ids] = \
                        self._compute_task_block(env_ids, privileged=True).unsqueeze(1)
            self._record_reset_prev_states(env_ids)
            self._task_reward_buf[env_ids] = 0.0
            self._goal_scored_buf[env_ids] = False
            self._ball_oob_buf[env_ids] = False

            self._aux_reward_buf[env_ids] = 0.0
            self._action_rate_buf[env_ids] = 0.0
            self._head_act_rate_buf[env_ids] = 0.0
            self._leg_act_rate_buf[env_ids] = 0.0
            if (self._dr_enabled or self._fixed_action_delay_ms is not None):
                self._engine.reset_action_delay(env_ids)
            self._prev_action[env_ids] = 0.0
            self._filt_fresh[env_ids] = True
            self._prev_ball_touch[env_ids] = False
            if (self._search_sweep_w != 0.0):
                # first post-reset step must not read a stale yaw from the
                # previous episode (would pay a fake sweep spike)
                char_id = self._get_char_id()
                head_rot = self._engine.get_body_rot(char_id)[env_ids, self._fov_body_id, :]
                self._head_yaw_prev[env_ids] = torch_util.calc_heading(head_rot)
            self._post_goal_steps_left[env_ids] = 0
            if (self._measurable_obs):
                # after prev_action is zeroed: the refilled history must match
                # the frame the actor sees on its first post-reset step
                self._meas_hist_buf[env_ids] = \
                    self._compute_actor_frame(env_ids).unsqueeze(1)
                self._meas_priv_blk_hist[env_ids] = \
                    self._compute_task_block(env_ids, privileged=True).unsqueeze(1)
            self._ball_moving_time[env_ids] = 0.0
            self._ball_still_time[env_ids] = 0.0
            char_id = self._get_char_id()
            self._prev_root_vel[env_ids] = self._engine.get_root_vel(char_id)[env_ids]
            self._stagnation_anchor_pos[env_ids] = self._engine.get_root_pos(char_id)[env_ids]
            self._stagnation_anchor_time[env_ids] = self._time_buf[env_ids]
            self._resample_char_push_times(env_ids)
        return

    def reset_to_spatial_benchmark(self, env_ids, ball_local_xy):
        """Reset trials to a fixed center pose and caller-provided ball cells."""
        if (env_ids.ndim != 1 or ball_local_xy.shape != (len(env_ids), 2)):
            raise ValueError("expected env_ids [N] and ball_local_xy [N, 2]")

        half_x = 0.5 * self._field_length
        half_y = 0.5 * self._field_width
        if (torch.any(torch.abs(ball_local_xy[:, 0]) >= half_x)
                or torch.any(torch.abs(ball_local_xy[:, 1]) >= half_y)):
            raise ValueError("benchmark ball positions must be strictly inside the field")

        # Goal/OOB normally use a soft reset. Every benchmark trial instead
        # starts from the same complete robot state.
        self._soft_done_buf[env_ids] = False
        if (self._nav_cmd_prob > 0.0 and self._nav_force is None):
            # the spatial benchmark is a kick benchmark: no nav draws, also
            # on the soft resets after a goal
            self._nav_force = False
        if (self._drib_cmd_prob > 0.0 and self._drib_force is None):
            self._drib_force = False
        if (self._cow_prob > 0.0 and self._cow_force is None):
            self._cow_force = False
        self._reset_envs(env_ids)
        self._nav_mode[env_ids] = False
        self._drib_mode[env_ids] = False
        self._cow_mode[env_ids] = False

        n = len(env_ids)
        char_id = self._get_char_id()
        root_pos = self._init_root_pos.unsqueeze(0).repeat(n, 1)
        if (self._stand_start_root_z is not None):
            root_pos[:, 2] = self._stand_start_root_z
        root_pos[:, 0:2] = self._field_offset[env_ids]
        root_pos[:, 2] += self._ground_z(env_ids)
        root_rot = self._init_root_rot.unsqueeze(0).repeat(n, 1)
        init_dof = self._init_dof_pos if self._stand_start_dof_pos is None else self._stand_start_dof_pos
        dof_pos = init_dof.unsqueeze(0).repeat(n, 1)
        zero_vel = torch.zeros([n, 3], device=self._device, dtype=torch.float)
        self._engine.set_root_pos(env_ids, char_id, root_pos)
        self._engine.set_root_rot(env_ids, char_id, root_rot)
        self._engine.set_root_vel(env_ids, char_id, zero_vel)
        self._engine.set_root_ang_vel(env_ids, char_id, zero_vel)
        self._engine.set_dof_pos(env_ids, char_id, dof_pos)
        self._engine.set_dof_vel(env_ids, char_id, 0.0)
        self._engine.set_body_vel(env_ids, char_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, char_id, 0.0)
        self._reset_char_rigid_body_state(env_ids)

        ball_id = self._get_ball_id()
        ball_pos = torch.zeros([n, 3], device=self._device, dtype=torch.float)
        ball_pos[:, 0:2] = ball_local_xy + self._field_offset[env_ids]
        ball_pos[:, 2] = self._ball_radius + self._ground_z(env_ids)
        ball_rot = torch.zeros([n, 4], device=self._device, dtype=torch.float)
        ball_rot[:, 3] = 1.0
        self._engine.set_root_pos(env_ids, ball_id, ball_pos)
        self._engine.set_root_rot(env_ids, ball_id, ball_rot)
        self._engine.set_root_vel(env_ids, ball_id, zero_vel)
        self._engine.set_root_ang_vel(env_ids, ball_id, zero_vel)

        self._prev_ball_pos[env_ids] = ball_pos
        self._reset_kick_prev(env_ids, ball_pos)
        self._ball_moving_time[env_ids] = 0.0
        self._ball_still_time[env_ids] = 0.0
        if (self._virtual_perception):
            self._reset_perception(env_ids)
        if (self._task_hist_steps > 0):
            self._task_hist_buf[env_ids] = self._compute_task_block(env_ids).unsqueeze(1)
            if (self._virtual_perception):
                self._critic_task_hist_buf[env_ids] = \
                    self._compute_task_block(env_ids, privileged=True).unsqueeze(1)
        if (self._measurable_obs):
            self._meas_hist_buf[env_ids] = \
                self._compute_actor_frame(env_ids).unsqueeze(1)
            self._meas_priv_blk_hist[env_ids] = \
                self._compute_task_block(env_ids, privileged=True).unsqueeze(1)
        self._record_reset_prev_states(env_ids)
        self._prev_root_vel[env_ids] = self._engine.get_root_vel(char_id)[env_ids]
        if (self._prev_dof_vel is not None):
            self._prev_dof_vel[env_ids] = self._engine.get_dof_vel(char_id)[env_ids]
        self._stagnation_anchor_pos[env_ids] = root_pos
        self._stagnation_anchor_time[env_ids] = self._time_buf[env_ids]
        self._ball_perturb_times[env_ids] = 1.0e9
        self._char_push_times[env_ids] = 1.0e9

        self._update_observations(env_ids)
        self._update_info(env_ids)
        return self._obs_buf, self._info

    def _soft_reset_envs(self, env_ids):
        """New episode after a goal / ball-out done (paper 4.1): the robot
        keeps its physical state, only the ball is repositioned and the
        episode clock and reward state restart."""
        # zero the clock FIRST: every resampled event time is time_buf-based.
        # _time_buf is recomputed from _timestep_buf each step, so both must
        # be cleared (mirrors sim_env._reset_envs without touching the char)
        self._timestep_buf[env_ids] = 0
        self._time_buf[env_ids] = 0.0
        self._done_buf[env_ids] = base_env.DoneFlags.NULL.value
        self._reset_ball(env_ids, near=not self._ball_soft_reset_far,
                         gauss=self._ball_event_gauss())
        self._reset_nav_mode(env_ids)
        self._reset_drib_mode(env_ids)
        self._post_goal_steps_left[:] = 0
        self._post_goal_steps_left[env_ids] = 16  # ~0.5 s at 30 Hz
        self._goal_odom_next_redraw[env_ids] = self._time_buf[env_ids] + self._goal_odom_redraw
        if (self._virtual_perception):
            self._reset_perception(env_ids)
        if (self._task_hist_steps > 0):
            self._task_hist_buf[env_ids] = self._compute_task_block(env_ids).unsqueeze(1)
            if (self._virtual_perception):
                self._critic_task_hist_buf[env_ids] = \
                    self._compute_task_block(env_ids, privileged=True).unsqueeze(1)
        if (self._measurable_obs):
            self._meas_hist_buf[env_ids] = \
                self._compute_actor_frame(env_ids).unsqueeze(1)
            self._meas_priv_blk_hist[env_ids] = \
                self._compute_task_block(env_ids, privileged=True).unsqueeze(1)
        self._record_reset_prev_states(env_ids)
        self._task_reward_buf[env_ids] = 0.0
        self._goal_scored_buf[env_ids] = False
        self._ball_oob_buf[env_ids] = False
        self._aux_reward_buf[env_ids] = 0.0
        char_id = self._get_char_id()
        self._prev_root_vel[env_ids] = self._engine.get_root_vel(char_id)[env_ids]
        self._stagnation_anchor_pos[env_ids] = self._engine.get_root_pos(char_id)[env_ids]
        self._stagnation_anchor_time[env_ids] = self._time_buf[env_ids]
        self._resample_char_push_times(env_ids)
        return

    def _reset_char_placement(self, env_ids):
        """Scatter the RSI-initialized robot over the field with a random
        heading. Character features (obs/disc obs) are heading-invariant, so
        this only changes the task geometry."""
        n = env_ids.shape[0]
        char_id = self._get_char_id()

        half_x = 0.5 * self._field_length - self._spawn_margin
        half_y = 0.5 * self._field_width - self._spawn_margin

        root_pos = self._engine.get_root_pos(char_id)[env_ids].clone()
        root_pos[:, 0] = half_x * (2.0 * torch.rand(n, device=self._device) - 1.0)
        root_pos[:, 1] = half_y * (2.0 * torch.rand(n, device=self._device) - 1.0)
        root_pos[:, 0:2] += self._field_offset[env_ids]
        root_pos[:, 2] += self._ground_z(env_ids)

        theta = 2.0 * np.pi * torch.rand(n, device=self._device) - np.pi
        axis = torch.zeros([n, 3], device=self._device, dtype=torch.float)
        axis[:, 2] = 1.0
        rand_rot = torch_util.axis_angle_to_quat(axis, theta)

        root_rot = self._engine.get_root_rot(char_id)[env_ids]
        root_vel = self._engine.get_root_vel(char_id)[env_ids]
        root_ang_vel = self._engine.get_root_ang_vel(char_id)[env_ids]

        new_rot = torch_util.quat_mul(rand_rot, root_rot)
        new_vel = torch_util.quat_rotate(rand_rot, root_vel)
        new_ang_vel = torch_util.quat_rotate(rand_rot, root_ang_vel)

        self._engine.set_root_pos(env_ids, char_id, root_pos)
        self._engine.set_root_rot(env_ids, char_id, new_rot)
        self._engine.set_root_vel(env_ids, char_id, new_vel)
        self._engine.set_root_ang_vel(env_ids, char_id, new_ang_vel)

        if (self._stand_start_prob > 0.0):
            stand = torch.rand(n, device=self._device) < self._stand_start_prob
            sid = env_ids[stand]
            m = int(sid.shape[0])
            if (m > 0):
                zero_vel = torch.zeros([m, 3], device=self._device, dtype=torch.float)
                s_pos = root_pos[stand].clone()
                s_z = self._init_root_pos[2] if self._stand_start_root_z is None else self._stand_start_root_z
                s_pos[:, 2] = s_z + self._ground_z(sid)
                s_rot = torch_util.quat_mul(rand_rot[stand],
                                            self._init_root_rot.unsqueeze(0).repeat(m, 1))
                self._engine.set_root_pos(sid, char_id, s_pos)
                self._engine.set_root_rot(sid, char_id, s_rot)
                self._engine.set_root_vel(sid, char_id, zero_vel)
                self._engine.set_root_ang_vel(sid, char_id, zero_vel)
                s_dof = self._init_dof_pos if self._stand_start_dof_pos is None else self._stand_start_dof_pos
                self._engine.set_dof_pos(sid, char_id, s_dof.unsqueeze(0).repeat(m, 1))
                self._engine.set_dof_vel(sid, char_id, 0.0)
                self._engine.set_body_vel(sid, char_id, 0.0)
                self._engine.set_body_ang_vel(sid, char_id, 0.0)
        return

    def _nav_active(self):
        return self._nav_cmd_prob > 0.0 or self._nav_force is not None

    def _reset_nav_mode(self, env_ids):
        """Draw the episode mode (navigate vs kick) and, for navigation, a
        fresh command with the ball parked far away. Called after _reset_ball
        and before the history refill, so the refilled frames carry the mode."""
        if (not self._nav_active() or len(env_ids) == 0):
            self._nav_mode[env_ids] = False
            return
        if (self._nav_force is not None):
            nav = torch.full([len(env_ids)], bool(self._nav_force), device=self._device, dtype=torch.bool)
        else:
            nav = torch.rand([len(env_ids)], device=self._device) < self._nav_cmd_prob
        self._nav_mode[env_ids] = nav
        self._nav_vel_filt[env_ids] = 0.0
        nav_ids = env_ids[nav]
        if (len(nav_ids) > 0):
            self._reset_ball(nav_ids, near=False)
            self._resample_nav_cmd(nav_ids)
        return

    def _resample_nav_cmd(self, env_ids):
        n = len(env_ids)
        if (self._nav_fixed_cmd is not None):
            cmd = torch.as_tensor(self._nav_fixed_cmd, device=self._device, dtype=torch.float)
            cmd = cmd.reshape(-1, 3).expand(n, 3).clone()
        else:
            u = torch.rand([n, 3], device=self._device)
            lo = torch.tensor([self._nav_vx_range[0], self._nav_vy_range[0], self._nav_yaw_range[0]],
                              device=self._device)
            hi = torch.tensor([self._nav_vx_range[1], self._nav_vy_range[1], self._nav_yaw_range[1]],
                              device=self._device)
            cmd = (lo + u * (hi - lo)) * self._nav_cmd_scale
            if (self._nav_axis_prob > 0.0):
                axis = torch.rand([n], device=self._device) < self._nav_axis_prob
                kind = torch.randint(0, 4 if self._nav_axis_all else 2, [n], device=self._device)
                back = kind == 0
                mag = 0.1 + 0.9 * torch.rand([n], device=self._device)
                sgn = torch.where(torch.rand([n], device=self._device) < 0.5, -1.0, 1.0)
                cmd[axis, 2] = 0.0
                cmd[axis & back, 0] = -mag[axis & back] * abs(self._nav_vx_range[0]) * self._nav_cmd_scale
                cmd[axis & back, 1] = 0.0
                lat = axis & (kind == 1)
                cmd[lat, 0] = 0.0
                cmd[lat, 1] = sgn[lat] * mag[lat] * self._nav_vy_range[1] * self._nav_cmd_scale
                fwd = axis & (kind == 2)
                cmd[fwd, 0] = mag[fwd] * self._nav_vx_range[1] * self._nav_cmd_scale
                cmd[fwd, 1] = 0.0
                yaw = axis & (kind == 3)
                cmd[yaw, 0] = 0.0
                cmd[yaw, 1] = 0.0
                cmd[yaw, 2] = sgn[yaw] * mag[yaw] * self._nav_yaw_range[1] * self._nav_cmd_scale
            still = torch.rand([n], device=self._device) < self._nav_still_prob
            cmd[still] = 0.0
        self._nav_cmd[env_ids] = cmd
        self._nav_cmd_set_t[env_ids] = self._time_buf[env_ids]
        lo_t, hi_t = self._nav_resample_s
        self._nav_cmd_time[env_ids] = self._time_buf[env_ids] + lo_t \
            + (hi_t - lo_t) * torch.rand([n], device=self._device)
        return

    def _update_nav_cmd(self):
        """Command resampling; on a resample a nav episode may hand over to
        kick mode (nav -> kick transition), with the ball placed near."""
        if (self._kick_to_nav_rate > 0.0 and self._nav_force is None):
            # kick -> nav mid-episode (per second rate), robot state kept
            p_step = self._kick_to_nav_rate * self._engine.get_timestep()
            k2n = torch.logical_and(~self._nonkick_mode(), torch.rand([self.get_num_envs()], device=self._device) < p_step)
            if (self._kick_to_nav_dist > 0.0):
                # only near the ball: the far-ball chase must not be cut short
                root_xy = self._engine.get_root_pos(self._get_char_id())[:, 0:2]
                k2n = torch.logical_and(k2n, torch.linalg.norm(self._get_ball_pos()[:, 0:2] - root_xy, dim=-1)
                                        < self._kick_to_nav_dist)
            k2n_ids = k2n.nonzero(as_tuple=False).flatten()
            if (len(k2n_ids) > 0):
                self.set_nav_mode(k2n_ids, True)
        if ((self._k2n_post_prob > 0.0 or self._k2n_abandon_prob > 0.0) and self._nav_force is None):
            self._update_k2n_events()
        due = torch.logical_and(self._nav_mode, self._time_buf >= self._nav_cmd_time)
        ids = due.nonzero(as_tuple=False).flatten()
        if (len(ids) == 0):
            return
        if (self._nav_switch_prob > 0.0 and self._nav_force is None):
            sw = torch.rand([len(ids)], device=self._device) < self._nav_switch_prob
            sw_ids = ids[sw]
            ids = ids[~sw]
            if (len(sw_ids) > 0):
                self.set_nav_mode(sw_ids, False)
        if (len(ids) > 0):
            self._resample_nav_cmd(ids)
        return

    def _update_k2n_events(self):
        """Event-triggered kick -> nav hand-over (kick episodes only): the first real kick
        of the episode arms a delayed switch with prob k2n_post_prob; independently the chase
        may be abandoned at a random time. The robot state is kept."""
        dt = self._engine.get_timestep()
        kick_mode = ~self._nonkick_mode()
        char_id = self._get_char_id()
        if (self._k2n_post_prob > 0.0):
            body = self._engine.get_body_pos(char_id)
            ball = self._get_ball_pos()
            fd = torch.stack([torch.linalg.norm(body[:, f, :] - ball, dim=-1) for f in self._foot_body_ids],
                             -1).min(-1).values
            sp = torch.linalg.norm(self._engine.get_root_vel(self._get_ball_id())[:, 0:2], dim=-1)
            kicked = kick_mode & ~self._k2n_decided & (fd < self._k2n_kick_dist) & (sp > self._k2n_kick_speed)
            k_ids = kicked.nonzero(as_tuple=False).flatten()
            if (len(k_ids) > 0):
                self._k2n_decided[k_ids] = True
                lo_d, hi_d = self._k2n_delay_s
                delay = lo_d + (hi_d - lo_d) * torch.rand([len(k_ids)], device=self._device)
                take = torch.rand([len(k_ids)], device=self._device) < self._k2n_post_prob
                self._k2n_left[k_ids] = torch.where(take, delay, torch.full_like(delay, float("inf")))
        self._k2n_left -= dt
        self._k2n_abandon_left -= dt
        fire = kick_mode & ((self._k2n_left <= 0.0) | (self._k2n_abandon_left <= 0.0))
        f_ids = fire.nonzero(as_tuple=False).flatten()
        if (len(f_ids) > 0):
            self.set_nav_mode(f_ids, True)
        return

    def set_nav_mode(self, env_ids, nav):
        """Mid-episode mode switch (training transitions and evaluation):
        kick mode gets a ball placed near, nav mode parks it far and draws a
        command. The robot keeps its state; the histories roll on."""
        if (len(env_ids) == 0):
            return
        self._nav_mode[env_ids] = bool(nav)
        self._reset_ball(env_ids, near=not nav)
        if (nav):
            self._resample_nav_cmd(env_ids)
        if (self._virtual_perception):
            self._reset_perception(env_ids)
        char_id = self._get_char_id()
        self._stagnation_anchor_pos[env_ids] = self._engine.get_root_pos(char_id)[env_ids]
        self._stagnation_anchor_time[env_ids] = self._time_buf[env_ids]
        self._record_reset_prev_states(env_ids)
        return

    def _drib_active(self):
        return self._drib_cmd_prob > 0.0 or self._drib_force is not None

    def _nonkick_mode(self):
        """Episodes whose task is not the kick task (navigation or dribble)."""
        return torch.logical_or(self._nav_mode, self._drib_mode)

    def _reset_drib_mode(self, env_ids):
        """Draw dribble episodes among the non-nav ones; the ball is placed near
        (or far with drib_far_prob, the chase stage) and a command is drawn."""
        if (not self._drib_active() or len(env_ids) == 0):
            self._drib_mode[env_ids] = False
            self._cow_mode[env_ids] = False
            return
        if (self._drib_force is not None):
            drib = torch.full([len(env_ids)], bool(self._drib_force), device=self._device, dtype=torch.bool)
        else:
            drib = torch.rand([len(env_ids)], device=self._device) < self._drib_cmd_prob
        drib &= ~self._nav_mode[env_ids]
        self._drib_mode[env_ids] = drib
        drib_ids = env_ids[drib]
        if (len(drib_ids) > 0):
            far = torch.rand([len(drib_ids)], device=self._device) < self._drib_far_prob
            if (far.any()):
                self._reset_ball(drib_ids[far], near=False)
            if ((~far).any()):
                self._reset_ball(drib_ids[~far], near=True)
            self._resample_drib_cmd(drib_ids)
        self._cow_mode[env_ids] = False
        if (self._cow_active() and len(drib_ids) > 0):
            if (self._cow_force is not None):
                cow = torch.full([len(drib_ids)], bool(self._cow_force), device=self._device, dtype=torch.bool)
            else:
                cow = torch.rand([len(drib_ids)], device=self._device) < self._cow_prob
            cow_ids = drib_ids[cow]
            if (len(cow_ids) > 0):
                # cow episodes start with a still ball near the robot (phase 0)
                self._reset_ball(cow_ids, near=True)
                self._cow_mode[cow_ids] = True
                self._cow_spawn_any(cow_ids)
        return

    def _resample_drib_cmd(self, env_ids):
        n = len(env_ids)
        if (self._drib_fixed_cmd is not None):
            cmd = torch.as_tensor(self._drib_fixed_cmd, device=self._device, dtype=torch.float)
            cmd = cmd.reshape(-1, 2).expand(n, 2).clone()
        else:
            cmd = dribble_util.sample_dribble_cmd(n, self._drib_speed_range,
                                                  np.deg2rad(self._drib_dir_max_deg),
                                                  self._drib_still_prob, self._device)
        self._drib_cmd[env_ids] = cmd
        if (self._drib_world_dir and self._drib_fixed_cmd is None):
            self._drib_spd[env_ids] = torch.linalg.norm(cmd, dim=-1)
            self._sample_drib_wp(env_ids)
            self._set_world_drib_cmd(env_ids)
        lo_t, hi_t = self._drib_resample_s
        self._drib_cmd_time[env_ids] = self._time_buf[env_ids] + lo_t \
            + (hi_t - lo_t) * torch.rand([n], device=self._device)
        return

    def _sample_drib_wp(self, ids):
        """Random waypoint inside the field (world frame), the BT stand-in of doc 36."""
        n = len(ids)
        hx = 0.5 * self._field_length - self._drib_wp_margin
        hy = 0.5 * self._field_width - self._drib_wp_margin
        u = 2.0 * torch.rand([n, 2], device=self._device) - 1.0
        self._drib_wp[ids] = self._field_offset[ids] + u * torch.tensor([hx, hy], device=self._device)

    def _set_world_drib_cmd(self, ids):
        """Heading-frame ball-velocity command toward the waypoint: speed * R(-heading) * unit(wp - ball)."""
        ball = self._get_ball_pos()[ids, 0:2]
        rel = self._drib_wp[ids] - ball
        reached = torch.linalg.norm(rel, dim=-1) < 0.7
        if (reached.any()):
            self._sample_drib_wp(ids[reached])
            rel = self._drib_wp[ids] - ball
        des = rel / torch.clamp(torch.linalg.norm(rel, dim=-1, keepdim=True), min=1.0e-6)
        rot = self._engine.get_root_rot(self._get_char_id())[ids]
        d3 = torch.cat([des, torch.zeros_like(des[:, 0:1])], dim=-1)
        v = torch_util.quat_rotate(torch_util.calc_heading_quat_inv(rot), d3)[:, 0:2]
        clip = float(np.deg2rad(self._drib_world_clip_deg))
        ang = torch.atan2(v[:, 1], v[:, 0]).clamp(-clip, clip)
        self._drib_cmd[ids] = self._drib_spd[ids].unsqueeze(-1) * torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)

    def _update_drib_cmd(self):
        if (self._drib_world_dir and self._drib_fixed_cmd is None):
            wm = self._drib_mode & ~self._cow_mode if self._cow2_stage > 0 else self._drib_mode
            ids = wm.nonzero(as_tuple=False).flatten()
            if (len(ids) > 0):
                self._set_world_drib_cmd(ids)
        due = torch.logical_and(self._drib_mode, self._time_buf >= self._drib_cmd_time)
        if (self._cow2_stage > 0):  # cow v2 envs keep the command set by the stage
            due &= ~self._cow_mode
        ids = due.nonzero(as_tuple=False).flatten()
        if (len(ids) > 0 and self._drib_fixed_cmd is None):
            self._resample_drib_cmd(ids)
        return

    def set_drib_mode(self, env_ids, on):
        """Mid-episode switch (evaluation / transitions): dribble keeps the ball where it is
        and draws a command; leaving it hands the ball to the kick task. Robot state is kept."""
        if (len(env_ids) == 0):
            return
        self._drib_mode[env_ids] = bool(on)
        if (on):
            self._nav_mode[env_ids] = False
            self._resample_drib_cmd(env_ids)
        if (self._virtual_perception):
            self._reset_perception(env_ids)
        self._record_reset_prev_states(env_ids)
        return

    def _cache_drib_reward(self):
        """Dribble episodes: ball-velocity tracking (heading frame) when the ball is near,
        chase when it is far, penalty for letting it go; replaces the kick task reward."""
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        ball_pos = self._get_ball_pos()
        ball_vel = self._engine.get_root_vel(self._get_ball_id())
        heading_inv = torch_util.calc_heading_quat_inv(root_rot)
        v_ball_h = torch_util.quat_rotate(heading_inv, ball_vel)[:, 0:2]
        vel_r = dribble_util.compute_dribble_vel_reward(v_ball_h, self._drib_cmd, self._drib_vel_sigma)
        dist = torch.linalg.norm(ball_pos[:, 0:2] - root_pos[:, 0:2], dim=-1)
        near = dribble_util.compute_dribble_near_weight(dist, self._drib_near_dist, self._drib_far_dist)
        chase_r = soccer_util.compute_ball_chase_reward(
            root_pos, self._prev_root_pos, ball_pos, self._steer_stop_dist,
            self._steer_speed_max, self._engine.get_timestep(), self._ball_chase_vel_scale)
        dist_pen = dribble_util.compute_dribble_dist_penalty(dist, self._drib_free_dist, self._drib_max_dist)
        prog_r = dribble_util.compute_dribble_progress_reward(v_ball_h, self._drib_cmd, self._drib_prog_over_pen)
        r_vel = self._reward_drib_vel_w * near * vel_r + self._reward_drib_prog_w * near * prog_r
        moving_f = (torch.linalg.norm(self._drib_cmd, dim=-1) > 0.05).float()
        turn_ang = torch.atan2(self._drib_cmd[:, 1], self._drib_cmd[:, 0])
        if (self._drib_turn_sigma > 0.0):
            turn_r = torch.exp(-torch.square(turn_ang / self._drib_turn_sigma)) * moving_f
        else:
            turn_r = torch.cos(turn_ang) * moving_f
        r_vel = r_vel + self._reward_drib_turn_w * turn_r
        # yaw-rate toward the command: rewards turning the heading toward the ball-velocity command
        # (sign of the command angle), saturating at 1 rad/s; scaled by min(|angle|/0.3, 1).
        wz = self._engine.get_root_ang_vel(char_id)[:, 2]
        yaw_r = torch.tanh(wz * torch.sign(turn_ang)) * torch.clamp(torch.abs(turn_ang) / 0.3, max=1.0) * moving_f
        r_vel = r_vel + self._reward_drib_yaw_w * yaw_r
        r_chase = self._reward_drib_chase_w * (1.0 - near) * chase_r
        r_dist = self._reward_drib_dist_w * dist_pen
        drib = self._drib_mode
        self._task_reward_buf[:] = torch.where(drib, r_vel + r_chase + r_dist, self._task_reward_buf)
        dm = drib.float()
        err = torch.linalg.norm(self._drib_cmd - v_ball_h, dim=-1)
        cmd_dir = torch.atan2(self._drib_cmd[:, 1], self._drib_cmd[:, 0])
        v_dir = torch.atan2(v_ball_h[:, 1], v_ball_h[:, 0])
        d_dir = torch.abs(torch.atan2(torch.sin(v_dir - cmd_dir), torch.cos(v_dir - cmd_dir)))
        moving = (torch.linalg.norm(self._drib_cmd, dim=-1) > 0.05).float()
        d = self._diag
        d.add_mean("drib_frac", dm)
        d.add_mean("reward_drib_vel", dm * r_vel)
        d.add_mean("drib_prog", dm * near * prog_r)
        d.add_mean("drib_turn_cos", dm * turn_r)
        d.add_mean("reward_drib_turn", dm * self._reward_drib_turn_w * turn_r)
        d.add_mean("reward_drib_yaw", dm * self._reward_drib_yaw_w * yaw_r)
        d.add_mean("reward_drib_chase", dm * r_chase)
        d.add_mean("reward_drib_dist", dm * r_dist)
        d.add_sum("drib_steps", dm)
        d.add_sum("drib_vel_err", dm * err)
        d.add_sum("drib_dir_err_deg", dm * moving * torch.rad2deg(d_dir))
        d.add_sum("drib_ball_dist", dm * dist)
        d.add_sum("drib_ball_lost", dm * (dist > self._drib_max_dist).float())
        return

    def _cow_active(self):
        return self._cow_prob > 0.0 or self._cow_force is not None

    def _cow_heading_dir(self, env_ids):
        root_rot = self._engine.get_root_rot(self._get_char_id())[env_ids]
        fwd = torch_util.quat_rotate(torch_util.calc_heading_quat(root_rot),
                                     torch.tensor([1.0, 0.0, 0.0], device=self._device).expand(len(env_ids), 3))
        return fwd[:, 0:2]

    def _cow_spawn(self, env_ids):
        """(Re)start the cow scenario of env_ids around the current ball: new virtual opponent, phase 0."""
        n = len(env_ids)
        ball = self._get_ball_pos()[env_ids, 0:2]
        root = self._engine.get_root_pos(self._get_char_id())[env_ids, 0:2]
        goal = self._goal_pos[env_ids]
        half_x = 0.5 * self._field_length - 0.3
        half_y = 0.5 * self._field_width - 0.3
        lo = torch.stack([-half_x + self._field_offset[env_ids, 0], -half_y + self._field_offset[env_ids, 1]], dim=-1)
        hi = torch.stack([half_x + self._field_offset[env_ids, 0], half_y + self._field_offset[env_ids, 1]], dim=-1)
        heading = self._cow_heading_dir(env_ids)
        opp = cow_util.sample_cow_opponent(ball, heading, goal, self._cow_d_range, np.deg2rad(self._cow_ang_deg),
                                           self._cow_goal_blend, lo, hi)
        self._cow_opp[env_ids] = opp
        self._cow_opp_vel[env_ids] = 0.0
        self._cow_est[env_ids] = opp
        self._cow_est_vel[env_ids] = 0.0
        self._cow_vis[env_ids] = 1.0
        self._cow_phase[env_ids] = cow_util.PH_CTRL
        self._cow_hold[env_ids] = 0.0
        self._cow_since[env_ids] = 0.0
        self._cow_min_bo[env_ids] = 10.0
        self._cow_min_ro[env_ids] = 10.0
        self._cow_axis[env_ids] = heading
        self._cow_goal0[env_ids] = torch.linalg.norm(goal - ball, dim=-1)
        self._cow_passed[env_ids] = False
        return

    def set_cow_mode(self, env_ids, on, reset_ball=False):
        """Mid-episode switch (evaluation / transitions). on: dribble+cow with a fresh opponent around the
        current ball (reset_ball=True also puts a still ball in front of the robot). off: hands the ball to kick."""
        if (len(env_ids) == 0):
            return
        self._cow_mode[env_ids] = bool(on)
        self._drib_mode[env_ids] = bool(on)
        if (on):
            self._nav_mode[env_ids] = False
            if (reset_ball):
                self._reset_ball(env_ids, near=True)
            self._resample_drib_cmd(env_ids)
            self._cow_spawn_any(env_ids)
        if (self._virtual_perception):
            self._reset_perception(env_ids)
        self._record_reset_prev_states(env_ids)
        return

    def _compute_cow_obs_block(self, root_pos, root_rot, env_ids, privileged):
        """[N, 5] cow block for the steering slots: opponent estimate in the heading frame (true + vis=1 when privileged)."""
        sel = (lambda t: t) if env_ids is None else (lambda t: t[env_ids])
        if (privileged):
            opp, vel, vis = sel(self._cow_opp), sel(self._cow_opp_vel), torch.ones_like(sel(self._cow_vis))
        else:
            opp, vel, vis = sel(self._cow_est), sel(self._cow_est_vel), sel(self._cow_vis)
        heading_inv = torch_util.calc_heading_quat_inv(root_rot)
        z = torch.zeros_like(opp[:, 0:1])
        rel = torch_util.quat_rotate(heading_inv, torch.cat([opp - root_pos[:, 0:2], z], dim=-1))[:, 0:2]
        vel_h = torch_util.quat_rotate(heading_inv, torch.cat([vel, z], dim=-1))[:, 0:2]
        return cow_util.compute_cow_obs(rel, vel_h, vis, self._cow_obs_scale)

    def _cow_spawn_any(self, env_ids):
        if (self._cow2_stage > 0):
            self._cow2_spawn(env_ids)
        else:
            self._cow_spawn(env_ids)

    def _cow2_spawn(self, env_ids, is_a=None):
        """Stage A scenario (doc 40 v2): ball at U[dist] m within +-ang of the heading, rolling at U[speed] in a
        random direction (0 = still); the dribble slot shows cow2_ctrl_cmd."""
        n = len(env_ids)
        if (n == 0):
            return
        ball_id = self._get_ball_id()
        root = self._engine.get_root_pos(self._get_char_id())[env_ids, 0:2]
        heading = self._cow_heading_dir(env_ids)
        rel = cow2_util.sample_ring(n, self._cow2_ball_dist[0], self._cow2_ball_dist[1],
                                    float(np.deg2rad(self._cow2_ball_ang_deg)), self._device)
        pos = root + cow2_util.heading_to_world(rel, heading)
        half_x = 0.5 * self._field_length - 0.3
        half_y = 0.5 * self._field_width - 0.3
        off = self._field_offset[env_ids]
        pos = torch.stack([torch.clamp(pos[:, 0], off[:, 0] - half_x, off[:, 0] + half_x),
                           torch.clamp(pos[:, 1], off[:, 1] - half_y, off[:, 1] + half_y)], dim=-1)
        ball3 = self._get_ball_pos()[env_ids].clone()
        ball3[:, 0:2] = pos
        vel3 = torch.zeros([n, 3], device=self._device, dtype=torch.float)
        vel3[:, 0:2] = cow2_util.sample_ball_vel(n, self._cow2_ball_speed[0], self._cow2_ball_speed[1], self._device)
        if (self._cow2_stage == 3 and self._cow2_d_in_prob > 0.0):
            inc = torch.rand(n, device=self._device) < self._cow2_d_in_prob
            rel_in = cow2_util.sample_ring(n, self._cow2_d_in_dist[0], self._cow2_d_in_dist[1],
                                           float(np.deg2rad(self._cow2_d_in_ang_deg)), self._device)
            pos_in = root + cow2_util.heading_to_world(rel_in, heading)
            pos_in = torch.stack([torch.clamp(pos_in[:, 0], off[:, 0] - half_x, off[:, 0] + half_x),
                                  torch.clamp(pos_in[:, 1], off[:, 1] - half_y, off[:, 1] + half_y)], dim=-1)
            v_in = cow2_util.heading_to_world(cow2_util.sample_ball_vel_toward(
                rel_in, self._cow2_d_in_speed[0], self._cow2_d_in_speed[1],
                float(np.deg2rad(self._cow2_d_in_jit_deg)), self._device), heading)
            ball3[:, 0:2] = torch.where(inc.unsqueeze(-1), pos_in, ball3[:, 0:2])
            vel3[:, 0:2] = torch.where(inc.unsqueeze(-1), v_in, vel3[:, 0:2])
        self._engine.set_root_pos(env_ids, ball_id, ball3)
        self._engine.set_root_vel(env_ids, ball_id, vel3)
        self._engine.set_root_ang_vel(env_ids, ball_id, torch.zeros_like(vel3))
        self._prev_ball_pos[env_ids] = ball3
        self._reset_kick_prev(env_ids, ball3)
        self._drib_cmd[env_ids] = torch.tensor(self._cow2_ctrl_cmd, device=self._device, dtype=torch.float)
        self._cow2_prev_fd[env_ids] = 10.0
        self._cow2_prev_speed[env_ids] = torch.linalg.norm(vel3[:, 0:2], dim=-1)
        if (self._cow2_stage >= 2):
            if (is_a is not None):
                self._cow2_isA[env_ids] = is_a
            elif (self._cow2_mix_a > 0.0):
                self._cow2_isA[env_ids] = torch.rand(n, device=self._device) < self._cow2_mix_a
            b_ids = env_ids[~self._cow2_isA[env_ids]]
            if (len(b_ids) > 0):
                if (self._cow2_stage == 3):
                    self._cow2d_new(b_ids)
                else:
                    self._cow2_new_target(b_ids)
        self._cow2_phase[env_ids] = cow2_util.PH_SEEK
        self._cow2_hold[env_ids] = 0.0
        self._cow2_since[env_ids] = 0.0
        self._cow2_htime[env_ids] = 0.0
        self._cow2_cyc_clock[env_ids] = -1.0
        self._cow2_cyc_pend[env_ids] = False
        self._cow2_tdur[env_ids] = 0.0
        return

    def _cow_ball_friction(self, cow):
        """Rolling resistance: constant deceleration of the cow balls' planar speed. Returns the [N, 2] velocity."""
        _bid = self._get_ball_id()
        _v3 = self._engine.get_root_vel(_bid).clone()
        if (self._cow_ball_decel > 0.0):
            dt = self._engine.get_timestep()
            _sp = torch.linalg.norm(_v3[:, 0:2], dim=-1)
            _k = torch.clamp(1.0 - self._cow_ball_decel * dt / torch.clamp(_sp, min=1.0e-6), min=0.0)
            _ids = (cow & (_sp > 0.02)).nonzero(as_tuple=False).flatten()
            if (_ids.numel() > 0):
                _v3[_ids, 0:2] = _v3[_ids, 0:2] * _k[_ids].unsqueeze(-1)
                self._engine.set_root_vel(_ids, _bid, _v3[_ids])
        return _v3[:, 0:2]

    def _cow2_cmd_decel(self):
        """Ball deceleration for the speed-calibrated P command (0 -> legacy |P - ball| / p_scale)."""
        return float(self._cow_ball_decel) if self._cow2_p_cmd_speed else 0.0

    def _cow2_new_target(self, env_ids):
        """Stage B+: new P around the current ball (the ball is not moved), phase CTRL."""
        ball = self._get_ball_pos()[env_ids, 0:2]
        heading = self._cow_heading_dir(env_ids)
        frame = heading
        if (self._cow2_p_approach):
            frame = cow2_util.approach_dir(self._engine.get_root_pos(self._get_char_id())[env_ids, 0:2], ball, heading)
        P, side = cow2_util.sample_target(ball, frame, self._cow2_p_fwd, self._cow2_p_lat,
                                          float(np.deg2rad(self._cow2_p_ang_deg)))
        rel = cow2_util.world_to_heading(P - ball, frame)
        self._cow2_pang[env_ids] = torch.atan2(rel[:, 1], rel[:, 0])
        off = self._field_offset[env_ids]
        hx = 0.5 * self._field_length - 0.5
        hy = 0.5 * self._field_width - 0.5
        P = torch.stack([torch.clamp(P[:, 0], off[:, 0] - hx, off[:, 0] + hx),
                         torch.clamp(P[:, 1], off[:, 1] - hy, off[:, 1] + hy)], dim=-1)
        self._cow2_T[env_ids] = P
        vT = torch.zeros_like(P[:, 0])
        if (self._cow2_pow_prob > 0.0 and self._cow2_pow_v[1] > 0.0):
            on = torch.rand_like(vT) < self._cow2_pow_prob
            vT = torch.where(on, self._cow2_pow_v[0] + (self._cow2_pow_v[1] - self._cow2_pow_v[0]) * torch.rand_like(vT), vT)
            u = P - ball
            u = u / torch.clamp(torch.linalg.norm(u, dim=-1, keepdim=True), min=1.0e-6)
            Pe = P + u * (vT * vT / (2.0 * float(self._cow_ball_decel))).unsqueeze(-1)
            Pe = torch.stack([torch.clamp(Pe[:, 0], off[:, 0] - hx, off[:, 0] + hx),
                              torch.clamp(Pe[:, 1], off[:, 1] - hy, off[:, 1] + hy)], dim=-1)
            # field clamp: keep the v_T the clamped point implies (along the ball->T line)
            vT = torch.sqrt(torch.clamp(2.0 * float(self._cow_ball_decel) * ((Pe - P) * u).sum(-1), min=0.0))
            self._diag.add_mean("cow2p_vT", vT)
            P = Pe
        self._cow2_vT[env_ids] = vT
        self._cow2_tmin[env_ids] = 1.0e6
        self._cow2_tspd[env_ids] = 0.0
        self._cow2_tdone[env_ids] = False
        self._cow2_P[env_ids] = P
        self._cow2_ball0[env_ids] = ball
        self._cow2_pside[env_ids] = side
        self._cow2_pred[env_ids] = ball
        self._cow2_touches[env_ids] = 0.0
        self._cow2_prev_pd[env_ids] = torch.linalg.norm(P - ball, dim=-1)
        self._cow2_phase[env_ids] = cow2_util.PH_CTRL
        self._cow2_since[env_ids] = 0.0
        self._cow2_htime[env_ids] = 0.0
        self._cow2_hold[env_ids] = 0.0
        if (self._cow2_pow_prob > 0.0):
            self._drib_cmd[env_ids] = cow2_util.target_cmd(ball, self._cow2_T[env_ids], heading, self._cow2_p_scale,
                                                           self._cow2_cmd_decel(), self._cow2_vT[env_ids].unsqueeze(-1))
        else:
            self._drib_cmd[env_ids] = cow2_util.target_cmd(ball, P, heading, self._cow2_p_scale, self._cow2_cmd_decel())
        return

    def _cow2d_new(self, env_ids):
        """Stage D scenario: still opponent ahead of the robot, region beside/behind it (ball already placed)."""
        n = len(env_ids)
        root = self._engine.get_root_pos(self._get_char_id())[env_ids, 0:2]
        heading = self._cow_heading_dir(env_ids)
        opp_rel, reg_rel = cow2_util.sample_region(n, self._cow2_d_opp_fwd, self._cow2_d_opp_lat, self._cow2_d_reg_dist,
                                                   float(np.deg2rad(self._cow2_d_reg_ang_deg)), self._device)
        off = self._field_offset[env_ids]
        hx = 0.5 * self._field_length - 0.6
        hy = 0.5 * self._field_width - 0.6
        def clamp(p):
            return torch.stack([torch.clamp(p[:, 0], off[:, 0] - hx, off[:, 0] + hx),
                                torch.clamp(p[:, 1], off[:, 1] - hy, off[:, 1] + hy)], dim=-1)
        # a scenario that does not fit in the field along the heading (robot near a wall) is laid toward the field
        # centre instead: clamping it would squash opponent and region onto the robot (instant "success", smoke c2d01s)
        frame = heading
        R_raw = root + cow2_util.heading_to_world(reg_rel, heading)
        fits = ((R_raw[:, 0] - off[:, 0]).abs() <= hx) & ((R_raw[:, 1] - off[:, 1]).abs() <= hy)
        to_c = off - root
        to_c = to_c / torch.clamp(torch.linalg.norm(to_c, dim=-1, keepdim=True), min=1.0e-6)
        frame = torch.where(fits.unsqueeze(-1), heading, to_c)
        opp = clamp(root + cow2_util.heading_to_world(opp_rel, frame))
        R = clamp(root + cow2_util.heading_to_world(reg_rel, frame))
        if (self._cow2_d_min_ball_opp > 0.0):
            b0 = self._get_ball_pos()[env_ids, 0:2]
            dv = opp - b0
            dn = torch.linalg.norm(dv, dim=-1, keepdim=True)
            u = torch.where(dn > 1.0e-4, dv / torch.clamp(dn, min=1.0e-4), heading)
            opp = clamp(torch.where(dn < self._cow2_d_min_ball_opp, b0 + u * self._cow2_d_min_ball_opp, opp))
        self._diag.add_sum("cow2d_spawn_n", torch.ones_like(fits, dtype=torch.float))
        self._diag.add_sum("cow2d_spawn_center", (~fits).float())
        ball = self._get_ball_pos()[env_ids, 0:2]
        self._cow_opp[env_ids] = opp
        self._cow_opp_vel[env_ids] = 0.0
        self._cow_est[env_ids] = opp
        self._cow_est_vel[env_ids] = 0.0
        self._cow_vis[env_ids] = 1.0
        self._cow2_P[env_ids] = R
        self._cow2_ball0[env_ids] = ball
        rel = cow2_util.world_to_heading(R - opp, frame)
        self._cow2_pang[env_ids] = torch.atan2(rel[:, 1], rel[:, 0])
        self._cow2_pside[env_ids] = torch.sign(rel[:, 1])
        self._cow2_touches[env_ids] = 0.0
        self._cow2_prev_pd[env_ids] = torch.linalg.norm(R - ball, dim=-1)
        self._cow2d_prev_rd[env_ids] = torch.linalg.norm(R - root, dim=-1)
        self._cow2d_ocont[env_ids] = False
        self._cow2d_tball[env_ids] = -1.0
        self._cow2_phase[env_ids] = cow2_util.PH_CTRL
        self._cow2_since[env_ids] = 0.0
        self._drib_cmd[env_ids] = cow2_util.target_cmd(ball, R, heading, self._cow2_p_scale, self._cow2_cmd_decel())
        return

    def _update_cow2_region(self, cow):
        """Stage D tick (doc 40 sec 4b): reach the region P (radius cow2_d_reg_r) with robot + controlled ball,
        past a still opponent. Ends: success | timeout (d_tmax) | ball lost | ball hits the opponent | ball OOB."""
        char_id = self._get_char_id()
        dt = self._engine.get_timestep()
        cf = cow.float()
        root3 = self._engine.get_root_pos(char_id)
        root = root3[:, 0:2]
        ball3 = self._get_ball_pos()
        ball = ball3[:, 0:2]
        if (self._cow2_d_opp_speed > 0.0):  # D3: opponent closes on the ball (pos + vel are in the task obs)
            u = ball - self._cow_opp
            u = u / torch.clamp(torch.linalg.norm(u, dim=-1, keepdim=True), min=1.0e-6)
            ov = torch.where(cow.unsqueeze(-1), u * self._cow2_d_opp_speed, torch.zeros_like(u))
            self._cow_opp_vel[:] = ov
            self._cow_opp[:] = self._cow_opp + ov * dt
        self._diag.add_mean("cow2d_opp_speed", cow.float() * torch.linalg.norm(self._cow_opp_vel, dim=-1))
        vel = self._cow_ball_friction(cow)
        speed = torch.linalg.norm(vel, dim=-1)
        dist = torch.linalg.norm(ball - root, dim=-1)
        fd = self._min_foot_ball_dist(ball3)
        R = self._cow2_P
        pd = torch.linalg.norm(R - ball, dim=-1)
        rd = torch.linalg.norm(R - root, dim=-1)
        self._cow2_since[:] = torch.where(cow, self._cow2_since + dt, self._cow2_since)
        t = self._cow2_since
        hit_ev = cow2_util.touch_start(self._cow2_prev_fd, fd, self._cow2_prev_speed, speed, self._cow2_touch_dist,
                                       self._kick_contact_min_gain, self._cow2_touch_speed) & cow
        self._cow2_touches[:] = self._cow2_touches + hit_ev.float()
        ocont = cow & (torch.linalg.norm(root - self._cow_opp, dim=-1) < self._cow2_d_opp_r)
        self._cow2d_ocont[:] = self._cow2d_ocont | ocont
        bcont = cow & (torch.linalg.norm(ball - self._cow_opp, dim=-1) < self._cow2_d_ball_opp_r)
        off = self._field_offset
        oob = cow & (((ball[:, 0] - off[:, 0]).abs() > 0.5 * self._field_length)
                     | ((ball[:, 1] - off[:, 1]).abs() > 0.5 * self._field_width))
        lost = cow & (dist > self._cow2_lost_dist) & ~bcont & ~oob
        succ = cow & (pd <= self._cow2_d_reg_r) & (rd <= self._cow2_d_reg_r) & ~bcont & ~oob
        if (not self._cow2_d_free_run):
            succ = succ & (fd <= self._cow2_cap_dist)
        if (self._cow2_d_ball_vmax > 0.0):
            succ = succ & (speed <= self._cow2_d_ball_vmax)
        t_bonus = t
        fail_rec = torch.zeros_like(succ)
        if (self._cow2_d_recover_s > 0.0):  # option C: success only if the robot recovers within recover_s of the ball arrival
            ball_in = cow & (pd <= self._cow2_d_reg_r) & ~bcont & ~oob
            self._cow2d_tball[:] = torch.where(ball_in & (self._cow2d_tball < 0.0), t, self._cow2d_tball)
            armed = cow & (self._cow2d_tball >= 0.0)
            in_win = armed & ((t - self._cow2d_tball) <= self._cow2_d_recover_s + 1.0e-6)
            succ = succ & in_win
            fail_rec = armed & ~succ & ~in_win & ~bcont & ~oob & ~lost
            if (not self._cow2_d_free_run):
                t_bonus = torch.where(self._cow2d_tball >= 0.0, self._cow2d_tball, t)
        fail_time = (cow & ~succ & ~bcont & ~oob & ~lost & (t >= self._cow2_d_tmax - 1.0e-6)) | fail_rec

        # rewards
        near = dribble_util.compute_dribble_near_weight(dist, self._drib_near_dist, self._drib_far_dist)
        chase_r = soccer_util.compute_ball_chase_reward(root3, self._prev_root_pos, ball3, self._steer_stop_dist,
                                                        self._steer_speed_max, dt, self._ball_chase_vel_scale)
        pc = self._cow2_d_prog_clip
        prog = torch.clamp((self._cow2_prev_pd - pd) / dt, -pc, pc)
        rprog = torch.clamp((self._cow2d_prev_rd - rd) / dt, -pc, pc)
        bonus = torch.clamp(1.0 - t_bonus / self._cow2_d_tmax, min=0.0) \
            * torch.clamp(1.0 - self._cow2_d_touch_k * torch.clamp(self._cow2_touches - 1.0, min=0.0), min=0.0)
        fallen = root3[:, 2] < 0.45
        r_chase = self._reward_cow2_chase_w * cf * (1.0 - near) * chase_r
        r_prog = self._reward_cow2_d_prog_w * cf * prog
        r_rprog = self._reward_cow2_d_rprog_w * cf * rprog
        r_succ = self._reward_cow2_d_succ_w * succ.float() * bonus
        r_ocont = -self._reward_cow2_d_ocont_w * ocont.float()
        r_xtouch = -self._reward_cow2_d_xtouch_w * (hit_ev & (self._cow2_touches > 1.0)).float()
        hinv = torch_util.calc_heading_quat_inv(self._engine.get_root_rot(char_id))
        v_ball_h = torch_util.quat_rotate(hinv, torch.cat([vel, torch.zeros_like(vel[:, 0:1])], dim=-1))[:, 0:2]
        verr = torch.clamp(torch.linalg.norm(self._drib_cmd - v_ball_h, dim=-1), max=2.0)
        r_verr = -self._reward_cow2_d_verr_w * cf * verr
        self._diag.add_mean("cow2d_verr", cf * verr)
        r_bcont = -self._reward_cow2_d_bcont_w * bcont.float()
        r_oob = -self._reward_cow2_d_oob_w * oob.float()
        r_lost = -self._reward_cow2_lost_w * lost.float()
        r_timeout = -self._reward_cow2_timeout_w * fail_time.float()
        r_fall = -self._reward_cow2_fall_w * fallen.float()
        r_wait = -self._reward_cow2_wait_w * cf
        r = r_chase + r_prog + r_rprog + r_succ + r_ocont + r_xtouch + r_bcont + r_oob + r_lost + r_timeout + r_fall + r_wait + r_verr
        if (self._cow2_replace_reward):
            self._task_reward_buf[:] = torch.where(cow, r, self._task_reward_buf)
        else:
            self._task_reward_buf[:] = self._task_reward_buf + cf * r

        d = self._diag
        d.add_mean("cow2_frac", cf)
        for name, val in (("chase", r_chase), ("d_prog", r_prog), ("d_rprog", r_rprog), ("d_succ", r_succ),
                          ("d_ocont", r_ocont), ("d_xtouch", r_xtouch), ("d_verr", r_verr), ("d_bcont", r_bcont), ("d_oob", r_oob), ("lost", r_lost),
                          ("timeout", r_timeout), ("fall", r_fall), ("wait", r_wait)):
            d.add_mean("reward_cow2_" + name, cf * val)
        d.add_sum("cow2d_succ", succ.float())
        d.add_sum("cow2d_succ_fast", (succ & (t <= self._cow2_d_fast_s + 1.0e-6)).float())
        d.add_sum("cow2d_succ_time", succ.float() * t)
        d.add_sum("cow2d_succ_touches", succ.float() * self._cow2_touches)
        d.add_sum("cow2d_succ_le3", (succ & (self._cow2_touches <= 3.0)).float())
        d.add_sum("cow2d_timeout", fail_time.float())
        d.add_sum("cow2d_lost", lost.float())
        d.add_sum("cow2d_bcont", bcont.float())
        d.add_sum("cow2d_oob", oob.float())
        end = succ | fail_time | lost | bcont | oob
        d.add_sum("cow2d_ocont_ep", (end & self._cow2d_ocont).float())
        d.add_sum("cow2d_ocont_steps", ocont.float())
        d.add_sum("cow2d_steps", cf)
        if (getattr(self, "_cow2d_log", None) is not None and bool(end.any())):  # eval only: per-scenario rows
            code = succ.float() * 1 + fail_time.float() * 2 + lost.float() * 3 + bcont.float() * 4 + oob.float() * 5
            self._cow2d_log.append(torch.stack([code, t.clone(), self._cow2_touches.clone(), self._cow2d_ocont.float(),
                                                self._cow2_pang.clone(), pd, self._cow2d_tball.clone()], dim=-1)[end].cpu())

        self._cow2_prev_fd[:] = fd
        self._cow2_prev_speed[:] = speed
        self._cow2_prev_pd[:] = pd
        self._cow2d_prev_rd[:] = rd
        ids = end.nonzero(as_tuple=False).flatten()
        if (len(ids) > 0):
            self._cow2_spawn(ids)
            if (self._virtual_perception):
                self._reset_perception(ids)
            self._record_reset_prev_states(ids)
        ball = self._get_ball_pos()[:, 0:2]
        cids = cow.nonzero(as_tuple=False).flatten()
        if (len(cids) > 0):
            tgt = self._cow2_P[cids]
            if (self._cow2_d_bypass_clear > 0.0):
                tgt = cow2_util.bypass_target(ball[cids], self._cow_opp[cids], tgt, self._cow2_d_bypass_clear)
                self._diag.add_mean("cow2d_cmd_bypass", (tgt != self._cow2_P[cids]).any(-1).float())
            self._drib_cmd[cids] = cow2_util.target_cmd(ball[cids], tgt, self._cow_heading_dir(cids),
                                                        self._cow2_p_scale, self._cow2_cmd_decel(), self._cow2_d_v_arr) * self._cow2_d_cmd_gain
            self._diag.add_mean("cow2d_cmd_speed", torch.linalg.norm(self._drib_cmd[cids], dim=-1))
        return

    def _update_cow2(self, cow=None):
        if (cow is None and self._cow2_stage >= 2):
            mA = self._cow_mode & self._cow2_isA
            if (self._cow2_stage == 3):
                self._update_cow2_region(self._cow_mode & ~self._cow2_isA)
            else:
                self._update_cow2_target(self._cow_mode & ~self._cow2_isA)
            if (self._cow2_mix_a > 0.0):
                self._diag.add_mean("cow2_mixa_frac", mA.float())
                self._update_cow2(mA)
            return
        """Stage A tick: seek (reach + settle the ball) -> hold (keep it close) -> respawn; rewards replace the
        task reward of cow envs (cow2_replace_reward), diagnostics cow2_*."""
        char_id = self._get_char_id()
        dt = self._engine.get_timestep()
        if (cow is None):
            cow = self._cow_mode
        cf = cow.float()
        root3 = self._engine.get_root_pos(char_id)
        root = root3[:, 0:2]
        ball = self._get_ball_pos()[:, 0:2]
        vel = self._cow_ball_friction(cow)
        rspeed = torch.linalg.norm(self._engine.get_root_vel(char_id)[:, 0:2], dim=-1)
        dist = torch.linalg.norm(ball - root, dim=-1)
        speed = torch.linalg.norm(vel, dim=-1)

        seek = cow & (self._cow2_phase == cow2_util.PH_SEEK)
        hold_ph = cow & (self._cow2_phase == cow2_util.PH_HOLD)
        self._cow2_since[:] = torch.where(cow, self._cow2_since + dt, self._cow2_since)
        self._cow2_hold[:] = torch.where(seek, cow2_util.capture_step(self._cow2_hold, dist, speed, dt, self._cow2_cap_dist,
                                                                      self._cow2_cap_speed), self._cow2_hold)
        self._cow2_htime[:] = torch.where(hold_ph, self._cow2_htime + dt, self._cow2_htime)
        cap = seek & (self._cow2_hold >= self._cow2_cap_hold - 1.0e-6)
        cap_time = self._cow2_since - self._cow2_cap_hold
        lost = cow & (dist > self._cow2_lost_dist)
        fail_time = seek & ~cap & ~lost & (self._cow2_since >= self._cow2_seek_s)
        succ = hold_ph & ~lost & (self._cow2_htime >= self._cow2_hold_s)
        self._cow2_phase[:] = torch.where(cap & ~lost, torch.full_like(self._cow2_phase, cow2_util.PH_HOLD), self._cow2_phase)
        self._cow2_htime[:] = torch.where(cap, torch.zeros_like(self._cow2_htime), self._cow2_htime)

        # rewards
        near = dribble_util.compute_dribble_near_weight(dist, self._drib_near_dist, self._drib_far_dist)
        ctrl = near * torch.exp(-torch.square(dist - self._cow_dstar) / self._cow_sigma_d) \
            * torch.exp(-torch.square(speed) / self._cow_sigma_v)
        chase_r = soccer_util.compute_ball_chase_reward(root3, self._prev_root_pos, self._get_ball_pos(), self._steer_stop_dist,
                                                        self._steer_speed_max, dt, self._ball_chase_vel_scale)
        idle = seek & (rspeed < self._cow2_still_speed) & (dist > self._cow2_cap_dist)
        fallen = root3[:, 2] < 0.45
        r_ctrl = self._reward_cow2_ctrl_w * ctrl
        r_chase = self._reward_cow2_chase_w * (1.0 - near) * chase_r
        r_cap = self._reward_cow2_cap_w * cap.float()
        r_idle = -self._reward_cow2_idle_w * idle.float()
        r_lost = -self._reward_cow2_lost_w * lost.float()
        r_timeout = -self._reward_cow2_timeout_w * fail_time.float()
        r_fall = -self._reward_cow2_fall_w * fallen.float()
        r = r_ctrl + r_chase + r_cap + r_idle + r_lost + r_timeout + r_fall
        if (self._cow2_replace_reward):
            self._task_reward_buf[:] = torch.where(cow, r, self._task_reward_buf)
        else:
            self._task_reward_buf[:] = self._task_reward_buf + cf * r

        d = self._diag
        d.add_mean("cow2_frac", cf)
        for name, val in (("ctrl", r_ctrl), ("chase", r_chase), ("cap", r_cap), ("idle", r_idle), ("lost", r_lost),
                          ("timeout", r_timeout), ("fall", r_fall)):
            d.add_mean("reward_cow2_" + name, cf * val)
        d.add_sum("cow2_cap", cap.float())
        d.add_sum("cow2_cap5", (cap & (cap_time <= 5.0 + 1.0e-6)).float())
        d.add_sum("cow2_cap_time", cap.float() * cap_time)
        d.add_sum("cow2_fail_time", fail_time.float())
        d.add_sum("cow2_lost_seek", (lost & seek).float())
        d.add_sum("cow2_lost_hold", (lost & hold_ph).float())
        d.add_sum("cow2_succ", succ.float())
        d.add_sum("cow2_seek_steps", (seek & (dist > self._cow2_cap_dist)).float())  # still fraction: ball not yet reached
        d.add_sum("cow2_seek_still", idle.float())
        d.add_sum("cow2_hold_steps", hold_ph.float())
        d.add_sum("cow2_hold_near", (hold_ph & (dist < 1.0)).float())
        d.add_sum("cow2_hold_dist", hold_ph.float() * dist)

        ids = (succ | fail_time | lost).nonzero(as_tuple=False).flatten()
        if (len(ids) > 0):
            self._cow2_spawn(ids)
            if (self._virtual_perception):
                self._reset_perception(ids)
            self._record_reset_prev_states(ids)
        return

    def _update_cow2_target(self, cow=None):
        """Stage B tick (doc 40 v2): CTRL (reach the ball, touch it to >= touch_speed) -> TOUCH (until the ball
        is > touch_end from the foot: rest prediction, touch reward) -> ROLL (until the ball rests: hit if
        <= hit_dist from P, hit reward) -> new P around the ball. Extra touches while rolling are counted."""
        char_id = self._get_char_id()
        dt = self._engine.get_timestep()
        if (cow is None):
            cow = self._cow_mode
        cf = cow.float()
        root3 = self._engine.get_root_pos(char_id)
        root = root3[:, 0:2]
        ball3 = self._get_ball_pos()
        ball = ball3[:, 0:2]
        vel = self._cow_ball_friction(cow)
        rspeed = torch.linalg.norm(self._engine.get_root_vel(char_id)[:, 0:2], dim=-1)
        dist = torch.linalg.norm(ball - root, dim=-1)
        speed = torch.linalg.norm(vel, dim=-1)
        fd = self._min_foot_ball_dist(ball3)
        if (self._cow2_pow_prob > 0.0):  # S2b: rest target on the current ball->T line (frozen while the ball rolls)
            _u = self._cow2_T - ball
            _dT = torch.linalg.norm(_u, dim=-1, keepdim=True)
            _Pe = ball + _u / torch.clamp(_dT, min=1.0e-6) * (_dT + (self._cow2_vT * self._cow2_vT).unsqueeze(-1) / (2.0 * float(self._cow_ball_decel)))
            _mv = cow & (self._cow2_phase != cow2_util.PH_ROLL) & (self._cow2_vT > 0.0)
            self._cow2_P[:] = torch.where(_mv.unsqueeze(-1), _Pe, self._cow2_P)
        P = self._cow2_P
        pd = torch.linalg.norm(P - ball, dim=-1)
        ph = self._cow2_phase
        ctrl_ph = cow & (ph == cow2_util.PH_CTRL)
        touch_ph = cow & (ph == cow2_util.PH_TOUCH)
        roll_ph = cow & (ph == cow2_util.PH_ROLL)
        self._cow2_since[:] = torch.where(cow, self._cow2_since + dt, self._cow2_since)

        hit_ev = cow2_util.touch_start(self._cow2_prev_fd, fd, self._cow2_prev_speed, speed, self._cow2_touch_dist,
                                       self._kick_contact_min_gain, self._cow2_touch_speed) & cow
        first = hit_ev & ctrl_ph
        extra = hit_ev & roll_ph
        touch_time = self._cow2_since.clone()
        self._cow2_touches[:] = self._cow2_touches + hit_ev.float() * (ctrl_ph | roll_ph).float()
        # touch end: the ball left the foot -> predicted rest point, touch reward (paid once per touch)
        # (a ball kept at the foot past seek_s, i.e. carried instead of touched, also ends the touch)
        end_t = ((touch_ph & ((fd > self._cow2_touch_end) | (self._cow2_since >= self._cow2_seek_s)))
                 | (first & (fd > self._cow2_touch_end)))
        pred = cow2_util.predict_rest(ball, vel, self._cow_ball_decel)
        self._cow2_pred[:] = torch.where(end_t.unsqueeze(-1), pred, self._cow2_pred)
        stay_set = first if self._cow2_stay_first else end_t
        self._cow2_stay_pos[:] = torch.where(stay_set.unsqueeze(-1), root3[:, 0:2], self._cow2_stay_pos)
        stay_d = torch.linalg.norm(root3[:, 0:2] - self._cow2_stay_pos, dim=-1)  # drift since the touch ended
        # area scenario: tolerance and sigma grow with the distance d0 = |P - ball0| (k = 0: constants, old behaviour)
        d0 = torch.linalg.norm(P - self._cow2_ball0, dim=-1)
        hit_d = self._cow2_hit_dist + self._cow2_hit_k * d0
        sig = self._cow2_p_sigma + self._cow2_p_sigma_k * d0 if self._cow2_p_sigma_k > 0.0 else self._cow2_p_sigma
        touch_r = cow2_util.touch_target_reward(pred, P, sig)
        # robot "still and in a good pose": slow, torso upright, standing height, low trunk angular speed
        rot = self._engine.get_root_rot(char_id)
        up_z = torch_util.quat_rotate(rot, torch.tensor([0.0, 0.0, 1.0], device=self._device).expand(rot.shape[0], 3))[:, 2]
        wnorm = torch.linalg.norm(self._engine.get_root_ang_vel(char_id), dim=-1)
        rstill = ((rspeed < self._cow2_still_speed) & (up_z >= self._cow2_area_tilt_cos)
                  & (root3[:, 2] >= self._cow2_area_min_h) & (wnorm < self._cow2_area_wmax))
        # rest: slow for rest_hold s while rolling (or roll timeout)
        slow = speed < self._cow2_rest_speed
        if (self._cow2_area_ball_vmax > 0.0):  # a ball still rolling slowly inside the area counts as arrived
            slow = slow | ((speed < self._cow2_area_ball_vmax) & (pd <= hit_d))
        if (self._cow2_area_still):  # the scenario ends only with the robot still (or on the roll timeout)
            slow = slow & rstill
        self._cow2_hold[:] = torch.where(roll_ph & slow, self._cow2_hold + dt, torch.where(roll_ph, torch.zeros_like(self._cow2_hold), self._cow2_hold))
        self._cow2_htime[:] = torch.where(roll_ph, self._cow2_htime + dt, self._cow2_htime)
        rest = roll_ph & ((self._cow2_hold >= self._cow2_rest_hold - 1.0e-6) | (self._cow2_htime >= self._cow2_roll_s))
        err = pd
        hit = rest & (err <= hit_d)
        succ = hit & rstill if self._cow2_area_still else hit
        if (self._cow2_stay_r > 0.0):
            succ = succ & (stay_d <= self._cow2_stay_r)
        if (self._cow2_max_touches > 0.0):
            succ = succ & (self._cow2_touches <= self._cow2_max_touches)
        # S2b pass metric (eval): closest approach of the rolling ball to T and its speed there; the event fires once
        # per target, when the ball moves away from T (> 5 cm past the minimum) or rests
        td = torch.linalg.norm(self._cow2_T - ball, dim=-1)
        trk = roll_ph & ~self._cow2_tdone
        closer = trk & (td < self._cow2_tmin)
        self._cow2_tmin[:] = torch.where(closer, td, self._cow2_tmin)
        self._cow2_tspd[:] = torch.where(closer, speed, self._cow2_tspd)
        pev = trk & ((td > self._cow2_tmin + 0.05) | rest)
        self._cow2_tdone[:] = self._cow2_tdone | pev
        pok = pev & (self._cow2_tmin <= self._cow2_pow_tol_d) & (torch.abs(self._cow2_tspd - self._cow2_vT) <= self._cow2_pow_tol_v)
        hit_r = cow2_util.touch_target_reward(ball, P, sig)
        lost = cow & (dist > self._cow2_lost_dist) & ~rest
        fail_time = ctrl_ph & ~first & (self._cow2_since >= self._cow2_seek_s)
        self._cow2_tdur[:] = torch.where(touch_ph, self._cow2_tdur + dt, torch.zeros_like(self._cow2_tdur))
        carry = touch_ph & ~end_t & (self._cow2_tdur >= self._cow2_touch_max_s - 1.0e-6)
        if (self._cow2_touch_max_fail):
            fail_time = fail_time | carry

        # phase transitions
        new_ph = ph.clone()
        new_ph = torch.where(first, torch.full_like(ph, cow2_util.PH_TOUCH), new_ph)
        new_ph = torch.where(end_t, torch.full_like(ph, cow2_util.PH_ROLL), new_ph)
        new_ph = torch.where(extra & (fd <= self._cow2_touch_end), torch.full_like(ph, cow2_util.PH_TOUCH), new_ph)
        self._cow2_phase[:] = torch.where(cow, new_ph, ph)
        self._cow2_htime[:] = torch.where(end_t & ~roll_ph, torch.zeros_like(self._cow2_htime), self._cow2_htime)

        # rewards
        near = dribble_util.compute_dribble_near_weight(dist, self._drib_near_dist, self._drib_far_dist)
        chase_r = soccer_util.compute_ball_chase_reward(root3, self._prev_root_pos, ball3, self._steer_stop_dist,
                                                        self._steer_speed_max, dt, self._ball_chase_vel_scale)
        idle = ctrl_ph & (rspeed < self._cow2_still_speed) & (dist > self._cow2_cap_dist)
        fallen = root3[:, 2] < 0.45
        prog = torch.clamp((self._cow2_prev_pd - pd) / dt, -1.0, 1.0)  # m/s of the ball toward P, in [-1, 1] like drib_prog
        r_chase = self._reward_cow2_chase_w * ctrl_ph.float() * (1.0 - near) * chase_r
        r_prog = self._reward_cow2_prog_w * (roll_ph.float() if self._cow2_prog_roll_only else cf) * prog
        r_touch = self._reward_cow2_touch_w * (end_t & (self._cow2_touches <= 1.0)).float() * touch_r  # 1st touch only
        rel_ev = end_t & (self._cow2_touches <= 1.0)
        to_p = P - ball
        d_p = torch.linalg.norm(to_p, dim=-1)
        v_req = torch.sqrt(2.0 * self._cow_ball_decel * d_p).unsqueeze(-1) * to_p / torch.clamp_min(d_p, 1.0e-6).unsqueeze(-1)
        rel_err = torch.linalg.norm(vel - v_req, dim=-1)
        rel_r = torch.exp(-torch.square(rel_err / self._cow2_rel_sigma))
        r_rel = self._reward_cow2_rel_w * rel_ev.float() * rel_r
        r_sep = self._reward_cow2_sep_w * touch_ph.float() * torch.clamp((fd - self._cow2_prev_fd) / dt, -1.0, 1.0)
        rf = rel_ev.float()
        cosang = torch.sum(vel * to_p, dim=-1) / torch.clamp_min(speed * d_p, 1.0e-6)
        r_hit = self._reward_cow2_hit_w * rest.float() * hit_r
        r_idle = -self._reward_cow2_idle_w * idle.float()
        r_lost = -self._reward_cow2_lost_w * lost.float()
        r_timeout = -self._reward_cow2_timeout_w * fail_time.float()
        r_fall = -self._reward_cow2_fall_w * fallen.float()
        # pre-touch setup potential: ball still in CTRL, so only the robot moves between steps
        to_p = P - ball
        to_p = to_p / torch.clamp(torch.linalg.norm(to_p, dim=-1, keepdim=True), min=1.0e-6)
        def _setup_pot(rxy):
            v = ball - rxy
            v = v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1.0e-6)
            return (v * to_p).sum(-1)
        setup_pot = _setup_pot(root)
        setup_gate = (ctrl_ph & (dist < self._cow2_setup_dist)).float()
        r_setup = self._reward_cow2_setup_w * setup_gate * (setup_pot - _setup_pot(self._prev_root_pos[:, 0:2]))
        r_wait = -self._reward_cow2_wait_w * ctrl_ph.float()
        r_still = self._reward_cow2_still_w * roll_ph.float() * torch.exp(-torch.square(rspeed / max(self._cow2_still_sigma, 1.0e-6)))
        if self._cow2_still_tilt_sigma > 0.0:
            r_still = r_still * torch.exp(-torch.square((1.0 - up_z).clamp(min=0.0) / self._cow2_still_tilt_sigma))
        stay_ph = (roll_ph | touch_ph) if self._cow2_stay_touch else roll_ph
        r_stay = -self._reward_cow2_stay_w * stay_ph.float() * torch.clamp(stay_d, max=self._cow2_stay_clamp)
        r_still = r_still + r_stay
        r_retouch = -self._reward_cow2_retouch_w * extra.float()
        r_succ = self._reward_cow2_succ_w * succ.float()
        r_pass = self._reward_cow2_pass_w * pok.float()
        r = (r_chase + r_prog + r_touch + r_hit + r_idle + r_lost + r_timeout + r_fall + r_setup + r_wait + r_rel + r_sep
             + r_still + r_retouch + r_succ + r_pass)
        if (self._cow2_replace_reward):
            self._task_reward_buf[:] = torch.where(cow, r, self._task_reward_buf)
        else:
            self._task_reward_buf[:] = self._task_reward_buf + cf * r

        d = self._diag
        d.add_mean("cow2_frac", cf)
        for name, val in (("chase", r_chase), ("prog", r_prog), ("touch", r_touch), ("hit", r_hit), ("idle", r_idle),
                          ("lost", r_lost), ("timeout", r_timeout), ("fall", r_fall), ("setup", r_setup), ("wait", r_wait),
                          ("rel", r_rel), ("sep", r_sep), ("still", r_still), ("retouch", r_retouch), ("succ", r_succ), ("pass", r_pass)):
            d.add_mean("reward_cow2_" + name, cf * val)
        # area scenario (doc 43): success, robot-still at rest, distance buckets (near < 3 m, mid 3-7 m, far >= 7 m)
        d.add_sum("cow2b_succ", succ.float())
        d.add_sum("cow2b_rest_still", (rest & rstill).float())
        d.add_sum("cow2b_up_z", rest.float() * up_z)
        d.add_sum("cow2b_stay_d", rest.float() * stay_d)
        d.add_sum("cow2b_stay_ok", (rest & (stay_d <= 0.5)).float())
        d.add_sum("cow2b_roll_n", roll_ph.float())
        d.add_sum("cow2b_roll_moving", (roll_ph & (rspeed > 0.2)).float())
        d.add_sum("cow2b_d0", rest.float() * d0)
        for nm, bk in (("near", d0 < 3.0), ("mid", (d0 >= 3.0) & (d0 < 7.0)), ("far", d0 >= 7.0)):
            d.add_sum("cow2b_rest_" + nm, (rest & bk).float())
            d.add_sum("cow2b_succ_" + nm, (succ & bk).float())
            d.add_sum("cow2b_err_" + nm, (rest & bk).float() * err)
        d.add_sum("cow2c_sep_n", touch_ph.float())
        d.add_sum("cow2c_sep_v", touch_ph.float() * torch.clamp((fd - self._cow2_prev_fd) / dt, -1.0, 1.0))
        d.add_sum("cow2c_rel_n", rf)
        d.add_sum("cow2c_rel_err", rf * rel_err)
        d.add_sum("cow2c_rel_r", rf * rel_r)
        d.add_sum("cow2c_rel_spd_err", rf * torch.abs(speed - torch.linalg.norm(v_req, dim=-1)))
        d.add_sum("cow2c_rel_ang45", rf * (cosang < 0.7071).float())
        d.add_mean("cow2b_setup_pot", setup_gate * setup_pot)
        d.add_sum("cow2b_setup_steps", setup_gate)
        d.add_sum("cow2b_touch_pot", first.float() * setup_pot)  # alignment at the 1st touch
        d.add_sum("cow2b_touch1", first.float())
        d.add_sum("cow2b_touch_le3", (first & (touch_time <= 3.0 + 1.0e-6)).float())
        d.add_sum("cow2b_rest", rest.float())
        d.add_sum("cow2b_hit", hit.float())
        d.add_sum("cow2p_ev", pev.float())
        d.add_sum("cow2p_ok", pok.float())
        d.add_sum("cow2p_ok_pow", (pok & (self._cow2_vT > 0.0)).float())
        d.add_sum("cow2p_ev_pow", (pev & (self._cow2_vT > 0.0)).float())
        d.add_sum("cow2p_fail_pow", ((fail_time | lost) & (self._cow2_vT > 0.0)).float())
        d.add_sum("cow2p_dmin", pev.float() * self._cow2_tmin)
        d.add_sum("cow2p_verr", pev.float() * torch.abs(self._cow2_tspd - self._cow2_vT))
        d.add_sum("cow2p_vsgn", pev.float() * (self._cow2_tspd - self._cow2_vT))
        for _nm, _lo, _hi in (("lo", 0.0, 1.0), ("mid", 1.0, 1.6), ("hi", 1.6, 99.0)):
            _m = (self._cow2_vT > _lo) & (self._cow2_vT <= _hi)
            d.add_sum("cow2p_ev_" + _nm, (pev & _m).float())
            d.add_sum("cow2p_ok_" + _nm, (pok & _m).float())
            d.add_sum("cow2p_vsgn_" + _nm, (pev & _m).float() * (self._cow2_tspd - self._cow2_vT))
            d.add_sum("cow2p_dok_" + _nm, (pev & _m & (self._cow2_tmin <= self._cow2_pow_tol_d)).float())
        d.add_sum("cow2b_err", rest.float() * err)
        d.add_sum("cow2b_err_le1", (rest & (err <= 1.0)).float())
        d.add_sum("cow2b_pred_err", rest.float() * torch.linalg.norm(self._cow2_pred - ball, dim=-1))
        d.add_sum("cow2b_pred_le02", (rest & (torch.linalg.norm(self._cow2_pred - ball, dim=-1) <= 0.2)).float())
        d.add_sum("cow2b_touches", rest.float() * self._cow2_touches)
        d.add_sum("cow2b_one_touch", (rest & (self._cow2_touches <= 1.0)).float())
        d.add_sum("cow2b_fail_time", fail_time.float())
        d.add_sum("cow2b_lost", lost.float())
        for nm, sd in (("pos", self._cow2_pside > 0), ("neg", self._cow2_pside < 0)):
            d.add_sum("cow2b_rest_" + nm, (rest & sd).float())
            d.add_sum("cow2b_hit_" + nm, (hit & sd).float())
        d.add_sum("cow2b_ctrl_steps", (ctrl_ph & (dist > self._cow2_cap_dist)).float())
        d.add_sum("cow2b_ctrl_still", idle.float())
        # stage C cycle: capture -> touch -> hit rest -> recapture (robot within cap_dist), timed from the capture
        near_b = cow & (dist < self._cow2_cap_dist)
        run = self._cow2_cyc_clock >= 0.0
        self._cow2_cyc_clock[:] = torch.where(cow & run, self._cow2_cyc_clock + dt, self._cow2_cyc_clock)
        recap = self._cow2_cyc_pend & near_b
        d.add_sum("cow2c_cyc", recap.float())
        d.add_sum("cow2c_cyc_T", (recap & (self._cow2_cyc_clock <= self._cow2_cyc_T + 1.0e-6)).float())
        d.add_sum("cow2c_cyc_time", recap.float() * self._cow2_cyc_clock)
        r_cyc = self._reward_cow2_cyc_w * (recap & (self._cow2_cyc_clock <= self._cow2_cyc_T + 1.0e-6)).float()
        self._task_reward_buf[:] = self._task_reward_buf + cf * r_cyc
        d.add_mean("reward_cow2_cyc", cf * r_cyc)
        start = (ctrl_ph & near_b & ~run) | recap  # the recapture starts the next cycle
        self._cow2_cyc_clock[:] = torch.where(start, torch.zeros_like(self._cow2_cyc_clock), self._cow2_cyc_clock)
        self._cow2_cyc_pend[:] = torch.where(recap, torch.zeros_like(self._cow2_cyc_pend), self._cow2_cyc_pend)
        self._cow2_cyc_pend[:] = self._cow2_cyc_pend | (hit & (self._cow2_cyc_clock >= 0.0))
        self._cow2_cyc_clock[:] = torch.where(rest & ~hit, -torch.ones_like(self._cow2_cyc_clock), self._cow2_cyc_clock)
        # where the time goes (per cow step): ctrl (before the 1st touch), touch (ball at the foot), roll
        d.add_sum("cow2c_ph_ctrl", ctrl_ph.float())
        d.add_sum("cow2c_carry", (carry & (self._cow2_tdur < self._cow2_touch_max_s + dt - 1.0e-6)).float())  # once per carry
        d.add_sum("cow2c_ph_touch", touch_ph.float())
        d.add_sum("cow2c_ph_roll", roll_ph.float())
        d.add_sum("cow2c_ph_pend", (cow & self._cow2_cyc_pend).float())
        if (getattr(self, "_cow2_fail_log", None) is not None and bool((fail_time | lost).any())):  # eval only: P angle of failed scenarios
            self._cow2_fail_log.append(self._cow2_pang[fail_time | lost].cpu())
        if (getattr(self, "_cow2_err_log", None) is not None and bool(rest.any())):  # eval only: per-scenario rows
            # signed error of the rest point in the ball0->P frame: along > 0 = overshoot, lat = sideways miss
            ax = self._cow2_P - self._cow2_ball0
            ax = ax / torch.clamp(torch.linalg.norm(ax, dim=-1, keepdim=True), min=1.0e-6)
            e = ball - self._cow2_P
            e_along = (e * ax).sum(-1)
            e_lat = e[:, 1] * ax[:, 0] - e[:, 0] * ax[:, 1]
            self._cow2_err_log.append(torch.stack([err, torch.linalg.norm(self._cow2_pred - ball, dim=-1),
                                                   self._cow2_touches, self._cow2_pside, e_along, e_lat,
                                                   self._cow2_pang],
                                                  dim=-1)[rest].cpu())

        self._cow2_prev_fd[:] = fd
        self._cow2_prev_speed[:] = speed
        self._cow2_prev_pd[:] = pd
        ids = rest.nonzero(as_tuple=False).flatten()  # ball at rest: next P around it, the ball stays
        if (len(ids) > 0 and self._cow2_mix_a > 0.0):  # or, with prob cow2_mix_a, a stage-A scenario
            to_a = torch.rand(len(ids), device=self._device) < self._cow2_mix_a
            a_ids = ids[to_a]
            ids = ids[~to_a]
            if (len(a_ids) > 0):
                self._cow2_spawn(a_ids, is_a=True)
                if (self._virtual_perception):
                    self._reset_perception(a_ids)
                self._record_reset_prev_states(a_ids)
        if (len(ids) > 0 and self._cow2_area_respawn):  # area scenario: a fresh scenario around the (standing) robot
            self._cow2_spawn(ids)
            if (self._virtual_perception):
                self._reset_perception(ids)
            self._record_reset_prev_states(ids)
        elif (len(ids) > 0):
            self._cow2_new_target(ids)
        ids = (lost | fail_time).nonzero(as_tuple=False).flatten()  # respawn the ball near the robot (+ new P)
        if (len(ids) > 0):
            self._cow2_spawn(ids)
            if (self._virtual_perception):
                self._reset_perception(ids)
            self._record_reset_prev_states(ids)
        ball = self._get_ball_pos()[:, 0:2]
        # P as the dribble command, re-expressed in the current heading every tick
        cids = cow.nonzero(as_tuple=False).flatten()
        if (len(cids) > 0):
            if (self._cow2_pow_prob > 0.0):  # S2b: aim at T from the current ball, launch speed sqrt(v_T^2 + 2 a d)
                self._drib_cmd[cids] = cow2_util.target_cmd(ball[cids], self._cow2_T[cids], self._cow_heading_dir(cids),
                                                            self._cow2_p_scale, self._cow2_cmd_decel(),
                                                            self._cow2_vT[cids].unsqueeze(-1))
            else:
                self._drib_cmd[cids] = cow2_util.target_cmd(ball[cids], self._cow2_P[cids], self._cow_heading_dir(cids),
                                                            self._cow2_p_scale, self._cow2_cmd_decel())
            if (self._cow2_roll_zero_cmd):
                rz = cids[self._cow2_phase[cids] == cow2_util.PH_ROLL]
                self._drib_cmd[rz] = 0.0
        return

    @staticmethod
    def _cow2_event_diags(sums):
        """Stage A rates (doc 40 v2 gate A1/A2). Seek ends = capture, timeout or ball lost while seeking;
        scenarios ended = success (hold completed), timeout, lost."""
        g = lambda k: sums.get(k, 0.0)
        ratio = lambda a, b: a / b if b > 0.0 else 0.0
        cap = g("cow2_cap")
        seek_end = cap + g("cow2_fail_time") + g("cow2_lost_seek")
        ended = g("cow2_succ") + g("cow2_fail_time") + g("cow2_lost_seek") + g("cow2_lost_hold")
        d_end = g("cow2d_succ") + g("cow2d_timeout") + g("cow2d_lost") + g("cow2d_bcont") + g("cow2d_oob")
        dd = {"cow2d_ended": d_end, "cow2d_succ_rate": ratio(g("cow2d_succ"), d_end),
              "cow2d_succ_fast_rate": ratio(g("cow2d_succ_fast"), d_end),
              "cow2d_time_mean": ratio(g("cow2d_succ_time"), g("cow2d_succ")),
              "cow2d_touches_mean": ratio(g("cow2d_succ_touches"), g("cow2d_succ")),
              "cow2d_le3_rate": ratio(g("cow2d_succ_le3"), g("cow2d_succ")),
              "cow2d_timeout_rate": ratio(g("cow2d_timeout"), d_end), "cow2d_lost_rate": ratio(g("cow2d_lost"), d_end),
              "cow2d_bcont_rate": ratio(g("cow2d_bcont"), d_end), "cow2d_oob_rate": ratio(g("cow2d_oob"), d_end),
              "cow2d_ocont_rate": ratio(g("cow2d_ocont_ep"), d_end),
              "cow2d_ocont_step_frac": ratio(g("cow2d_ocont_steps"), g("cow2d_steps"))}
        return {**dd, "cow2_seek_end": seek_end, "cow2_cap_rate": ratio(cap, seek_end), "cow2_cap5_rate": ratio(g("cow2_cap5"), seek_end),
                "cow2_cap_time_mean": ratio(g("cow2_cap_time"), cap), "cow2_ended": ended,
                "cow2_succ_rate": ratio(g("cow2_succ"), ended),
                "cow2_lost_rate": ratio(g("cow2_lost_seek") + g("cow2_lost_hold"), ended),
                "cow2_timeout_rate": ratio(g("cow2_fail_time"), seek_end),
                "cow2_still_frac": ratio(g("cow2_seek_still"), g("cow2_seek_steps")),
                "cow2_hold_near_frac": ratio(g("cow2_hold_near"), g("cow2_hold_steps")),
                "cow2_hold_dist_mean": ratio(g("cow2_hold_dist"), g("cow2_hold_steps")),
                "cow2_cap_n": cap, "cow2_cap5_n": g("cow2_cap5"), "cow2_lost_n": g("cow2_lost_seek") + g("cow2_lost_hold"),
                "cow2_succ_n": g("cow2_succ"),
                # stage B (cow2b_*): scenarios ended = rest + no touch in time + lost
                "cow2b_ended": g("cow2b_rest") + g("cow2b_fail_time") + g("cow2b_lost"),
                "cow2b_hit_rate": ratio(g("cow2b_hit"), g("cow2b_rest") + g("cow2b_fail_time") + g("cow2b_lost")),
                "cow2b_touch_le3_rate": ratio(g("cow2b_touch_le3"), g("cow2b_touch1") + g("cow2b_fail_time")),
                "cow2b_err_mean": ratio(g("cow2b_err"), g("cow2b_rest")),
                "cow2b_err_le1_rate": ratio(g("cow2b_err_le1"), g("cow2b_rest")),
                "cow2b_pred_err_mean": ratio(g("cow2b_pred_err"), g("cow2b_rest")),
                "cow2b_pred_le02_rate": ratio(g("cow2b_pred_le02"), g("cow2b_rest")),
                "cow2b_touches_mean": ratio(g("cow2b_touches"), g("cow2b_rest")),
                "cow2b_one_touch_rate": ratio(g("cow2b_one_touch"), g("cow2b_rest")),
                "cow2b_timeout_rate": ratio(g("cow2b_fail_time"), g("cow2b_rest") + g("cow2b_fail_time") + g("cow2b_lost")),
                "cow2b_lost_rate": ratio(g("cow2b_lost"), g("cow2b_rest") + g("cow2b_fail_time") + g("cow2b_lost")),
                "cow2b_succ_rate": ratio(g("cow2b_succ"), g("cow2b_rest") + g("cow2b_fail_time") + g("cow2b_lost")),
                "cow2b_rest_still_rate": ratio(g("cow2b_rest_still"), g("cow2b_rest")),
                "cow2b_stay_d_mean": ratio(g("cow2b_stay_d"), g("cow2b_rest")),
                "cow2b_stay_ok_rate": ratio(g("cow2b_stay_ok"), g("cow2b_rest")),
                "cow2b_roll_moving_rate": ratio(g("cow2b_roll_moving"), g("cow2b_roll_n")),
                "cow2b_up_z_mean": ratio(g("cow2b_up_z"), g("cow2b_rest")),
                "cow2b_d0_mean": ratio(g("cow2b_d0"), g("cow2b_rest")),
                "cow2b_succ_near_rate": ratio(g("cow2b_succ_near"), g("cow2b_rest_near")),
                "cow2b_succ_mid_rate": ratio(g("cow2b_succ_mid"), g("cow2b_rest_mid")),
                "cow2b_succ_far_rate": ratio(g("cow2b_succ_far"), g("cow2b_rest_far")),
                "cow2b_err_near_mean": ratio(g("cow2b_err_near"), g("cow2b_rest_near")),
                "cow2b_err_mid_mean": ratio(g("cow2b_err_mid"), g("cow2b_rest_mid")),
                "cow2b_err_far_mean": ratio(g("cow2b_err_far"), g("cow2b_rest_far")),
                "cow2b_hit_pos_rate": ratio(g("cow2b_hit_pos"), g("cow2b_rest_pos")),
                "cow2b_hit_neg_rate": ratio(g("cow2b_hit_neg"), g("cow2b_rest_neg")),
                "cow2b_still_frac": ratio(g("cow2b_ctrl_still"), g("cow2b_ctrl_steps")),
                "cow2c_cyc_T_rate": ratio(g("cow2c_cyc_T"), g("cow2b_rest") + g("cow2b_fail_time") + g("cow2b_lost")),
                "cow2c_cyc_rate": ratio(g("cow2c_cyc"), g("cow2b_rest") + g("cow2b_fail_time") + g("cow2b_lost")),
                "cow2c_cyc_time_mean": ratio(g("cow2c_cyc_time"), g("cow2c_cyc")),
                "cow2c_ph_touch_s": ratio(g("cow2c_ph_touch"), g("cow2b_rest")) / 30.0,  # s per ended-at-rest scenario
                "cow2c_ph_roll_s": ratio(g("cow2c_ph_roll"), g("cow2b_rest")) / 30.0,
                "cow2c_rel_err_mean": ratio(g("cow2c_rel_err"), g("cow2c_rel_n")),
                "cow2c_rel_r_mean": ratio(g("cow2c_rel_r"), g("cow2c_rel_n")),
                "cow2c_rel_spd_err_mean": ratio(g("cow2c_rel_spd_err"), g("cow2c_rel_n")),
                "cow2c_rel_ang45_rate": ratio(g("cow2c_rel_ang45"), g("cow2c_rel_n")),
                "cow2c_carry_rate": ratio(g("cow2c_carry"), g("cow2b_rest") + g("cow2b_fail_time") + g("cow2b_lost")),
                "cow2c_ph_ctrl_s": ratio(g("cow2c_ph_ctrl"), g("cow2b_rest")) / 30.0,
                "cow2c_ph_pend_s": ratio(g("cow2c_ph_pend"), g("cow2c_cyc")) / 30.0,
                "cow2b_hit_n": g("cow2b_hit"), "cow2b_rest_n": g("cow2b_rest"),
                # S2b pass metric (doc 43): ok = closest approach to T <= tol_d with |v - v_T| <= tol_v
                "cow2p_ok_n": g("cow2p_ok"), "cow2p_ev_n": g("cow2p_ev"), "cow2p_ok_pow_n": g("cow2p_ok_pow"),
                "cow2p_ev_pow_n": g("cow2p_ev_pow"), "cow2p_fail_pow_n": g("cow2p_fail_pow"),
                "cow2p_vsgn_mean": ratio(g("cow2p_vsgn"), g("cow2p_ev")),
                **{"cow2p_%s_%s" % (q, k): ratio(g("cow2p_%s_%s" % (q0, k)), g("cow2p_ev_" + k))
                   for k in ("lo", "mid", "hi") for q, q0 in (("ok_rate", "ok"), ("vsgn_mean", "vsgn"), ("dok_rate", "dok"))},
                "cow2p_dmin_mean": ratio(g("cow2p_dmin"), g("cow2p_ev")), "cow2p_verr_mean": ratio(g("cow2p_verr"), g("cow2p_ev"))}

    def _update_cow(self):
        """One tick of the cow episodes: phase machine, rewards (replace the task reward), respawn on
        success/failure (the episode goes on), opponent estimate (noise/dropout)."""
        char_id = self._get_char_id()
        dt = self._engine.get_timestep()
        root = self._engine.get_root_pos(char_id)[:, 0:2]
        ball = self._get_ball_pos()[:, 0:2]
        vel = self._engine.get_root_vel(self._get_ball_id())[:, 0:2]
        cow = self._cow_mode
        cf = cow.float()
        if (self._cow_ball_decel > 0.0):  # rolling resistance: constant deceleration of the ball's horizontal speed
            _bid = self._get_ball_id()
            _v3 = self._engine.get_root_vel(_bid).clone()
            _sp = torch.linalg.norm(_v3[:, 0:2], dim=-1)
            _k = torch.clamp(1.0 - self._cow_ball_decel * dt / torch.clamp(_sp, min=1.0e-6), min=0.0)
            _ids = (cow & (_sp > 0.02)).nonzero(as_tuple=False).flatten()
            if (_ids.numel() > 0):
                _v3[_ids, 0:2] = _v3[_ids, 0:2] * _k[_ids].unsqueeze(-1)
                self._engine.set_root_vel(_ids, _bid, _v3[_ids])
                vel = _v3[:, 0:2]
        p = self._cow_phase_p

        # reactive opponent (stage 4): after the touch it closes on the ball
        if (self._cow_opp_speed > 0.0):
            chase = torch.logical_and(cow, self._cow_phase == cow_util.PH_FLIGHT)
            u = ball - self._cow_opp
            u = u / torch.clamp(torch.linalg.norm(u, dim=-1, keepdim=True), min=1.0e-6)
            ov = torch.where(chase.unsqueeze(-1), u * self._cow_opp_speed, torch.zeros_like(u))
            self._cow_opp_vel[:] = ov
            self._cow_opp[:] = self._cow_opp + ov * dt

        old_phase = self._cow_phase.clone()
        old_since = self._cow_since.clone()
        (phase, hold, since, min_bo, min_ro, axis, ev) = cow_util.cow_phase_step(
            self._cow_phase, self._cow_hold, self._cow_since, self._cow_min_bo, self._cow_min_ro, self._cow_axis,
            ball, vel, root, self._cow_opp, dt, p)
        self._cow_phase[:] = torch.where(cow, phase, self._cow_phase)
        self._cow_hold[:] = torch.where(cow, hold, self._cow_hold)
        self._cow_since[:] = torch.where(cow, since, self._cow_since)
        self._cow_min_bo[:] = torch.where(cow, min_bo, self._cow_min_bo)
        self._cow_min_ro[:] = torch.where(cow, min_ro, self._cow_min_ro)
        self._cow_axis[:] = torch.where(cow.unsqueeze(-1), axis, self._cow_axis)
        ph = self._cow_phase
        gate, touch = ev["gate"] & cow, ev["touch"] & cow
        succ = ev["succ"] & cow
        fail_ball, fail_contact = ev["fail_ball"] & cow, ev["fail_contact"] & cow
        fail_lost, fail_time = ev["fail_lost"] & cow, ev["fail_time"] & cow
        fail = fail_ball | fail_contact | fail_lost | fail_time

        # rewards
        dist = torch.linalg.norm(ball - root, dim=-1)
        speed = torch.linalg.norm(vel, dim=-1)
        near = dribble_util.compute_dribble_near_weight(dist, self._drib_near_dist, self._drib_far_dist)
        ctrl = near * torch.exp(-torch.square(dist - self._cow_dstar) / self._cow_sigma_d) \
            * torch.exp(-torch.square(speed) / self._cow_sigma_v)
        ball3 = self._get_ball_pos()
        chase_r = soccer_util.compute_ball_chase_reward(
            self._engine.get_root_pos(char_id), self._prev_root_pos, ball3, self._steer_stop_dist,
            self._steer_speed_max, dt, self._ball_chase_vel_scale)
        touch_r = cow_util.compute_cow_touch_reward(ball, vel, self._cow_opp, self._cow_radius, self._cow_margin,
                                                    self._cow_v_target)
        clear = torch.clamp((cow_util.ball_pass_clearance(ball, vel, self._cow_opp) - self._cow_radius)
                            / max(self._cow_margin, 1.0e-6), 0.0, 1.0)
        goal_t = cow_util.compute_cow_goal_term(ball, vel, self._goal_pos, clear, self._cow_v_target)
        rec_r = torch.exp(-dist / self._cow_rec_sigma)
        in0 = (old_phase == cow_util.PH_CTRL) & ~gate
        in1 = (old_phase == cow_util.PH_TOUCH) | gate
        in2 = (old_phase == cow_util.PH_FLIGHT) | touch
        in1 = in1 & ~touch
        r_ctrl = self._reward_cow_ctrl_w * in0.float() * ctrl
        r_chase = self._reward_cow_chase_w * in0.float() * (1.0 - near) * chase_r
        t_win = (self._cow_since <= self._cow_touch_window_s + 1.0e-6) if self._cow_touch_window_s > 0.0 \
            else torch.ones_like(cow)
        _past_ok = ((ball - self._cow_opp) * self._cow_axis).sum(-1) > 0.0
        _g2 = _past_ok.float() if self._cow_ctrl2_past > 0.5 else torch.ones_like(ctrl)
        r_ctrl2 = self._reward_cow_ctrl2_w * in2.float() * _g2 * ctrl
        r_chase2 = self._reward_cow_chase2_w * in2.float() * (1.0 - near) * chase_r
        r_touch = self._reward_cow_touch_w * (in2 & t_win).float() * touch_r
        r_idle = -self._reward_cow_idle_w * in1.float()
        r_rec = self._reward_cow_rec_w * in2.float() * rec_r
        r_goal = self._reward_cow_goal_w * in2.float() * goal_t
        r_succ = self._reward_cow_succ_w * succ.float()
        r_fail = -self._reward_cow_fail_w * fail.float()
        fallen = self._engine.get_root_pos(char_id)[:, 2] < 0.45
        r_fall = -self._reward_cow_fall_w * fallen.float()
        r = r_ctrl + r_chase + r_chase2 + r_ctrl2 + r_touch + r_idle + r_rec + r_goal + r_succ + r_fail + r_fall
        self._task_reward_buf[:] = torch.where(cow, r, self._task_reward_buf)

        # diagnostics
        d = self._diag
        d.add_mean("cow_frac", cf)
        for name, val in (("ctrl", r_ctrl), ("chase", r_chase), ("chase2", r_chase2), ("ctrl2", r_ctrl2), ("touch", r_touch), ("idle", r_idle),
                          ("rec", r_rec), ("goal", r_goal), ("succ", r_succ), ("fail", r_fail), ("fall", r_fall)):
            d.add_mean("reward_cow_" + name, cf * val)
        d.add_sum("cow_steps", cf)
        d.add_sum("cow_gate", gate.float())
        d.add_sum("cow_gate_time", gate.float() * old_since)
        d.add_sum("cow_touch", touch.float())
        new_pass = ev["pass_ok"] & cow & ~self._cow_passed  # first pass_ok step of the scenario
        self._cow_passed |= new_pass
        d.add_sum("cow_pass", new_pass.float())
        d.add_sum("cow_succ", succ.float())
        d.add_sum("cow_fail_ball", fail_ball.float())
        d.add_sum("cow_fail_contact", fail_contact.float())
        d.add_sum("cow_fail_lost", fail_lost.float())
        d.add_sum("cow_fail_time", fail_time.float())
        d.add_sum("cow_fail_time_ctrl", (fail_time & (old_phase == cow_util.PH_CTRL)).float())
        side = cow_util.cow_side(self._cow_axis, ball, self._cow_opp)
        end2 = (succ | fail_ball | fail_contact) & (old_phase == cow_util.PH_FLIGHT)
        d.add_sum("cow_side_pos_succ", (end2 & succ & (side > 0)).float())
        d.add_sum("cow_side_pos_fail", (end2 & ~succ & (side > 0)).float())
        d.add_sum("cow_side_neg_succ", (end2 & succ & (side < 0)).float())
        d.add_sum("cow_side_neg_fail", (end2 & ~succ & (side < 0)).float())
        gain = self._cow_goal0 - torch.linalg.norm(self._goal_pos - ball, dim=-1)
        d.add_sum("cow_goal_gain_pos", (succ & (gain > 0)).float())
        d.add_sum("cow_goal_gain", succ.float() * gain)
        d.add_sum("cow_opp_dist", cf * torch.linalg.norm(self._cow_opp - ball, dim=-1))

        # respawn finished scenarios in place (episode continues)
        ids = (succ | fail).nonzero(as_tuple=False).flatten()
        if (len(ids) > 0):
            self._reset_ball(ids, near=True)
            self._cow_spawn(ids)
            if (self._virtual_perception):
                self._reset_perception(ids)
            self._record_reset_prev_states(ids)

        # opponent estimate for the actor: noise + dropout (hold last estimate, vis=0)
        n = len(cow)
        keep = torch.rand([n], device=self._device) >= self._cow_vis_dropout_p
        new_est = self._cow_opp + self._cow_obs_noise_pos * torch.randn_like(self._cow_opp)
        new_vel = self._cow_opp_vel + self._cow_obs_noise_vel * torch.randn_like(self._cow_opp_vel)
        self._cow_est[:] = torch.where(keep.unsqueeze(-1), new_est, self._cow_est)
        self._cow_est_vel[:] = torch.where(keep.unsqueeze(-1), new_vel, self._cow_est_vel)
        self._cow_vis[:] = keep.float()
        return

    def _cache_nav_reward(self):
        """Velocity tracking reward of nav episodes (heading-frame planar
        velocity and yaw rate); replaces the kick task reward there."""
        char_id = self._get_char_id()
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        wz = self._engine.get_root_ang_vel(char_id)[:, 2]
        heading_inv = torch_util.calc_heading_quat_inv(root_rot)
        v_local = torch_util.quat_rotate(heading_inv, root_vel)[:, 0:2]
        raw_v_local = v_local
        raw_yaw_err = self._nav_cmd[:, 2] - wz
        if (self._nav_vel_filter_s > 0.0):
            alpha = min(1.0, self._engine.get_timestep() / self._nav_vel_filter_s)
            vel = torch.cat([v_local, wz.unsqueeze(-1)], dim=-1)
            self._nav_vel_filt.mul_(1.0 - alpha).add_(alpha * vel)
            v_local = self._nav_vel_filt[:, 0:2]
            wz = self._nav_vel_filt[:, 2]
        lin_err2 = torch.sum(torch.square(self._nav_cmd[:, 0:2] - v_local), dim=-1)
        yaw_err = self._nav_cmd[:, 2] - wz
        lin_r = torch.exp(-lin_err2 / self._nav_lin_sigma)
        yaw_r = torch.exp(-torch.square(yaw_err) / self._nav_yaw_sigma)
        nav_r = self._reward_nav_lin_w * lin_r + self._reward_nav_yaw_w * yaw_r
        # stop-command penalty; always computed so the raw value reaches the diag log at weight 0
        feet_v = self._engine.get_body_vel(char_id)[:, self._foot_body_ids, 0:2]
        still_pen = compute_still_penalty(raw_v_local, feet_v, self._nav_cmd, self._still_foot_w)
        settled = (self._time_buf - self._nav_cmd_set_t) >= self._still_settle_s
        still_pen = torch.where(settled, still_pen, torch.zeros_like(still_pen))
        nav_r = nav_r + self._reward_still_w * still_pen
        nav = self._nav_mode
        self._task_reward_buf[:] = torch.where(nav, nav_r, self._task_reward_buf)
        navf = nav.float()
        d = self._diag
        d.add_mean("nav_frac", navf)
        d.add_mean("reward_nav_lin", navf * self._reward_nav_lin_w * lin_r)
        d.add_mean("reward_nav_yaw", navf * self._reward_nav_yaw_w * yaw_r)
        d.add_mean("reward_still", navf * self._reward_still_w * still_pen)
        d.add_mean("still_pen_raw", navf * still_pen)
        d.add_sum("nav_steps", navf)
        d.add_sum("nav_lin_err", navf * torch.sqrt(lin_err2))
        d.add_sum("nav_yaw_err", navf * torch.abs(yaw_err))
        d.add_sum("nav_yaw_err_raw", navf * torch.abs(raw_yaw_err))
        return

    def _record_reset_prev_states(self, env_ids):
        char_id = self._get_char_id()
        self._prev_root_pos[env_ids] = self._engine.get_root_pos(char_id)[env_ids]
        self._prev_ball_pos[env_ids] = self._get_ball_pos()[env_ids]
        self._reset_kick_prev(env_ids, self._get_ball_pos()[env_ids])
        self._ep_peak_ball_speed[env_ids] = 0.0
        self._ep_peak_ball_angle[env_ids] = 0.0
        self._ep_min_foot_dist[env_ids] = 10.0
        return

    def _reset_kick_prev(self, env_ids, ball_pos):
        body_pos = self._engine.get_body_pos(self._get_char_id())[env_ids]
        d = torch.linalg.norm(body_pos[:, self._foot_body_ids, :] - ball_pos.unsqueeze(1), dim=-1)
        self._prev_foot_ball_dist[env_ids] = d.min(dim=1).values
        if (self._prev_foot_pos is not None):
            self._store_prev_foot_state(env_ids)
        self._prev_ball_speed[env_ids] = 0.0
        self._kick_pending_r[env_ids] = 0.0
        self._kick_pending_t[env_ids] = 0

    def _sample_ball_pos(self, env_ids, near=True, gauss=None):
        n = env_ids.shape[0]
        half_x = 0.5 * self._field_length - self._ball_spawn_margin
        half_y = 0.5 * self._field_width - self._ball_spawn_margin

        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)[env_ids]

        ball_pos = torch.zeros([n, 3], device=self._device, dtype=torch.float)
        near_mask = torch.full([n], self._ball_spawn_near and near,
                               device=self._device, dtype=torch.bool)
        if (near_mask.any() and self._ball_curriculum_start_samples >= 0):
            # ball-spawn curriculum (C): growing share of field-wide spawns
            p = soccer_util.compute_curriculum_progress(
                self._get_schedule_samples(),
                self._ball_curriculum_start_samples, self._ball_curriculum_end_samples)
            far_prob = p * self._ball_curriculum_far_prob
            far = torch.rand(n, device=self._device) < far_prob
            near_mask &= ~far
        behind_mask = torch.zeros([n], device=self._device, dtype=torch.bool)
        if (self._ball_behind_start_samples >= 0):
            # c10: growing share of rear-cone spawns (out of the FOV)
            p = soccer_util.compute_curriculum_progress(
                self._get_schedule_samples(),
                self._ball_behind_start_samples, self._ball_behind_end_samples)
            behind_mask = torch.rand(n, device=self._device) \
                < p * self._ball_behind_prob
            near_mask &= ~behind_mask
        if (near_mask.any()):
            # in front of the robot in its heading frame, then clamped into
            # the field so border spawns stay in bounds
            root_rot = self._engine.get_root_rot(char_id)[env_ids]
            heading_rot = torch_util.calc_heading_quat(root_rot)
            offset = torch.zeros([n, 3], device=self._device, dtype=torch.float)
            f_min, f_max = self._ball_spawn_front_min, self._ball_spawn_front_max
            if (self._ball_spawn_front_early is not None):
                q = self._schedule_progress(self._ball_curriculum_start_samples,
                                            self._ball_curriculum_end_samples)
                f_min = (1.0 - q) * self._ball_spawn_front_early[0] + q * f_min
                f_max = (1.0 - q) * self._ball_spawn_front_early[1] + q * f_max
            offset[:, 0] = (f_max - f_min) \
                * torch.rand(n, device=self._device) + f_min
            offset[:, 1] = self._ball_spawn_lateral \
                * (2.0 * torch.rand(n, device=self._device) - 1.0)
            offset = torch_util.quat_rotate(heading_rot, offset)

            ball_local = root_pos[:, 0:2] + offset[:, 0:2] - self._field_offset[env_ids]
            ball_local[:, 0] = torch.clamp(ball_local[:, 0], -half_x, half_x)
            ball_local[:, 1] = torch.clamp(ball_local[:, 1], -half_y, half_y)
            ball_pos[:, 0:2] = ball_local + self._field_offset[env_ids]
        if (behind_mask.any()):
            # rear cone in the heading frame: starts outside the FOV
            root_rot_b = self._engine.get_root_rot(char_id)[env_ids]
            heading_rot_b = torch_util.calc_heading_quat(root_rot_b)
            off2 = soccer_util.compute_behind_spawn_offsets(
                n, self._ball_behind_dist_min, self._ball_behind_dist_max,
                self._ball_behind_half_cone, self._device)
            offset_b = torch_util.quat_rotate(heading_rot_b,
                                              torch.cat([off2, torch.zeros([n, 1], device=self._device)], dim=-1))
            ball_local_b = root_pos[:, 0:2] + offset_b[:, 0:2] - self._field_offset[env_ids]
            ball_local_b[:, 0] = torch.clamp(ball_local_b[:, 0], -half_x, half_x)
            ball_local_b[:, 1] = torch.clamp(ball_local_b[:, 1], -half_y, half_y)
            ball_pos[behind_mask, 0:2] = (ball_local_b + self._field_offset[env_ids])[behind_mask]
        if (not near_mask.all()):
            uni_mask = ~(near_mask | behind_mask)
            if (uni_mask.any()):
                uni = torch.zeros([n, 2], device=self._device, dtype=torch.float)
                uni[:, 0] = half_x * (2.0 * torch.rand(n, device=self._device) - 1.0)
                uni[:, 1] = half_y * (2.0 * torch.rand(n, device=self._device) - 1.0)
                uni += self._field_offset[env_ids]
                ball_pos[uni_mask, 0:2] = uni[uni_mask]
        if (self._ball_goal_strip_prob > 0.0 and near):
            strip = torch.rand(n, device=self._device) < self._ball_goal_strip_prob
            sx = half_x - (half_x - self._ball_goal_strip_min) \
                * torch.rand(n, device=self._device) ** self._ball_goal_strip_pow
            strip_y = half_y if self._ball_goal_strip_y <= 0.0 else min(half_y, self._ball_goal_strip_y)
            sy = strip_y * (2.0 * torch.rand(n, device=self._device) - 1.0)
            sp = torch.stack([sx, sy], dim=-1) + self._field_offset[env_ids]
            ball_pos[strip, 0:2] = sp[strip]
        if (self._ball_goal_core_prob > 0.0 and near):
            core = torch.rand(n, device=self._device) < self._ball_goal_core_prob
            cx = self._ball_goal_core_min + (half_x - self._ball_goal_core_min) * torch.rand(n, device=self._device)
            cy = min(half_y, self._ball_goal_core_y) * (2.0 * torch.rand(n, device=self._device) - 1.0)
            cp = torch.stack([cx, cy], dim=-1) + self._field_offset[env_ids]
            ball_pos[core, 0:2] = cp[core]
        if (gauss is not None and gauss[env_ids].any()):
            # c4: close-range event spawns; clamped into the field and kept
            # off the robot by the min-distance check below. g_mask/g are
            # LOCAL indices into this call's env_ids subset (gauss is a
            # global [num_envs] mask, so indexing by env_ids alone would
            # select the wrong slots and leak out-of-range entries)
            g_mask = gauss[env_ids]
            g_local = g_mask.nonzero(as_tuple=False).flatten()
            g = env_ids[g_local]
            gn = len(g)
            gp = root_pos[g_local, 0:2] \
                + self._ball_event_gauss_sigma * torch.randn([gn, 2], device=self._device)
            gp[:, 0] = torch.clamp(gp[:, 0] - self._field_offset[g, 0], -half_x, half_x)
            gp[:, 1] = torch.clamp(gp[:, 1] - self._field_offset[g, 1], -half_y, half_y)
            ball_pos[g_local, 0:2] = gp + self._field_offset[g]
        ball_pos[:, 2] = self._ball_radius + self._ground_z(env_ids)

        # keep the ball from spawning inside the robot
        delta = ball_pos[:, 0:2] - root_pos[:, 0:2]
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        too_close = (dist < self._ball_spawn_min_dist).squeeze(-1)
        safe_dist = torch.clamp_min(dist, 1e-6)
        pushed = root_pos[:, 0:2] + delta / safe_dist * self._ball_spawn_min_dist
        ball_pos[too_close, 0:2] = pushed[too_close]

        return ball_pos

    def _ball_event_gauss(self):
        """Per-event draw: Gaussian near the robot vs the baseline spawn."""
        p = soccer_util.compute_curriculum_progress(
            self._get_schedule_samples(),
            self._ball_event_gauss_start_samples, self._ball_event_gauss_end_samples)
        return torch.rand(self.get_num_envs(), device=self._device) \
            < p * self._ball_event_gauss_prob

    def _reset_ball(self, env_ids, near=True, gauss=None):
        n = env_ids.shape[0]
        ball_id = self._get_ball_id()

        ball_pos = self._sample_ball_pos(env_ids, near=near, gauss=gauss)
        ball_rot = torch.zeros([n, 4], device=self._device, dtype=torch.float)
        ball_rot[:, 3] = 1.0
        zero_vel = torch.zeros([n, 3], device=self._device, dtype=torch.float)

        self._engine.set_root_pos(env_ids, ball_id, ball_pos)
        self._engine.set_root_rot(env_ids, ball_id, ball_rot)
        self._engine.set_root_vel(env_ids, ball_id, zero_vel)
        self._engine.set_root_ang_vel(env_ids, ball_id, zero_vel)

        self._prev_ball_pos[env_ids] = ball_pos
        self._reset_kick_prev(env_ids, ball_pos)
        self._ball_moving_time[env_ids] = 0.0
        self._ball_still_time[env_ids] = 0.0
        self._resample_ball_perturb_times(env_ids)
        self._k2n_decided[env_ids] = False
        self._k2n_left[env_ids] = float("inf")
        self._k2n_abandon_left[env_ids] = float("inf")
        if (near and self._k2n_abandon_prob > 0.0):
            lo_a, hi_a = self._k2n_abandon_s
            t_ab = lo_a + (hi_a - lo_a) * torch.rand([n], device=self._device)
            take = torch.rand([n], device=self._device) < self._k2n_abandon_prob
            self._k2n_abandon_left[env_ids] = torch.where(take, t_ab, self._k2n_abandon_left[env_ids])
        self._resample_kick_tgt(env_ids, ball_pos)
        return

    def _resample_kick_tgt(self, env_ids, ball_pos):
        """Per-episode kick target (docs/t1/43, P1a). With probability kick_tgt_prob the target is a point at
        distance U(d_lo, d_hi) from the ball, at an angle U(-A, A) around the ball->goal direction (A grows with
        the sample schedule from kick_tgt_ang_deg_start to kick_tgt_ang_deg), clamped into the field.
        Otherwise it is the goal centre, which is the pre-existing behaviour."""
        goal = self._goal_pos[env_ids]
        self._kick_tgt[env_ids] = goal
        self._kick_tgt_on[env_ids] = False
        if (self._kick_tgt_prob <= 0.0):
            return
        n = env_ids.shape[0]
        use = torch.rand([n], device=self._device) < self._kick_tgt_prob
        if (self._kick_tgt_ang_start_samples >= 0):
            prog = soccer_util.compute_curriculum_progress(
                self._get_schedule_samples(), self._kick_tgt_ang_start_samples, self._kick_tgt_ang_end_samples)
            amax = self._kick_tgt_ang_deg_start + prog * (self._kick_tgt_ang_deg - self._kick_tgt_ang_deg_start)
        else:
            amax = self._kick_tgt_ang_deg
        tgt = soccer_util.sample_kick_target(ball_pos[:, 0:2], goal, self._kick_tgt_dist[0], self._kick_tgt_dist[1],
                                             float(np.deg2rad(amax)), self._field_length, self._field_width,
                                             self._field_offset[env_ids], self._ball_spawn_margin)
        self._kick_tgt[env_ids] = torch.where(use.unsqueeze(-1), tgt, goal)
        self._kick_tgt_on[env_ids] = use
        if (self._kick_pow_enable):
            self._resample_kick_pow(env_ids, use)
        return

    def _resample_kick_pow(self, env_ids, use):
        """Power p for the kick-target episodes (docs/t1/44): U(lo, 1), lo annealed from kick_pow_p_lo_start to
        kick_pow_p_lo by samples (top-down curriculum), or kick_pow_p_fixed for evaluation. Resets the event state."""
        n = env_ids.shape[0]
        if (self._kick_pow_p_fixed >= 0.0):
            p = torch.full([n], self._kick_pow_p_fixed, device=self._device)
        else:
            lo = self._kick_pow_p_lo
            if (self._kick_pow_p_lo_start_samples >= 0):
                prog = soccer_util.compute_curriculum_progress(
                    self._get_schedule_samples(), self._kick_pow_p_lo_start_samples, self._kick_pow_p_lo_end_samples)
                lo = self._kick_pow_p_lo_start + prog * (self._kick_pow_p_lo - self._kick_pow_p_lo_start)
            if (self._kick_pow_adapt):
                # half of the draws on the frontier [lo, lo + win), the rest on [lo, 1]
                lo = self._kick_pow_adapt_lo
                hi = torch.where(torch.rand([n], device=self._device) < self._kick_pow_adapt_front_prob,
                                 torch.full([n], min(lo + self._kick_pow_adapt_win, 1.0), device=self._device),
                                 torch.ones([n], device=self._device))
                p = lo + (hi - lo) * torch.rand([n], device=self._device)
            else:
                p = lo + (1.0 - lo) * torch.rand([n], device=self._device)
        self._kick_pow_on[env_ids] = use
        self._kick_pow_p[env_ids] = torch.where(use, p, torch.zeros_like(p))
        self._kick_pow_n[env_ids] = -1
        self._kick_pow_peak[env_ids] = 0.0
        self._kick_pow_derr[env_ids] = 0.0
        self._kick_pow_paid[env_ids] = False
        self._kick_pow_paid_t[env_ids] = 0.0
        return

    def _kick_pow_active(self):
        return torch.logical_and(self._kick_pow_on, ~self._nonkick_mode())



@torch.jit.script
def compute_still_penalty(root_vel_xy, feet_vel_xy, nav_cmd, foot_w):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """Olives' stop penalty (steering env, 7e525b5) on our (vx, vy, wz) command: where the whole command is
    zero, ||v_root_xy|| + foot_w * mean over feet ||v_foot_xy||; zero elsewhere. root_vel_xy [N, 2],
    feet_vel_xy [N, F, 2], nav_cmd [N, 3] -> [N]."""
    base = torch.sqrt(torch.sum(torch.square(root_vel_xy), dim=-1))
    feet = torch.mean(torch.sqrt(torch.sum(torch.square(feet_vel_xy), dim=-1)), dim=-1)
    pen = base + foot_w * feet
    stop = torch.sum(torch.abs(nav_cmd), dim=-1) < 1e-6
    return torch.where(stop, pen, torch.zeros_like(pen))


def compute_nav_cmd_obs(cmd, face_horizon):
    # type: (Tensor, float) -> Tensor
    """Velocity command in the steering-slot layout (heading frame):
    [vx/|v|, vy/|v|, |v|, cos(wz*T), sin(wz*T)]; zero direction when still."""
    speed = torch.linalg.norm(cmd[:, 0:2], dim=-1, keepdim=True)
    direction = cmd[:, 0:2] / torch.clamp(speed, min=1.0e-6)
    ang = cmd[:, 2:3] * face_horizon
    return torch.cat([direction, speed, torch.cos(ang), torch.sin(ang)], dim=-1)
