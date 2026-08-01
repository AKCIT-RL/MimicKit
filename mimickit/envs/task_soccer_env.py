import numpy as np
import torch

import engines.engine as engine
import envs.base_env as base_env
import envs.smp_env as smp_env
import envs.soccer_util as soccer_util
import envs.task_steering_env as task_steering_env
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
        self._ball_spawn_min_dist = float(env_config.get("ball_spawn_min_dist", 0.5))

        # auto steering command toward the ball (T1 kicking-env style); fills
        # the steering-task obs slots preserved by the warm-start transplant
        self._steer_speed_max = float(env_config.get("steer_speed_max", 1.5))
        self._steer_stop_dist = float(env_config.get("steer_stop_dist", 0.45))

        # spawn the ball in front of the robot (T1: "closer ball so robot can
        # actually reach and kick it") instead of uniformly over the field
        self._ball_spawn_near = bool(env_config.get("ball_spawn_near", True))
        self._ball_spawn_front_min = float(env_config.get("ball_spawn_front_min", 0.5))
        self._ball_spawn_front_max = float(env_config.get("ball_spawn_front_max", 2.0))
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

        # goal-stream reward weights (Table 3)
        self._reward_goal_scored_w = float(env_config.get("reward_goal_scored_w", 15.0))
        self._reward_ball_approach_w = float(env_config.get("reward_ball_approach_w", 50.0))
        self._reward_goal_progress_w = float(env_config.get("reward_goal_progress_w", 500.0))

        # directional kick reward (T1 kicking env): ball velocity toward the
        # goal above a threshold, credit concentrated at the impact
        self._reward_kick_direction_w = float(env_config.get("reward_kick_direction_w", 25.0))
        self._kick_direction_min_vel = float(env_config.get("kick_direction_min_vel", 1.0))
        self._kick_direction_decay = float(env_config.get("kick_direction_decay", 0.2))
        self._kick_direction_max = float(env_config.get("kick_direction_max", 30.0))

        # ball-state reward gating (T1 ball_rolling_scale): while the ball
        # rolls, the robot->ball approach potential is zeroed so chasing the
        # ball it just kicked cannot be farmed
        self._ball_rolling_speed = float(env_config.get("ball_rolling_speed", 0.1))
        self._gate_approach_when_rolling = bool(env_config.get("gate_approach_when_rolling", True))

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
        self._reward_foot_proximity_w = float(env_config.get("reward_foot_proximity_w", -5.0))
        self._reward_action_rate_w = float(env_config.get("reward_action_rate_w", -1.0))
        self._reward_joint_limit_w = float(env_config.get("reward_joint_limit_w", -100.0))
        self._reward_base_accel_w = float(env_config.get("reward_base_accel_w", -0.001))

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

        if (self._visualize and self._visualize_field):
            self._build_field_visuals()
        return

    def _build_envs(self, config, num_envs):
        self._ball_id = None
        super()._build_envs(config, num_envs)
        return

    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)

        ball_id = self._build_ball(env_id)
        if (env_id == 0):
            self._ball_id = ball_id
        else:
            assert(ball_id == self._ball_id)
        return

    def _build_ball(self, env_id):
        ball_asset_file = "data/assets/objects/soccer_ball.xml"
        start_pos = np.array([2.0, 0.0, self._ball_radius], dtype=np.float32)
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

        # per-env field centers on a grid (world frame)
        n_cols = int(np.ceil(np.sqrt(num_envs)))
        n_rows = int(np.ceil(num_envs / n_cols))
        pitch_x = self._field_length + 2.0 * self._field_sep
        pitch_y = self._field_width + 2.0 * self._field_sep
        idx = torch.arange(num_envs, device=self._device)
        col = (idx % n_cols).float()
        row = torch.div(idx, n_cols, rounding_mode="floor").float()
        self._field_offset = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._field_offset[:, 0] = (col - 0.5 * (n_cols - 1)) * pitch_x
        self._field_offset[:, 1] = (row - 0.5 * (n_rows - 1)) * pitch_y

        self._goal_pos = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._goal_pos[:, 0] = 0.5 * self._field_length
        self._goal_pos += self._field_offset
        self._goal_dir = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._goal_dir[:, 0] = -1.0

        self._prev_root_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._prev_ball_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)

        self._task_reward_buf = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._goal_scored_buf = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._ball_oob_buf = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._ball_perturb_times = torch.zeros([num_envs], device=self._device, dtype=torch.float)

        # ball motion timers (T1): drive the kick-direction decay, the
        # approach gating and the waiting penalty
        self._ball_moving_time = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._ball_still_time = torch.zeros([num_envs], device=self._device, dtype=torch.float)

        # aux-stream state
        self._aux_reward_buf = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._prev_root_vel = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._stagnation_anchor_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._stagnation_anchor_time = torch.zeros([num_envs], device=self._device, dtype=torch.float)

        action_dim = self._action_space.shape[0]
        self._prev_action = torch.zeros([num_envs, action_dim], device=self._device, dtype=torch.float)
        self._action_rate_buf = torch.zeros([num_envs], device=self._device, dtype=torch.float)

        char_id = self._get_char_id()
        dof_low, dof_high = self._engine.get_obj_dof_limits(0, char_id)
        self._dof_limits_low = torch.tensor(np.asarray(dof_low), device=self._device,
                                            dtype=torch.float)
        self._dof_limits_high = torch.tensor(np.asarray(dof_high), device=self._device,
                                             dtype=torch.float)

        self._foot_body_ids = self._build_body_ids_tensor(self._kick_feet_bodies)
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
        to_goal = self._goal_pos - ball_pos[:, 0:2]
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

        self._action_rate_buf[:] = soccer_util.compute_action_rate_penalty(actions,
                                                                           self._prev_action)
        self._prev_action[:] = actions
        return

    def _record_prev_states(self):
        char_id = self._get_char_id()
        self._prev_root_pos[:] = self._engine.get_root_pos(char_id)
        self._prev_root_vel[:] = self._engine.get_root_vel(char_id)
        self._prev_ball_pos[:] = self._get_ball_pos()
        return

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)

        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        ball_pos = self._get_ball_pos()
        goal_pos = self._goal_pos
        goal_dir = self._goal_dir

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            ball_pos = ball_pos[env_ids]
            goal_pos = goal_pos[env_ids]
            goal_dir = goal_dir[env_ids]

        # steering command slots first, so the obs prefix (char + steering
        # task dims) matches the steering checkpoint layout column-for-column
        steer_cmd = soccer_util.compute_ball_steer_command(root_pos, ball_pos,
                                                           self._steer_stop_dist,
                                                           self._steer_speed_max)
        steer_obs = task_steering_env.compute_steering_observations(
            root_rot, steer_cmd[..., 0:2], steer_cmd[..., 2], steer_cmd[..., 0:2])

        task_obs = soccer_util.compute_soccer_observations(root_pos, root_rot, ball_pos,
                                                           goal_pos, goal_dir)
        # ball detection mask (Table 2); the perception model is not simulated
        # yet, so the ball is always visible
        ball_mask = torch.ones_like(task_obs[..., 0:1])
        obs = torch.cat([obs, steer_obs, task_obs, ball_mask], dim=-1)
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

        # 3. goal or out of bounds resets ONLY the ball; the robot and the
        #    episode clock keep going (soft reset, per the paper)
        soft_reset_mask = torch.logical_or(self._goal_scored_buf, self._ball_oob_buf)
        soft_reset_ids = soft_reset_mask.nonzero(as_tuple=False).flatten()
        if (len(soft_reset_ids) > 0):
            self._reset_ball(soft_reset_ids, near=not self._ball_soft_reset_far)

        # 4. random ball perturbations
        self._update_ball_perturb()
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
                                                              self._goal_pos)
        dir_r = soccer_util.compute_kick_direction_reward(
            ball_pos, ball_vel, self._goal_pos, self._kick_direction_min_vel,
            self._kick_direction_decay, self._ball_moving_time, self._kick_direction_max)
        goal_r = self._goal_scored_buf.float()

        # on the goal step the ball crosses past the potential's minimum, so
        # the shaping terms would fire a large negative spike that cancels the
        # terminal reward; the event step pays only the goal reward
        shaping_mask = (~self._goal_scored_buf).float()

        self._task_reward_buf[:] = shaping_mask * (self._reward_ball_approach_w * approach_r
                                                   + self._reward_goal_progress_w * progress_r
                                                   + self._reward_kick_direction_w * dir_r) \
            + self._reward_goal_scored_w * goal_r
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
        aux_r += self._reward_stagnation_w * stagnant.float()
        if (window_done.any()):
            self._stagnation_anchor_pos[window_done] = root_pos[window_done]
            self._stagnation_anchor_time[window_done] = self._time_buf[window_done]

        # kick shaping: sideways foot motion in ball contact is rewarded,
        # frontal (toe-poke) motion is penalized
        body_pos = self._engine.get_body_pos(char_id)
        body_vel = self._engine.get_body_vel(char_id)
        for i in range(len(self._foot_body_ids)):
            foot_id = self._foot_body_ids[i]
            foot_pos = body_pos[:, foot_id, :]
            foot_vel = body_vel[:, foot_id, :]
            contact = soccer_util.compute_ball_contact_flags(foot_pos, ball_pos,
                                                             self._ball_contact_dist)
            kick = soccer_util.compute_kick_components(root_rot, foot_vel, contact)
            aux_r += self._reward_kick_sideways_w * kick[:, 0] \
                + self._reward_kick_forward_w * kick[:, 1]

        # regularizations
        left_foot_pos = body_pos[:, self._foot_body_ids[0], :]
        right_foot_pos = body_pos[:, self._foot_body_ids[1], :]
        foot_prox = soccer_util.compute_foot_proximity_penalty(left_foot_pos, right_foot_pos,
                                                               self._foot_min_dist)
        aux_r += self._reward_foot_proximity_w * foot_prox

        aux_r += self._reward_action_rate_w * self._action_rate_buf

        dof_pos = self._engine.get_dof_pos(char_id)
        joint_limit = soccer_util.compute_joint_limit_penalty(dof_pos, self._dof_limits_low,
                                                              self._dof_limits_high)
        aux_r += self._reward_joint_limit_w * joint_limit

        dt = self._engine.get_timestep()
        base_accel = soccer_util.compute_base_accel_penalty(root_vel, self._prev_root_vel, dt)
        aux_r += self._reward_base_accel_w * base_accel

        # waiting penalty (T1): grows quadratically with ball-still time,
        # saturating at waiting_time_max; zero while the ball rolls
        wait_frac = torch.clamp(self._ball_still_time / self._waiting_time_max, max=1.0)
        aux_r += self._reward_waiting_w * wait_frac * wait_frac

        self._aux_reward_buf[:] = aux_r
        return

    def _update_done(self):
        super()._update_done()
        # fall termination penalty joins the aux stream once dones are known
        fail_mask = (self._done_buf == base_env.DoneFlags.FAIL.value)
        self._aux_reward_buf += self._reward_termination_w * fail_mask.float()
        return

    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)
        self._info["aux_reward"] = self._aux_reward_buf
        return

    def _update_ball_perturb(self):
        trigger_mask = self._time_buf >= self._ball_perturb_times
        env_ids = trigger_mask.nonzero(as_tuple=False).flatten()
        n = len(env_ids)
        if (n > 0):
            ball_id = self._get_ball_id()
            teleport_mask = torch.rand(n, device=self._device) < self._ball_perturb_teleport_prob

            teleport_ids = env_ids[teleport_mask]
            if (len(teleport_ids) > 0):
                self._reset_ball(teleport_ids, near=not self._ball_soft_reset_far)

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

    def _update_reward(self):
        self._reward_buf[:] = self._task_reward_buf
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._reset_char_placement(env_ids)
            # the placement moved/rotated the root, so the rigid body state
            # written by char_env's reset is stale; recompute it with FK or
            # key-body obs will mix the old pose with the new root position
            self._reset_char_rigid_body_state(env_ids)
            self._reset_ball(env_ids)
            self._record_reset_prev_states(env_ids)
            self._task_reward_buf[env_ids] = 0.0
            self._goal_scored_buf[env_ids] = False
            self._ball_oob_buf[env_ids] = False

            self._aux_reward_buf[env_ids] = 0.0
            self._action_rate_buf[env_ids] = 0.0
            self._prev_action[env_ids] = 0.0
            self._ball_moving_time[env_ids] = 0.0
            self._ball_still_time[env_ids] = 0.0
            char_id = self._get_char_id()
            self._prev_root_vel[env_ids] = self._engine.get_root_vel(char_id)[env_ids]
            self._stagnation_anchor_pos[env_ids] = self._engine.get_root_pos(char_id)[env_ids]
            self._stagnation_anchor_time[env_ids] = self._time_buf[env_ids]
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
        return

    def _record_reset_prev_states(self, env_ids):
        char_id = self._get_char_id()
        self._prev_root_pos[env_ids] = self._engine.get_root_pos(char_id)[env_ids]
        self._prev_ball_pos[env_ids] = self._get_ball_pos()[env_ids]
        return

    def _sample_ball_pos(self, env_ids, near=True):
        n = env_ids.shape[0]
        half_x = 0.5 * self._field_length - self._spawn_margin
        half_y = 0.5 * self._field_width - self._spawn_margin

        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)[env_ids]

        ball_pos = torch.zeros([n, 3], device=self._device, dtype=torch.float)
        if (self._ball_spawn_near and near):
            # in front of the robot in its heading frame, then clamped into
            # the field so border spawns stay in bounds
            root_rot = self._engine.get_root_rot(char_id)[env_ids]
            heading_rot = torch_util.calc_heading_quat(root_rot)
            offset = torch.zeros([n, 3], device=self._device, dtype=torch.float)
            offset[:, 0] = (self._ball_spawn_front_max - self._ball_spawn_front_min) \
                * torch.rand(n, device=self._device) + self._ball_spawn_front_min
            offset[:, 1] = self._ball_spawn_lateral \
                * (2.0 * torch.rand(n, device=self._device) - 1.0)
            offset = torch_util.quat_rotate(heading_rot, offset)

            ball_local = root_pos[:, 0:2] + offset[:, 0:2] - self._field_offset[env_ids]
            ball_local[:, 0] = torch.clamp(ball_local[:, 0], -half_x, half_x)
            ball_local[:, 1] = torch.clamp(ball_local[:, 1], -half_y, half_y)
            ball_pos[:, 0:2] = ball_local + self._field_offset[env_ids]
        else:
            ball_pos[:, 0] = half_x * (2.0 * torch.rand(n, device=self._device) - 1.0)
            ball_pos[:, 1] = half_y * (2.0 * torch.rand(n, device=self._device) - 1.0)
            ball_pos[:, 0:2] += self._field_offset[env_ids]
        ball_pos[:, 2] = self._ball_radius

        # keep the ball from spawning inside the robot
        delta = ball_pos[:, 0:2] - root_pos[:, 0:2]
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        too_close = (dist < self._ball_spawn_min_dist).squeeze(-1)
        safe_dist = torch.clamp_min(dist, 1e-6)
        pushed = root_pos[:, 0:2] + delta / safe_dist * self._ball_spawn_min_dist
        ball_pos[too_close, 0:2] = pushed[too_close]

        return ball_pos

    def _reset_ball(self, env_ids, near=True):
        n = env_ids.shape[0]
        ball_id = self._get_ball_id()

        ball_pos = self._sample_ball_pos(env_ids, near=near)
        ball_rot = torch.zeros([n, 4], device=self._device, dtype=torch.float)
        ball_rot[:, 3] = 1.0
        zero_vel = torch.zeros([n, 3], device=self._device, dtype=torch.float)

        self._engine.set_root_pos(env_ids, ball_id, ball_pos)
        self._engine.set_root_rot(env_ids, ball_id, ball_rot)
        self._engine.set_root_vel(env_ids, ball_id, zero_vel)
        self._engine.set_root_ang_vel(env_ids, ball_id, zero_vel)

        self._prev_ball_pos[env_ids] = ball_pos
        self._ball_moving_time[env_ids] = 0.0
        self._ball_still_time[env_ids] = 0.0
        self._resample_ball_perturb_times(env_ids)
        return
