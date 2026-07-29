import numpy as np
import torch

import engines.engine as engine
import envs.base_env as base_env
import envs.smp_env as smp_env
import envs.soccer_util as soccer_util
import util.torch_util as torch_util


class TaskSoccerEnv(smp_env.SMPEnv):
    """Soccer task env (arXiv:2511.03996): one robot, one ball, one goal.

    The field is a fixed region of the shared world frame, centered at the
    origin (envs only interact with their own ball via per-env collision
    groups). The goal is a virtual line segment on the +x edge of the field;
    there are no physical goal posts.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._field_length = float(env_config.get("field_length", 14.0))
        self._field_width = float(env_config.get("field_width", 9.0))
        self._goal_width = float(env_config.get("goal_width", 2.6))
        self._ball_radius = float(env_config.get("ball_radius", 0.11))
        # keep spawns away from the field border so the ball does not
        # immediately go out of bounds
        self._spawn_margin = float(env_config.get("spawn_margin", 1.0))
        self._ball_spawn_min_dist = float(env_config.get("ball_spawn_min_dist", 0.5))

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

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
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
        self._goal_pos = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._goal_pos[:, 0] = 0.5 * self._field_length
        self._goal_dir = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._goal_dir[:, 0] = -1.0

        self._prev_root_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._prev_ball_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)

        self._task_reward_buf = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._goal_scored_buf = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._ball_oob_buf = torch.zeros([num_envs], device=self._device, dtype=torch.bool)
        self._ball_perturb_times = torch.zeros([num_envs], device=self._device, dtype=torch.float)

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

        task_obs = soccer_util.compute_soccer_observations(root_pos, root_rot, ball_pos,
                                                           goal_pos, goal_dir)
        # ball detection mask (Table 2); the perception model is not simulated
        # yet, so the ball is always visible
        ball_mask = torch.ones_like(task_obs[..., 0:1])
        obs = torch.cat([obs, task_obs, ball_mask], dim=-1)
        return obs

    def _update_misc(self):
        super()._update_misc()
        self._update_task()
        return

    def _update_task(self):
        # 1. detect events with the pre-reset ball state
        ball_pos = self._get_ball_pos()
        self._goal_scored_buf[:] = soccer_util.compute_goal_scored_flags(
            ball_pos, self._goal_pos, self._goal_dir, self._goal_width, self._ball_radius)
        self._ball_oob_buf[:] = soccer_util.compute_ball_out_flags(
            ball_pos, self._field_length, self._field_width,
            self._goal_pos, self._goal_dir, self._goal_width, self._ball_radius)
        # a scored ball is also outside the field; count it only as a goal
        self._ball_oob_buf &= ~self._goal_scored_buf

        # 2. cache the rewards before any ball teleport corrupts the
        #    potentials or the contact geometry
        self._cache_task_reward(ball_pos)
        self._cache_aux_reward(ball_pos)

        # 3. goal or out of bounds resets ONLY the ball; the robot and the
        #    episode clock keep going (soft reset, per the paper)
        soft_reset_mask = torch.logical_or(self._goal_scored_buf, self._ball_oob_buf)
        soft_reset_ids = soft_reset_mask.nonzero(as_tuple=False).flatten()
        if (len(soft_reset_ids) > 0):
            self._reset_ball(soft_reset_ids)

        # 4. random ball perturbations
        self._update_ball_perturb()
        return

    def _cache_task_reward(self, ball_pos):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)

        approach_r = soccer_util.compute_ball_approach_reward(root_pos, self._prev_root_pos,
                                                              ball_pos, self._prev_ball_pos)
        progress_r = soccer_util.compute_goal_progress_reward(ball_pos, self._prev_ball_pos,
                                                              self._goal_pos)
        goal_r = self._goal_scored_buf.float()

        # on the goal step the ball crosses past the potential's minimum, so
        # the shaping terms would fire a large negative spike that cancels the
        # terminal reward; the event step pays only the goal reward
        shaping_mask = (~self._goal_scored_buf).float()

        self._task_reward_buf[:] = shaping_mask * (self._reward_ball_approach_w * approach_r
                                                   + self._reward_goal_progress_w * progress_r) \
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
                self._reset_ball(teleport_ids)

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
            self._reset_ball(env_ids)
            self._record_reset_prev_states(env_ids)
            self._task_reward_buf[env_ids] = 0.0
            self._goal_scored_buf[env_ids] = False
            self._ball_oob_buf[env_ids] = False

            self._aux_reward_buf[env_ids] = 0.0
            self._action_rate_buf[env_ids] = 0.0
            self._prev_action[env_ids] = 0.0
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

    def _sample_ball_pos(self, env_ids):
        n = env_ids.shape[0]
        half_x = 0.5 * self._field_length - self._spawn_margin
        half_y = 0.5 * self._field_width - self._spawn_margin

        ball_pos = torch.zeros([n, 3], device=self._device, dtype=torch.float)
        ball_pos[:, 0] = half_x * (2.0 * torch.rand(n, device=self._device) - 1.0)
        ball_pos[:, 1] = half_y * (2.0 * torch.rand(n, device=self._device) - 1.0)
        ball_pos[:, 2] = self._ball_radius

        # keep the ball from spawning inside the robot
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)[env_ids]
        delta = ball_pos[:, 0:2] - root_pos[:, 0:2]
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        too_close = (dist < self._ball_spawn_min_dist).squeeze(-1)
        safe_dist = torch.clamp_min(dist, 1e-6)
        pushed = root_pos[:, 0:2] + delta / safe_dist * self._ball_spawn_min_dist
        ball_pos[too_close, 0:2] = pushed[too_close]

        return ball_pos

    def _reset_ball(self, env_ids):
        n = env_ids.shape[0]
        ball_id = self._get_ball_id()

        ball_pos = self._sample_ball_pos(env_ids)
        ball_rot = torch.zeros([n, 4], device=self._device, dtype=torch.float)
        ball_rot[:, 3] = 1.0
        zero_vel = torch.zeros([n, 3], device=self._device, dtype=torch.float)

        self._engine.set_root_pos(env_ids, ball_id, ball_pos)
        self._engine.set_root_rot(env_ids, ball_id, ball_rot)
        self._engine.set_root_vel(env_ids, ball_id, zero_vel)
        self._engine.set_root_ang_vel(env_ids, ball_id, zero_vel)

        self._prev_ball_pos[env_ids] = ball_pos
        self._resample_ball_perturb_times(env_ids)
        return
