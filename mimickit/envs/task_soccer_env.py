import numpy as np
import torch

import gymnasium.spaces as spaces
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

        # per-env static domain randomization (Frente C; ranges follow the
        # HTWK T1 deploy stack, which is validated on real hardware)
        self._rand_ball_props = bool(env_config.get("rand_ball_props", False))
        self._rand_ball_mass_scale = list(env_config.get("rand_ball_mass_scale", [0.7, 1.3]))
        self._rand_ball_friction = list(env_config.get("rand_ball_friction", [0.2, 1.2]))
        self._rand_ball_restitution = list(env_config.get("rand_ball_restitution", [0.1, 0.9]))

        self._rand_char_props = bool(env_config.get("rand_char_props", False))
        self._rand_char_friction = list(env_config.get("rand_char_friction", [0.1, 2.0]))
        self._rand_char_base_mass_scale = list(env_config.get("rand_char_base_mass_scale", [0.8, 1.2]))
        self._rand_char_base_com = list(env_config.get("rand_char_base_com", [-0.1, 0.1]))
        self._rand_char_other_mass_scale = list(env_config.get("rand_char_other_mass_scale", [0.98, 1.02]))

        # random velocity pushes on the robot (paper: physical confrontation;
        # magnitude matches the HTWK T1 push: ~10 N x 1 s / ~30 kg)
        self._char_push_enable = bool(env_config.get("char_push_enable", False))
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

        # uneven ground (engine-side): inject one tile per field so the
        # engine's ground meshes and the env's field grid cannot diverge;
        # spawns are raised by the bump amplitude
        self._ground_z_offset = 0.0
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
            # uneven ground uses env_spacing 0 (one global mesh region per
            # field); spread actors over their fields already at creation so
            # they are not all piled at the origin, which explodes the PhysX
            # GPU broadphase pair count before the first reset
            self._build_field_offsets = offsets

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
        # head gaze reward (Frente G, paper Table 3 head term): cosine of the
        # angle between the camera body boresight and the ball; requires
        # percep_fov_body. 0 disables (G1 baseline parity).
        self._reward_head_gaze_w = float(env_config.get("reward_head_gaze_w", 0.0))

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
        if (self._build_field_offsets is not None):
            # temporarily shift the char spawn to this env's field so actors
            # are created spread out (restored right after)
            offset = self._build_field_offsets[env_id]
            orig_root_pos = self._init_root_pos
            self._init_root_pos = orig_root_pos.clone()
            self._init_root_pos[0] += float(offset[0])
            self._init_root_pos[1] += float(offset[1])
            self._init_root_pos[2] += self._ground_z_offset
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

        self._randomize_env_props(env_id, ball_id)
        return

    def _randomize_env_props(self, env_id, ball_id):
        """Per-env static randomization, applied at build time (before the sim
        is initialized, which is required by Isaac Gym's GPU pipeline)."""
        if (self._rand_ball_props):
            friction = np.random.uniform(self._rand_ball_friction[0], self._rand_ball_friction[1])
            restitution = np.random.uniform(self._rand_ball_restitution[0],
                                            self._rand_ball_restitution[1])
            self._engine.set_obj_shape_props(env_id, ball_id, friction=friction,
                                             restitution=restitution)
            num_bodies = self._engine.get_obj_num_bodies(ball_id)
            mass_scale = np.random.uniform(self._rand_ball_mass_scale[0],
                                           self._rand_ball_mass_scale[1])
            self._engine.scale_obj_masses(env_id, ball_id, np.full([num_bodies], mass_scale))

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
        start_pos = np.array([2.0, 0.0, self._ball_radius + self._ground_z_offset],
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

        # ball motion timers (T1): drive the kick-direction decay, the
        # approach gating and the waiting penalty
        self._ball_moving_time = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._ball_still_time = torch.zeros([num_envs], device=self._device, dtype=torch.float)

        # aux-stream state
        self._aux_reward_buf = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._prev_root_vel = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
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
        self._percep_buf_head = torch.zeros([num_envs], device=self._device, dtype=torch.long)
        self._percep_ball_pos = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        self._percep_ball_valid = torch.ones([num_envs], device=self._device, dtype=torch.bool)

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
        self._action_rate_buf = torch.zeros([num_envs], device=self._device, dtype=torch.float)

        # measurable-frame history (Frente F): 6 proprio + 3 * D + 12 task
        if (self._measurable_obs):
            self._meas_frame_dim = 6 + 3 * action_dim + 12
            self._meas_hist_buf = torch.zeros(
                [num_envs, self._meas_hist_steps, self._meas_frame_dim],
                device=self._device, dtype=torch.float)

        char_id = self._get_char_id()
        dof_low, dof_high = self._engine.get_obj_dof_limits(0, char_id)
        self._dof_limits_low = torch.tensor(np.asarray(dof_low), device=self._device,
                                            dtype=torch.float)
        self._dof_limits_high = torch.tensor(np.asarray(dof_high), device=self._device,
                                             dtype=torch.float)

        self._foot_body_ids = self._build_body_ids_tensor(self._kick_feet_bodies)
        if (self._percep_fov_body != ""):
            self._fov_body_id = int(self._build_body_ids_tensor([self._percep_fov_body])[0].item())
        else:
            self._fov_body_id = None
        assert (self._reward_head_gaze_w == 0.0 or self._fov_body_id is not None), \
            "reward_head_gaze_w requires percep_fov_body (the gaze is measured " \
            "about the camera body's boresight)"
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
        self._total_env_steps += 1

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

    def _compute_task_block(self, env_ids=None, privileged=False):
        """The 12-dim task block exactly as the actor sees it this step:
        steering obs (annealed), soccer obs (perceived ball when the virtual
        perception is on) and the ball detection mask. With privileged=True
        the block uses the TRUE ball state and mask 1 (asymmetric critic)."""
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        ball_pos = self._get_ball_pos()
        goal_pos = self._goal_pos
        goal_dir = self._goal_dir

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
            self._total_env_steps * self.get_num_envs(),
            self._steer_anneal_start_samples, self._steer_anneal_end_samples)
        if (steer_scale < 1.0):
            steer_obs = steer_obs * steer_scale

        task_obs = soccer_util.compute_soccer_observations(root_pos, root_rot, ball_pos,
                                                           goal_pos, goal_dir)
        # ball detection mask (Table 2); real dropout when the virtual
        # perception pipeline is enabled, constant 1 otherwise
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
            frame = self._compute_measurable_frame(env_ids)
            hist = self._meas_hist_buf if env_ids is None else self._meas_hist_buf[env_ids]
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

        # 3. goal / out of bounds terminate the episode (paper 4.1); the
        #    flags become dones in _update_done and the reset keeps the
        #    robot in place (see _reset_envs)

        # 4. random ball perturbations + robot pushes
        self._update_ball_perturb()
        self._update_char_push()
        if (self._virtual_perception):
            self._update_perception()
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
            self._meas_hist_buf[:, -1] = self._compute_measurable_frame()
        return

    def _update_perception(self):
        """One tick of the virtual camera pipeline (Frente E).

        Deliver first (measurements captured ~latency ago become what the
        actor sees; if several are due, the most recent wins), then capture
        (a new noisy measurement enters the ring at the sampled frame rate)."""
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
            self._percep_ball_pos[upd] = self._percep_buf_pos[env_ids, slots][valid]
            self._percep_ball_valid[env_ids] = valid
            self._percep_buf_deliver[due] = float("inf")

        # capture
        capture = self._time_buf >= self._percep_next_capture
        if (capture.any()):
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
                    0.5 * self._percep_fov_deg * np.pi / 180.0)
            else:
                in_fov = soccer_util.compute_ball_in_fov(
                    root_pos, root_rot, ball_pos,
                    0.5 * self._percep_fov_deg * np.pi / 180.0)
            detect_prob = soccer_util.compute_ball_detection_prob(
                dist, in_fov, self._percep_detect_prob,
                self._percep_detect_full_range, self._percep_detect_decay_range)
            detected = torch.rand_like(dist) < detect_prob

            noise_std = soccer_util.compute_perception_noise_std(
                dist, self._percep_noise_dist_coef, self._percep_noise_base)
            noisy_pos = ball_pos[:, 0:2] + noise_std.unsqueeze(-1) * torch.randn_like(ball_pos[:, 0:2])

            latency = self._percep_latency_mean \
                + self._percep_latency_std * torch.randn_like(dist)
            latency = torch.clamp(latency, min=0.0)

            head = self._percep_buf_head[env_ids]
            self._percep_buf_pos[env_ids, head] = noisy_pos
            self._percep_buf_valid[env_ids, head] = detected
            self._percep_buf_deliver[env_ids, head] = self._time_buf[env_ids] + latency
            self._percep_buf_head[env_ids] = (head + 1) % self._percep_buf_slots
            self._percep_next_capture[env_ids] = self._percep_next_capture[env_ids] \
                + self._percep_period[env_ids]
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
        self._percep_buf_head[env_ids] = 0
        self._percep_ball_pos[env_ids] = self._get_ball_pos()[env_ids][:, 0:2]
        self._percep_ball_valid[env_ids] = True
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

        # head gaze (Frente G): true ball state, like every other reward term
        if (self._reward_head_gaze_w != 0.0):
            body_rot = self._engine.get_body_rot(char_id)
            head_pos = body_pos[:, self._fov_body_id, :]
            head_rot = body_rot[:, self._fov_body_id, :]
            gaze_r = soccer_util.compute_head_gaze_reward(head_pos, head_rot, ball_pos)
            aux_r += self._reward_head_gaze_w * gaze_r

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
        # fall termination penalty joins the aux stream once dones are known;
        # gated BEFORE ball events also become FAIL (out-of-bounds ends the
        # episode but pays no fall penalty)
        fall_mask = (self._done_buf == base_env.DoneFlags.FAIL.value)
        self._aux_reward_buf += self._reward_termination_w * fall_mask.float()

        done, soft = soccer_util.apply_ball_event_dones(
            self._done_buf, self._goal_scored_buf, self._ball_oob_buf,
            base_env.DoneFlags.NULL.value, base_env.DoneFlags.SUCC.value,
            base_env.DoneFlags.FAIL.value)
        self._done_buf[:] = done
        self._soft_done_buf[:] = soft
        return

    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)
        self._info["aux_reward"] = self._aux_reward_buf
        if (self.has_critic_obs()):
            # asymmetric critic (paper Table 2): the critic trains on the
            # TRUE ball state while the actor only ever sees the perceived
            # one. Fresh tensor every call: no aliasing with later mutations.
            self._info["critic_obs"] = self._compute_critic_obs()
        if (self._measurable_obs):
            self._info["recon_tar"] = self._compute_recon_tar()
        return

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
            return torch.cat([obs, ball_state[..., 2:4]], dim=-1)
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
        return soccer_util.compute_ball_state_local(
            self._engine.get_root_pos(char_id),
            self._engine.get_root_rot(char_id),
            self._get_ball_pos(),
            self._engine.get_root_vel(self._get_ball_id()))

    def get_recon_tar_size(self):
        assert self._measurable_obs
        return 4

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

    def _update_char_push(self):
        if (not self._char_push_enable):
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
            self._reset_char_placement(env_ids)
            # the placement moved/rotated the root, so the rigid body state
            # written by char_env's reset is stale; recompute it with FK or
            # key-body obs will mix the old pose with the new root position
            self._reset_char_rigid_body_state(env_ids)
            self._reset_ball(env_ids)
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
            self._prev_action[env_ids] = 0.0
            if (self._measurable_obs):
                # after prev_action is zeroed: the refilled history must match
                # the frame the actor sees on its first post-reset step
                self._meas_hist_buf[env_ids] = \
                    self._compute_measurable_frame(env_ids).unsqueeze(1)
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
        self._reset_envs(env_ids)

        n = len(env_ids)
        char_id = self._get_char_id()
        root_pos = self._init_root_pos.unsqueeze(0).repeat(n, 1)
        root_pos[:, 0:2] = self._field_offset[env_ids]
        root_pos[:, 2] += self._ground_z_offset
        root_rot = self._init_root_rot.unsqueeze(0).repeat(n, 1)
        dof_pos = self._init_dof_pos.unsqueeze(0).repeat(n, 1)
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
        ball_pos[:, 2] = self._ball_radius + self._ground_z_offset
        ball_rot = torch.zeros([n, 4], device=self._device, dtype=torch.float)
        ball_rot[:, 3] = 1.0
        self._engine.set_root_pos(env_ids, ball_id, ball_pos)
        self._engine.set_root_rot(env_ids, ball_id, ball_rot)
        self._engine.set_root_vel(env_ids, ball_id, zero_vel)
        self._engine.set_root_ang_vel(env_ids, ball_id, zero_vel)

        self._prev_ball_pos[env_ids] = ball_pos
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
                self._compute_measurable_frame(env_ids).unsqueeze(1)
        self._record_reset_prev_states(env_ids)
        self._prev_root_vel[env_ids] = self._engine.get_root_vel(char_id)[env_ids]
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
        self._reset_ball(env_ids, near=not self._ball_soft_reset_far)
        if (self._virtual_perception):
            self._reset_perception(env_ids)
        if (self._task_hist_steps > 0):
            self._task_hist_buf[env_ids] = self._compute_task_block(env_ids).unsqueeze(1)
            if (self._virtual_perception):
                self._critic_task_hist_buf[env_ids] = \
                    self._compute_task_block(env_ids, privileged=True).unsqueeze(1)
        if (self._measurable_obs):
            self._meas_hist_buf[env_ids] = \
                self._compute_measurable_frame(env_ids).unsqueeze(1)
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
        root_pos[:, 2] += self._ground_z_offset

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
        ball_pos[:, 2] = self._ball_radius + self._ground_z_offset

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
