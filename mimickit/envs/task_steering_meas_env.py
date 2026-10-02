"""Steering with the paper's observation split (arXiv:2511.03996, Table 3).

Same task, rewards and command sampling as TaskSteeringEnv -- it subclasses it
so the two stay one variable apart, since the baseline is this experiment's
control. What differs is only the observation:

  actor    [current frame | H past frames], frame = projected gravity (3),
           base angular velocity (3), joint offsets (D), joint velocities (D),
           previous action (D), steering command (5). Nothing here needs more
           than the T1's IMU, its joint encoders and the policy's own output.
  critic   the same frame plus base-frame linear velocity (3) and base height
           (1), which exist only in simulation. No history: the critic sees the
           true state and has nothing to estimate.
  decoder  reconstructs the base-frame linear velocity from the encoder latent
           (training only, never reaches the actor).

  disc     with disc_obs_table3 on, the Disc. column of Table 3 (projected
           gravity, angular velocity, joint offsets, joint velocities, linear
           velocity, feet in the torso frame) instead of MimicKit's generic
           imitation features. Off by default, so the E1 configs keep the old
           1970-dim window; see envs/steering_disc.py for the two differences.
"""

import gymnasium.spaces as spaces
import numpy as np
import torch

import envs.diag_util as diag_util
import envs.steering_dr as steering_dr
import envs.steering_disc as steering_disc
import envs.steering_reward as steering_reward
import envs.steering_util as steering_util
import envs.task_steering_env as task_steering_env
import util.torch_util as torch_util

TASK_BLOCK_DIM = 5      # local_tar_dir (2), tar_speed (1), local_face_dir (2)
RECON_TAR_DIM = 3       # base-frame linear velocity

# DOF groups for the Table 4 action-rate terms, which weigh the head (-15) and
# the legs (-1) differently and give the arms no term at all. Named by BODY,
# resolved through the char model, so a renamed link raises instead of silently
# weighing the wrong joints. Overridable per config for another embodiment.
DEFAULT_HEAD_BODIES = ["aahead_yaw_link", "aahead_pitch_link"]
DEFAULT_LEG_BODIES = ["{}_{}_link".format(side, joint)
                      for side in ("left", "right")
                      for joint in ("hip_pitch", "hip_roll", "hip_yaw",
                                    "knee_pitch", "ankle_pitch", "ankle_roll")]
# "collision on body parts except the feet" -- the foot body is ankle_roll,
# which is where the foot collision box lives in t1.xml.
DEFAULT_COLLISION_EXEMPT = ["left_ankle_roll_link", "right_ankle_roll_link"]


class TaskSteeringMeasEnv(task_steering_env.TaskSteeringEnv):

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        self._meas_hist_steps = int(env_config.get("measurable_hist_steps", 30))
        assert self._meas_hist_steps > 0, \
            "measurable_hist_steps must be > 0: the encoder has nothing to " \
            "compress otherwise, and linear velocity is not recoverable from " \
            "a single frame"
        # read by ppo_agent to pick the measurable mirror map
        self._measurable_obs = True

        # actor-only sensor noise (paper, Appendix A: "only the actor's
        # observations ... were corrupted with simulated sensor noise"). All
        # stds default to 0, and a zero std leaves the env bit-identical.
        self._obs_noise_cfg = {k: env_config[k]
                               for k in steering_util.OBS_NOISE_KEYS + ("obs_noise_scale",)
                               if k in env_config}

        # domain randomization (Table 2). Off by default so this env stays
        # bit-identical to the E1 runs unless a config asks for it.
        self._dr_enabled = bool(env_config.get("domain_randomization", False))
        self._dr_ranges = steering_dr.load_ranges(env_config) if self._dr_enabled else None
        self._dr_seed = env_config.get("dr_seed", None)
        self._dr_foot_bodies = list(env_config.get(
            "dr_foot_bodies", ["left_ankle_roll_link", "right_ankle_roll_link"]))
        self._dr_samples = []

        # discriminator observation (Table 3, Disc. column). Off by default so
        # the env keeps producing the generic 197-dim frames the E1 runs used.
        self._disc_table3 = bool(env_config.get("disc_obs_table3", False))
        self._disc_foot_bodies = list(env_config.get(
            "disc_foot_bodies", ["left_ankle_roll_link", "right_ankle_roll_link"]))

        # regularization rewards (Table 4). Weight 0 keeps the env bit-identical
        # to the runs that came before, so this is inert unless a config asks.
        self._reward_foot_proximity_w = float(env_config.get("reward_foot_proximity_w", 0.0))
        self._foot_proximity_min_dist = float(env_config.get("foot_proximity_min_dist", 0.2))
        self._reward_head_action_rate_w = float(env_config.get("reward_head_action_rate_w", 0.0))
        self._reward_leg_action_rate_w = float(env_config.get("reward_leg_action_rate_w", 0.0))
        self._reward_joint_limit_w = float(env_config.get("reward_joint_limit_w", 0.0))
        self._reward_base_accel_w = float(env_config.get("reward_base_accel_w", 0.0))
        self._reward_collision_w = float(env_config.get("reward_collision_w", 0.0))

        # gait terms from Booster Gym (see steering_reward, "Gait terms"). Weight
        # 0 keeps the env bit-identical; each is measured raw before a weight is
        # picked, because Booster Gym's weights belong to a different stream.
        self._reward_feet_slip_w = float(env_config.get("reward_feet_slip_w", 0.0))
        self._reward_feet_yaw_diff_w = float(env_config.get("reward_feet_yaw_diff_w", 0.0))
        self._reward_feet_lat_dist_w = float(env_config.get("reward_feet_lat_dist_w", 0.0))
        # Booster Gym uses ref 0.2 m. NOT a safe default here: the reference
        # motions walk with the feet 0.10-0.16 m apart sideways (p50, measured),
        # so 0.2 would fight the discriminator on its own data. Configs set it.
        self._feet_lat_dist_ref = float(env_config.get("feet_lat_dist_ref", 0.2))
        self._feet_lat_dist_cap = float(env_config.get("feet_lat_dist_cap", 0.1))
        # Booster Gym's contact rule: the sole within 1 cm of the ground
        self._feet_contact_height = float(env_config.get("feet_contact_height", 0.01))
        self._foot_box_half = tuple(env_config.get("foot_box_half",
                                                   steering_util.DEFAULT_FOOT_BOX_HALF))
        self._foot_box_pos = tuple(env_config.get("foot_box_pos",
                                                  steering_util.DEFAULT_FOOT_BOX_POS))

        # Booster Gym's commanded gait clock + feet_swing. OFF by default: it
        # adds two dims to the actor frame, which changes the obs contract, the
        # export and the deploy side. The frequency range defaults to the
        # reference motions' measured cadence (1.31-2.27 Hz over the 16 clips of
        # dataset_t1_locomotion_wrturn), not Booster Gym's U(1, 2), so that the
        # clock and the discriminator do not ask for two different cadences.
        self._gait_clock = bool(env_config.get("gait_clock", False))
        self._gait_clock_freq = tuple(env_config.get("gait_clock_freq", [1.3, 2.3]))
        # below this commanded speed the clock stops (Booster Gym's "still")
        self._gait_clock_still_speed = float(env_config.get("gait_clock_still_speed", 0.1))
        self._reward_feet_swing_w = float(env_config.get("reward_feet_swing_w", 0.0))
        self._feet_swing_period = float(env_config.get("feet_swing_period", 0.2))
        assert self._gait_clock or self._reward_feet_swing_w == 0.0, \
            "reward_feet_swing_w needs gait_clock: true -- it has no clock to read"
        self._task_dim = TASK_BLOCK_DIM + (2 if self._gait_clock else 0)

        # 0 on purpose: the reference motions rest the knee at exactly 0.0 rad,
        # its hard lower limit, for part of every stride (see
        # steering_reward.compute_joint_limit_penalty). Any guard band above 0
        # therefore penalizes the pose the discriminator is paying for.
        self._joint_limit_margin = float(env_config.get("joint_limit_margin", 0.0))
        # False is the paper-literal reading ("joint positions exceeding
        # limits") and is provably inert here: the engine clamps dof_pos, so
        # the measured excursion is 1.5e-5 rad. The COMMANDED target is not
        # clamped by anything -- the action bound is +-1.4x the joint range --
        # and the same rollout measures 0.279 rad of excursion on it, ~18000x
        # more. True penalizes that instead, which is the only version of this
        # term that can act at all in a position-controlled setup.
        self._joint_limit_on_action = bool(env_config.get("joint_limit_on_action", False))
        # 1 N, not the 0.1 N of the fall check: that one reads ground forces
        # the engine has already height-filtered, while these are raw and 0.1 N
        # is inside resting-contact noise.
        self._collision_force_thresh = float(env_config.get("collision_force_thresh", 1.0))

        self._head_dof_bodies = list(env_config.get("head_dof_bodies", DEFAULT_HEAD_BODIES))
        self._leg_dof_bodies = list(env_config.get("leg_dof_bodies", DEFAULT_LEG_BODIES))
        self._collision_exempt_bodies = list(env_config.get(
            "collision_exempt_bodies", DEFAULT_COLLISION_EXEMPT))

        # every weight has to flip the gate. An OR that forgot one would leave
        # _aux_reward_buf unallocated and info["aux_reward"] unpublished, and
        # that config would train as if the term were not there.
        self._reg_weights = (self._reward_foot_proximity_w,
                             self._reward_head_action_rate_w,
                             self._reward_leg_action_rate_w,
                             self._reward_joint_limit_w,
                             self._reward_base_accel_w,
                             self._reward_collision_w,
                             self._reward_feet_slip_w,
                             self._reward_feet_yaw_diff_w,
                             self._reward_feet_lat_dist_w,
                             self._reward_feet_swing_w)
        self._aux_reward_enabled = any(w != 0.0 for w in self._reg_weights)

        # measure-before-you-spend: logs every raw term with ALL weights at 0,
        # so the magnitudes can be read off a short run before a weight is
        # picked. The reward path stays untouched -- _update_info still does
        # not publish aux_reward, so the agent never sees a second stream.
        self._reg_diag_enabled = self._aux_reward_enabled or \
            bool(env_config.get("reward_reg_diagnostics", False))

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    # ------------------------------------------------------ domain randomization

    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)
        if (self._dr_enabled):
            self._randomize_env_props(env_id)
        return

    def _get_dr_rng(self):
        if (not hasattr(self, "_dr_rng")):
            # np.random is already seeded from --rand_seed (run.py:91-101), so
            # deriving from it keeps the draw reproducible; dr_seed overrides it
            # so the holdout env can randomize differently from training.
            seed = self._dr_seed if self._dr_seed is not None else np.random.randint(2**31 - 1)
            self._dr_rng = np.random.RandomState(int(seed))
        return self._dr_rng

    def _randomize_env_props(self, env_id):
        """Per-env static randomization, applied at build time (before the sim
        is initialized, which the Isaac Gym mass/shape APIs require). The motor
        gains and the action delay are NOT applied here - their tensors only
        exist after initialize_sim; see _apply_dr_runtime."""
        char_id = self._get_char_id()
        num_bodies = self._engine.get_obj_num_bodies(char_id)
        num_dofs = self._engine.get_obj_num_dofs(char_id)

        params = steering_dr.sample_env_params(self._get_dr_rng(), self._dr_ranges,
                                               num_bodies, num_dofs)
        self._dr_samples.append(params)

        self._engine.scale_obj_masses(env_id, char_id, params["mass_scales"],
                                      params["com_offsets"])

        foot_ids = [self._engine.find_obj_body_id(char_id, name)
                    for name in self._dr_foot_bodies]
        # compliance is written too, with the caveat recorded in steering_dr:
        # it reads back as 0.0 while friction and restitution read back correct
        self._engine.set_obj_shape_props(env_id, char_id,
                                         friction=params["foot_friction"],
                                         restitution=params["foot_restitution"],
                                         compliance=params["foot_compliance"],
                                         body_ids=foot_ids)
        return

    def _apply_dr_runtime(self):
        """The half of Table 2 that needs the post-initialize_sim tensors."""
        char_id = self._get_char_id()
        num_envs = self.get_num_envs()

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

        # what the critic observes (Table 3, "Mass randomization")
        self._dr_torso_params = torch.as_tensor(
            np.stack([p["torso_params"] for p in self._dr_samples]),
            device=self._device, dtype=torch.float)
        assert self._dr_torso_params.shape == (num_envs, steering_dr.TORSO_PARAM_DIM)
        return

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)

        num_envs = self.get_num_envs()
        action_dim = self._action_space.shape[0]

        self._prev_action = torch.zeros([num_envs, action_dim], device=self._device,
                                        dtype=torch.float)
        # the clock buffers exist even when it is off, so the task block code
        # has one path; with gait_clock off they are never read
        self._gait_phase = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._gait_freq = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._meas_frame_dim = 6 + 3 * action_dim + self._task_dim
        self._meas_hist_buf = torch.zeros(
            [num_envs, self._meas_hist_steps, self._meas_frame_dim],
            device=self._device, dtype=torch.float)

        self._obs_noise_std = steering_util.build_obs_noise_std(
            self._obs_noise_cfg, action_dim, self._task_dim).to(self._device)
        assert self._obs_noise_std.shape[0] == self._meas_frame_dim
        self._obs_noise_on = bool(torch.any(self._obs_noise_std > 0))

        if (self._reg_diag_enabled):
            self._aux_reward_buf = torch.zeros([num_envs], device=self._device,
                                               dtype=torch.float)
            self._diag = diag_util.DiagWindow(self._device)
            char_id = self._get_char_id()
            self._reward_foot_body_ids = [
                self._engine.find_obj_body_id(char_id, name)
                for name in self._disc_foot_bodies]

            self._dt = self._engine.get_timestep()

            # DOF ids come from the char model; BODY ids for contact forces come
            # from the engine. The two orderings are not the same and mixing
            # them is silent.
            self._head_dof_ids = torch.tensor(
                steering_util.build_dof_group_ids(self._kin_char_model,
                                                  self._head_dof_bodies),
                device=self._device, dtype=torch.long)
            self._leg_dof_ids = torch.tensor(
                steering_util.build_dof_group_ids(self._kin_char_model,
                                                  self._leg_dof_bodies),
                device=self._device, dtype=torch.long)

            # read once from env 0: domain randomization does not randomize
            # joint limits (steering_dr.DEFAULT_RANGES has no such key), so
            # they are the same in every env.
            dof_low, dof_high = self._engine.get_obj_dof_limits(0, char_id)
            self._soft_dof_low = torch.as_tensor(
                np.asarray(dof_low), device=self._device,
                dtype=torch.float) + self._joint_limit_margin
            self._soft_dof_high = torch.as_tensor(
                np.asarray(dof_high), device=self._device,
                dtype=torch.float) - self._joint_limit_margin
            assert torch.all(self._soft_dof_low < self._soft_dof_high), \
                "joint_limit_margin {} is wider than some joint's range".format(
                    self._joint_limit_margin)

            exempt = set(self._engine.find_obj_body_id(char_id, name)
                         for name in self._collision_exempt_bodies)
            num_bodies = self._engine.get_obj_num_bodies(char_id)
            self._collision_body_ids = torch.tensor(
                [b for b in range(num_bodies) if b not in exempt],
                device=self._device, dtype=torch.long)

            self._prev_root_vel = torch.zeros([num_envs, 3], device=self._device,
                                              dtype=torch.float)
            self._foot_box_corners = steering_util.build_foot_box_corners(
                self._foot_box_half, self._foot_box_pos).to(self._device)
            assert len(self._reward_foot_body_ids) == 2, \
                "gait terms assume two feet, left then right"
            self._head_act_rate_buf = torch.zeros([num_envs], device=self._device,
                                                  dtype=torch.float)
            self._leg_act_rate_buf = torch.zeros([num_envs], device=self._device,
                                                 dtype=torch.float)

        if (self._disc_table3):
            # before _build_data_buffers, which sizes the disc obs from a demo
            char_id = self._get_char_id()
            self._disc_foot_body_ids = [
                self._engine.find_obj_body_id(char_id, name)
                for name in self._disc_foot_bodies]

        # runs here, not in _build_env: the gain and command tensors only exist
        # after the engine's initialize_sim
        if (self._dr_enabled):
            self._apply_dr_runtime()
        return

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        if (self._reg_diag_enabled):
            self._cache_pre_physics_terms(actions)
        self._prev_action[:] = actions
        return

    def _cache_pre_physics_terms(self, actions):
        """Everything that needs a_t and a_{t-1} together, or the velocity from
        BEFORE the physics step.

        This is the only moment both exist. By the time _update_reward runs,
        _prev_action below has already been overwritten with a_t, so an action
        rate computed there is identically zero -- and a term that is always
        zero is indistinguishable from a term that does not help.
        super()._pre_physics_step only writes the command, it does not advance
        the simulation, so get_root_vel here is still the pre-step velocity
        that the acceleration needs.
        """
        # the first step of an episode has no predecessor: _prev_action was
        # zeroed on reset and our action is an ABSOLUTE joint target, so the
        # "rate" would be ||a_1||^2 of a whole pose. Left in, it scales with
        # 1/episode_length and would make the term move when episode_length
        # changes and nothing else does.
        first = (self._timestep_buf == 0)

        head = steering_reward.compute_action_rate_penalty(
            actions.index_select(-1, self._head_dof_ids),
            self._prev_action.index_select(-1, self._head_dof_ids))
        leg = steering_reward.compute_action_rate_penalty(
            actions.index_select(-1, self._leg_dof_ids),
            self._prev_action.index_select(-1, self._leg_dof_ids))

        zero = torch.zeros_like(head)
        self._head_act_rate_buf[:] = torch.where(first, zero, head)
        self._leg_act_rate_buf[:] = torch.where(first, zero, leg)

        char_id = self._get_char_id()
        self._prev_root_vel[:] = self._engine.get_root_vel(char_id)
        return

    def _apply_action(self, actions):
        if (not self._dr_enabled):
            super()._apply_action(actions)
            return

        # motor bias (Table 2) is the encoder's zero being off, so it lands on
        # the joint TARGET and AFTER the clip -- folded in before it, the clip
        # would eat the bias at the limits. Mirrors char_env._apply_action
        # otherwise.
        char_id = self._get_char_id()
        clip_action = torch.minimum(torch.maximum(actions, self._action_bound_low),
                                    self._action_bound_high)
        self._engine.set_cmd(char_id, clip_action + self._motor_bias)
        return

    # ------------------------------------------------------------- observation

    def _compute_task_block(self, env_ids=None):
        """The 5-dim steering command block, identical to the baseline's."""
        char_id = self._get_char_id()
        root_rot = self._engine.get_root_rot(char_id)
        tar_dir = self._tar_dir
        tar_speed = self._tar_speed
        face_dir = self._face_dir

        if (env_ids is not None):
            root_rot = root_rot[env_ids]
            tar_dir = tar_dir[env_ids]
            tar_speed = tar_speed[env_ids]
            face_dir = face_dir[env_ids]

        block = task_steering_env.compute_steering_observations(root_rot, tar_dir,
                                                                 tar_speed, face_dir)
        if (not self._gait_clock):
            return block
        phase, freq = self._gait_phase, self._gait_freq
        if (env_ids is not None):
            phase, freq = phase[env_ids], freq[env_ids]
        return torch.cat([block, steering_util.compute_gait_clock_obs(phase, freq)], dim=-1)

    def get_mirror_task_key(self):
        """mirror_util.TASK_OBS_MIRROR entry for this config's task block."""
        return "TaskSteeringMeasEnvClock" if self._gait_clock else type(self).__name__

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)
        if (self._gait_clock and len(env_ids) > 0):
            # resampled with the command, as Booster Gym does; stopped when the
            # commanded speed is a stand
            lo, hi = self._gait_clock_freq
            f = lo + (hi - lo) * torch.rand(len(env_ids), device=self._device)
            still = self._tar_speed[env_ids] < self._gait_clock_still_speed
            self._gait_freq[env_ids] = torch.where(still, torch.zeros_like(f), f)
        return

    def _compute_measurable_frame(self, env_ids=None):
        """One measurable frame: [proprio (6 + 3D) | steering command (5)]."""
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

        proprio = steering_util.compute_proprio_frame(root_rot, root_ang_vel, dof_pos,
                                                      dof_vel, prev_action,
                                                      self._init_dof_pos)
        return torch.cat([proprio, self._compute_task_block(env_ids)], dim=-1)

    def _compute_obs(self, env_ids=None):
        # the history is rolled once per step in _update_task (and refilled on
        # reset), NEVER here: _compute_obs doubles as a shape probe for
        # get_obs_space() and is called more than once per step
        hist = self._meas_hist_buf if env_ids is None else self._meas_hist_buf[env_ids]
        if (self._obs_noise_on):
            # the noisy frame was drawn ONCE, when it entered the history.
            # Re-drawing here would hand the actor a current frame that
            # differs from hist[-1] -- the deploy side sends the same frame in
            # both places -- and a different one on every call.
            frame = hist[:, -1]
        else:
            frame = self._compute_measurable_frame(env_ids)
        return torch.cat([frame, hist.flatten(start_dim=1)], dim=-1)

    def _compute_actor_frame(self, env_ids=None):
        """The frame the actor receives: the measurable frame, corrupted with
        sensor noise when configured. Called exactly once per step (history
        roll) and once per reset (refill). The critic never goes through here:
        _compute_critic_obs calls _compute_measurable_frame, which is clean."""
        return steering_util.apply_obs_noise(self._compute_measurable_frame(env_ids),
                                             self._obs_noise_std)

    def get_measurable_frame_dim(self):
        return self._meas_frame_dim

    def get_measurable_hist_steps(self):
        return self._meas_hist_steps

    # --------------------------------------------------- privileged / decoder

    def has_critic_obs(self):
        return True

    def _compute_critic_obs(self):
        """Measurable frame plus the simulator-only state (paper Table 3, the
        Critic-only rows). DIFFERENT SHAPE from the actor obs; consumers must
        size the critic from get_critic_obs_space()."""
        char_id = self._get_char_id()
        priv = steering_util.compute_privileged_block(
            self._engine.get_root_pos(char_id),
            self._engine.get_root_rot(char_id),
            self._engine.get_root_vel(char_id))
        blocks = [self._compute_measurable_frame(), priv]
        if (self._dr_enabled):
            # Table 3, "Mass randomization": mass and CoM of the base link,
            # which only exists to be observed once Table 2 randomizes it
            blocks.append(self._dr_torso_params)
        return torch.cat(blocks, dim=-1)

    def get_critic_obs_space(self):
        obs = self._compute_critic_obs()
        return spaces.Box(low=-np.inf, high=np.inf, shape=list(obs.shape[1:]),
                          dtype=torch_util.torch_dtype_to_numpy(obs.dtype))

    def _compute_recon_tar(self):
        char_id = self._get_char_id()
        return steering_util.compute_root_lin_vel_b(
            self._engine.get_root_rot(char_id),
            self._engine.get_root_vel(char_id))

    def get_recon_tar_size(self):
        return RECON_TAR_DIM

    def _update_reward(self):
        super()._update_reward()
        # the looser gate: with every weight at 0 this still fills the buffer
        # and the diagnostics, but _update_info below does not publish the key,
        # so the agent path stays bit-identical. That is the measurement mode.
        if (self._reg_diag_enabled):
            self._cache_aux_reward()
        return

    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)
        if (self._aux_reward_enabled):
            # _update_info runs BEFORE _update_reward (sim_env._post_physics_step),
            # so this publishes the buffer before _cache_aux_reward fills it.
            # That is safe ONLY because _cache_aux_reward mutates in place
            # (buf[:] = ...): the agent reads this reference after step() has
            # returned, by which point the values are current. Rebinding the
            # buffer there instead would leave stale zeros here, silently.
            self._info["aux_reward"] = self._aux_reward_buf
        # always full-N regardless of env_ids: the agent records these straight
        # into a [num_envs, ...] buffer, on reset as well as on step
        self._info["critic_obs"] = self._compute_critic_obs()
        self._info["recon_tar"] = self._compute_recon_tar()
        return

    # --------------------------------------------------- regularization reward

    def _cache_aux_reward(self):
        """Table 4 regularization terms, published as info["aux_reward"].

        mcwamp_agent picks this up on its own (it checks for the key and adds
        it to the auxiliary critic stream), so nothing changes agent-side. The
        goal critic keeps seeing only the steering task reward."""
        char_id = self._get_char_id()
        body_pos = self._engine.get_body_pos(char_id)
        left = body_pos[:, self._reward_foot_body_ids[0], :]
        right = body_pos[:, self._reward_foot_body_ids[1], :]

        foot_prox = steering_reward.compute_foot_proximity_penalty(
            left, right, self._foot_proximity_min_dist)

        dof_pos = self._engine.get_dof_pos(char_id)
        pos_exc = steering_reward.compute_joint_limit_penalty(
            dof_pos, self._soft_dof_low, self._soft_dof_high)
        # _prev_action holds a_t here: _pre_physics_step overwrote it with this
        # step's action before the physics ran. That is the target the policy
        # asked for, which is the quantity the gradient can actually shape.
        act_exc = steering_reward.compute_joint_limit_penalty(
            self._prev_action, self._soft_dof_low, self._soft_dof_high)
        joint_lim = act_exc if self._joint_limit_on_action else pos_exc

        base_accel = steering_reward.compute_base_accel_penalty(
            self._engine.get_root_vel(char_id), self._prev_root_vel, self._dt)

        # the RAW contact forces. get_ground_contact_forces zeroes every body
        # above ground_contact_height (0.3 m), which is exactly where an
        # arm-into-hip self-contact lives -- it is also why the existing fall
        # termination cannot see one.
        forces = self._engine.get_contact_forces(char_id)
        collision = steering_reward.compute_collision_penalty(
            forces, self._collision_body_ids, self._collision_force_thresh)

        # --- Booster Gym gait terms ------------------------------------------
        foot_ids = self._reward_foot_body_ids
        f_pos = body_pos[:, foot_ids, :]
        f_rot = self._engine.get_body_rot(char_id)[:, foot_ids, :]
        f_vel = self._engine.get_body_vel(char_id)[:, foot_ids, :]
        f_ang = self._engine.get_body_ang_vel(char_id)[:, foot_ids, :]
        sole_z, cp_vel = steering_util.compute_sole_contact_state(
            f_pos, f_rot, f_vel, f_ang, self._foot_box_corners)
        contact_geo = sole_z < self._feet_contact_height
        feet_slip = steering_reward.compute_feet_slip_penalty(cp_vel, contact_geo)

        root_rot = self._engine.get_root_rot(char_id)
        base_yaw = torch_util.quat_to_euler_xyz(root_rot)[..., 2]
        foot_euler = torch_util.quat_to_euler_xyz(f_rot)
        foot_yaw = foot_euler[..., 2]
        feet_yaw_diff = steering_reward.compute_feet_yaw_diff_penalty(foot_yaw)
        feet_lat = steering_reward.compute_feet_lateral_distance_penalty(
            base_yaw, left, right, self._feet_lat_dist_ref, self._feet_lat_dist_cap)
        feet_swing = steering_reward.compute_feet_swing_reward(
            self._gait_phase, self._gait_freq, contact_geo, self._feet_swing_period)

        # one in-place write at the end. _update_info published this buffer's
        # REFERENCE before _update_reward ran, so rebinding it here would leave
        # stale zeros there and nothing would report it.
        self._aux_reward_buf[:] = (
            self._reward_foot_proximity_w * foot_prox
            + self._reward_head_action_rate_w * self._head_act_rate_buf
            + self._reward_leg_action_rate_w * self._leg_act_rate_buf
            + self._reward_joint_limit_w * joint_lim
            + self._reward_base_accel_w * base_accel
            + self._reward_collision_w * collision
            + self._reward_feet_slip_w * feet_slip
            + self._reward_feet_yaw_diff_w * feet_yaw_diff
            + self._reward_feet_lat_dist_w * feet_lat
            + self._reward_feet_swing_w * feet_swing)
        self._diag.add_mean("rew_feet_swing", self._reward_feet_swing_w * feet_swing)
        self._diag.add_mean("feet_swing", feet_swing)

        self._diag.add_mean("rew_feet_slip", self._reward_feet_slip_w * feet_slip)
        self._diag.add_mean("rew_feet_yawdiff", self._reward_feet_yaw_diff_w * feet_yaw_diff)
        self._diag.add_mean("rew_feet_latdist", self._reward_feet_lat_dist_w * feet_lat)
        self._diag.add_mean("feet_slip_cp", feet_slip)
        self._diag.add_mean("feet_yaw_diff", feet_yaw_diff)
        self._diag.add_mean("feet_lat_pen", feet_lat)
        # metrics, not rewards: what the terms are meant to move
        self._diag.add_mean("toe_in_mean", steering_reward.compute_toe_in(
            base_yaw, foot_yaw).mean(dim=-1))
        self._diag.add_mean("feet_lat_dist_m", steering_reward.compute_feet_lateral_distance(
            base_yaw, left, right))
        self._diag.add_mean("feet_yaw_mean", steering_reward.compute_feet_yaw_mean_penalty(
            base_yaw, foot_yaw))

        # Weighted AND raw, per term. Weighted alone cannot tell "the weight is
        # too small" from "the signal is not there"; raw alone cannot tell
        # whether the term reaches the critic. Inferring either from a shift in
        # Aux_Reward_Mean does not work: that total also carries the style
        # reward, which moves on its own.
        # Names stay <= 18 chars. The logger pads columns to 25 and prefixes
        # them with Train_/Test_, so a longer name overflows the pad and runs
        # into the next column, breaking the whitespace parsing that
        # compare_train_logs.py depends on.
        self._diag.step()
        self._diag.add_mean("rew_foot_prox", self._reward_foot_proximity_w * foot_prox)
        self._diag.add_mean("rew_head_actrate",
                            self._reward_head_action_rate_w * self._head_act_rate_buf)
        self._diag.add_mean("rew_leg_actrate",
                            self._reward_leg_action_rate_w * self._leg_act_rate_buf)
        self._diag.add_mean("rew_joint_limit", self._reward_joint_limit_w * joint_lim)
        self._diag.add_mean("rew_base_accel", self._reward_base_accel_w * base_accel)
        self._diag.add_mean("rew_collision", self._reward_collision_w * collision)

        self._diag.add_mean("head_act_rate", self._head_act_rate_buf)
        self._diag.add_mean("leg_act_rate", self._leg_act_rate_buf)
        # BOTH, always: whichever one the reward uses, the other says whether
        # the choice mattered. pos_exc reading ~0 while act_exc reads ~0.3 is
        # the whole reason joint_limit_on_action exists.
        self._diag.add_mean("joint_lim_exc", pos_exc)
        self._diag.add_mean("act_lim_exc", act_exc)
        self._diag.add_mean("base_accel_sq", base_accel)
        self._diag.add_mean("collision_rate", collision)
        # how MANY bodies are in contact, not whether any is. A count pinned
        # near K means two collision primitives permanently interpenetrate,
        # which the binary term cannot distinguish from a real constant event.
        self._diag.add_mean("collision_bodies", steering_reward.compute_collision_count(
            forces, self._collision_body_ids, self._collision_force_thresh))
        self._diag.add_mean("foot_dist_planar",
                            torch.linalg.norm(left[..., 0:2] - right[..., 0:2], dim=-1))
        return

    def record_diagnostics(self):
        diags = dict(super().record_diagnostics())
        if (self._reg_diag_enabled):
            means, _ = self._diag.pop()
            diags.update(means)
        return diags

    # -------------------------------------------------------- discriminator

    def _build_table3_disc_obs(self, root_pos, root_rot, root_vel, root_ang_vel,
                               joint_rot, dof_vel, body_pos):
        """The Disc. column of Table 3, from the window the AMP buffers hold.

        Both callers below reach this with the same seven tensors - one from
        the simulator, one from the motion library - which is what makes the
        two sides of the discriminator comparable."""
        dof_pos = self._kin_char_model.rot_to_dof(joint_rot)
        foot_pos = body_pos[..., self._disc_foot_body_ids, :]
        return steering_disc.compute_disc_obs(
            root_pos=root_pos, root_rot=root_rot, root_vel=root_vel,
            root_ang_vel=root_ang_vel, dof_pos=dof_pos, dof_vel=dof_vel,
            foot_pos=foot_pos, init_dof_pos=self._init_dof_pos)

    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        if (not self._disc_table3):
            return super()._compute_disc_obs_demo(motion_ids, motion_times0)

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos = \
            self._fetch_disc_demo_data(motion_ids, motion_times0)
        return self._build_table3_disc_obs(root_pos, root_rot, root_vel,
                                           root_ang_vel, joint_rot, dof_vel, body_pos)

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
        return

    # ----------------------------------------------------------- history roll

    def _update_task(self):
        # after super(): it may resample the command this step, and the newest
        # history slot must hold the frame the actor was actually given
        super()._update_task()

        if (self._gait_clock):
            # advanced BEFORE the frame is built, so the clock in the history is
            # the clock the reward reads this step
            self._gait_phase[:] = torch.fmod(
                self._gait_phase + self._engine.get_timestep() * self._gait_freq, 1.0)

        frame = self._compute_actor_frame()
        self._meas_hist_buf[:, :-1] = self._meas_hist_buf[:, 1:].clone()
        self._meas_hist_buf[:, -1] = frame
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            # order matters: zero the previous action BEFORE filling the
            # history, otherwise every episode starts with H frames carrying
            # the last action of the previous one
            self._prev_action[env_ids] = 0.0
            if (self._gait_clock):
                # Booster Gym never resets the phase; a random start does the
                # same job here without all envs stepping in lockstep
                self._gait_phase[env_ids] = torch.rand(len(env_ids), device=self._device)
            if (self._dr_enabled):
                # same reasoning for the delay buffer, which knows nothing
                # about episode boundaries
                self._engine.reset_action_delay(env_ids)
            if (self._reg_diag_enabled):
                # super()._reset_envs has already teleported the character to a
                # motion frame, so this re-seeds from the NEW velocity. A stale
                # value would make the first step of every episode pay
                # |v_new - v_old| / dt -- at 2.5 m/s and dt = 1/30 that is
                # ~5600, a constant bias that reads like the term working.
                char_id = self._get_char_id()
                self._prev_root_vel[env_ids] = self._engine.get_root_vel(char_id)[env_ids]
                self._head_act_rate_buf[env_ids] = 0.0
                self._leg_act_rate_buf[env_ids] = 0.0
            frame = self._compute_actor_frame(env_ids)
            self._meas_hist_buf[env_ids] = frame.unsqueeze(1)
        return
