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

import envs.steering_dr as steering_dr
import envs.steering_disc as steering_disc
import envs.steering_reward as steering_reward
import envs.steering_util as steering_util
import envs.task_steering_env as task_steering_env
import util.torch_util as torch_util

TASK_BLOCK_DIM = 5      # local_tar_dir (2), tar_speed (1), local_face_dir (2)
RECON_TAR_DIM = 3       # base-frame linear velocity


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
        self._aux_reward_enabled = (self._reward_foot_proximity_w != 0.0)

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
        self._meas_frame_dim = 6 + 3 * action_dim + TASK_BLOCK_DIM
        self._meas_hist_buf = torch.zeros(
            [num_envs, self._meas_hist_steps, self._meas_frame_dim],
            device=self._device, dtype=torch.float)

        if (self._aux_reward_enabled):
            self._aux_reward_buf = torch.zeros([num_envs], device=self._device,
                                               dtype=torch.float)
            char_id = self._get_char_id()
            self._reward_foot_body_ids = [
                self._engine.find_obj_body_id(char_id, name)
                for name in self._disc_foot_bodies]

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
        self._prev_action[:] = actions
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

        return task_steering_env.compute_steering_observations(root_rot, tar_dir,
                                                                tar_speed, face_dir)

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
        frame = self._compute_measurable_frame(env_ids)
        hist = self._meas_hist_buf if env_ids is None else self._meas_hist_buf[env_ids]
        return torch.cat([frame, hist.flatten(start_dim=1)], dim=-1)

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
        if (self._aux_reward_enabled):
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
        self._foot_prox_last = foot_prox
        self._aux_reward_buf[:] = self._reward_foot_proximity_w * foot_prox
        return

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

        frame = self._compute_measurable_frame()
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
            if (self._dr_enabled):
                # same reasoning for the delay buffer, which knows nothing
                # about episode boundaries
                self._engine.reset_action_delay(env_ids)
            frame = self._compute_measurable_frame(env_ids)
            self._meas_hist_buf[env_ids] = frame.unsqueeze(1)
        return
