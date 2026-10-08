"""Location task with the measurable observation: start standing, walk to a
target a fixed distance ahead, stop there.

Isolation experiment (2026-10-02). The steering task samples speed and
direction and trains on 16 motion clips with no fixed goal; AMP's own
locomotion tasks (Peng et al. 2021, appendix A) use a fixed objective. This env
keeps EVERYTHING of TaskSteeringMeasEnv -- measurable actor obs + history,
asymmetric critic, decoder, domain randomization, Table 4 and Booster Gym gait
terms -- and replaces only the goal:

  task block   the target in the heading frame (x, y): 2 dims instead of the
               steering command's 5 (frame 6 + 3D + 2)
  reward       AMP eq. 12 with a stopping mask (steering_reward.
               compute_location_reward)
  reset        always the first frame of one clip (the stand_to_walk start, a
               standing pose), never a random frame -- the task is "leave from
               standing"
  target       loc_tar_dist ahead of the character's heading at reset, never
               resampled inside the episode

The steering buffers of the parent (tar_dir, tar_speed, ...) still exist and
are still filled, because TaskSteeringEnv's constructor and reset need them;
nothing in this env reads them.
"""

import torch

import envs.steering_reward as steering_reward
import envs.task_location_env as task_location_env
import envs.task_steering_meas_env as task_steering_meas_env
import util.torch_util as torch_util


class TaskLocationMeasEnv(task_steering_meas_env.TaskSteeringMeasEnv):

    TASK_BLOCK_DIM = 2      # target (x, y) in the heading frame

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        self._loc_tar_dist = float(env_config.get("loc_tar_dist", 3.0))
        # 0.8, not AMP's 1 m/s: stand_to_walk peaks at 0.86 m/s and
        # walk_to_stand starts at 0.78 m/s (measured), so 1 m/s would ask for
        # a gait the reference does not contain
        self._loc_tar_speed = float(env_config.get("loc_tar_speed", 0.8))
        self._loc_pos_w = float(env_config.get("loc_pos_w", 0.7))
        self._loc_vel_w = float(env_config.get("loc_vel_w", 0.3))
        self._loc_pos_err_scale = float(env_config.get("loc_pos_err_scale", 0.5))
        self._loc_vel_err_scale = float(env_config.get("loc_vel_err_scale", 1.0))
        self._loc_stop_radius = float(env_config.get("loc_stop_radius", 0.3))
        # index, in the motion dataset, of the clip whose FIRST frame every
        # episode starts from (dataset order: stand_to_walk first)
        self._loc_reset_motion = int(env_config.get("loc_reset_motion", 0))
        assert not env_config.get("rand_reset", True), \
            "task_location_meas resets at t=0 of loc_reset_motion: set rand_reset: false"
        assert not env_config.get("gait_clock", False), \
            "the gait clock is not wired for the location task block"
        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    def _build_sim_tensors(self, config):
        # before super(): the parent sizes and fills the measurable history,
        # which already reads the target through _compute_task_block
        num_envs = self.get_num_envs()
        self._loc_tar_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        super()._build_sim_tensors(config)
        return

    def get_mirror_task_key(self):
        return "TaskLocationMeasEnv"

    # ------------------------------------------------------------------ reset

    def _sample_reset_motion_times(self, n):
        ids = torch.full([n], self._loc_reset_motion, device=self._device, dtype=torch.long)
        times = torch.zeros([n], device=self._device, dtype=torch.float)
        return ids, times

    def _sample_motion_times(self, n):
        """Times for the DISCRIMINATOR's demo windows (amp_env.fetch_disc_obs_demo).

        The parent returns zeros whenever rand_reset is False, which this task
        needs for its reset -- and which, inherited here, collapsed every demo
        window to t = 0 of each clip: the critic only ever saw the first pose
        of stand_to_walk and of walk_to_stand (measured: 1 distinct demo time in
        20000 draws; the first arm-L runs trained against that). Demos must
        cover the whole clips whatever the reset does, so they always sample.
        """
        motion_ids = self._motion_lib.sample_motions(n)
        motion_times = self._motion_lib.sample_time(motion_ids)
        return motion_ids, motion_times

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)
        if (len(env_ids) == 0):
            return
        # TaskSteeringEnv calls this on episode reset (after the character has
        # been placed) and on command resampling; with tar_change_time set past
        # the episode length only the first happens
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)[env_ids]
        root_rot = self._engine.get_root_rot(char_id)[env_ids]
        self._loc_tar_pos[env_ids] = compute_target_ahead(root_pos, root_rot,
                                                          self._loc_tar_dist)
        return

    # ------------------------------------------------------------- task block

    def _compute_task_block(self, env_ids=None):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        tar_pos = self._loc_tar_pos
        if (env_ids is not None):
            root_pos, root_rot, tar_pos = root_pos[env_ids], root_rot[env_ids], tar_pos[env_ids]
        return task_location_env.compute_location_observations(root_pos, root_rot, tar_pos)

    # ----------------------------------------------------------------- reward

    def _update_reward(self):
        # NOT super(): the parent's chain computes the steering reward
        char_id = self._get_char_id()
        self._reward_buf[:] = steering_reward.compute_location_reward(
            self._engine.get_root_pos(char_id), self._prev_root_pos, self._loc_tar_pos,
            self._loc_tar_speed, self._engine.get_timestep(),
            self._loc_pos_err_scale, self._loc_vel_err_scale,
            self._loc_pos_w, self._loc_vel_w, self._loc_stop_radius)
        if (self._reg_diag_enabled):
            self._cache_aux_reward()
        return

    def get_target_pos(self):
        return self._loc_tar_pos


@torch.jit.script
def compute_target_ahead(root_pos, root_rot, dist):
    # type: (Tensor, Tensor, float) -> Tensor
    """root + dist along the HEADING (yaw only) direction, on the ground plane."""
    heading = torch_util.calc_heading_quat(root_rot)
    fwd = torch.zeros_like(root_pos)
    fwd[..., 0] = 1.0
    fwd = torch_util.quat_rotate(heading, fwd)
    tar = root_pos + dist * fwd
    tar[..., 2] = 0.0
    return tar
