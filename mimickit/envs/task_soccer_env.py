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

        self._reward_ball_approach_w = float(env_config.get("reward_ball_approach_w", 50.0))

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
        return

    def _get_ball_id(self):
        return self._ball_id

    def _get_ball_pos(self):
        return self._engine.get_root_pos(self._get_ball_id())

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._record_prev_states()
        return

    def _record_prev_states(self):
        char_id = self._get_char_id()
        self._prev_root_pos[:] = self._engine.get_root_pos(char_id)
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

    def _update_reward(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        ball_pos = self._get_ball_pos()

        approach_r = soccer_util.compute_ball_approach_reward(root_pos, self._prev_root_pos,
                                                              ball_pos, self._prev_ball_pos)
        self._reward_buf[:] = self._reward_ball_approach_w * approach_r
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._reset_ball(env_ids)
            self._record_reset_prev_states(env_ids)
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
        return
