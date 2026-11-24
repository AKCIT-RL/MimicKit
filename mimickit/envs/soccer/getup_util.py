"""Get-up (fall recovery) mode for the soccer env (docs/t1/47).

Episodes in get-up mode start lying on the ground (supine / prone / side / random tilt, or a
small standing fraction), body contact does not end the episode, and the reward streams are
replaced by the literature terms of doc 47 section 4 (AGILE StandUp-T1 + HiFAR T1 + HoST):

  task stream (goal critic):  base height (3 widths), upright x head height, stand, survival,
                              post-task stillness/pose terms gated on standing, orientation
  aux stream (style critic):  regularization (torques, tiredness, power, joint vel/acc, action
                              rate, smoothness, joint limits, hip roll), plus the AMP style
                              reward, whose gain is 0 while the robot is down

Every weight is per second (Isaac Lab / legged_gym convention): the per-step reward is
weight * term * dt. The actor observation is NOT changed: the student sees the same vector
and infers "fallen" from the projected gravity already in its frame.
"""
import math
import re

import torch

import util.torch_util as torch_util
import envs.base_env as base_env


# HiFAR / AGILE / HoST weights (doc 47 section 4); "w_" keys are per-second weights
_DEFAULTS = {
    "prob": 1.0,                 # fraction of episodes in get-up mode
    "stand_prob": 0.1,           # of those, fraction that starts standing (AGILE standing_ratio)
    "pose_probs": [0.3, 0.3, 0.25, 0.15],  # supine, prone, side, random tilt
    "rot_noise": 0.25,           # rad, roll/pitch noise on the lying orientation
    "joint_noise": 0.35,         # fraction of the half joint range, uniform around the default pose
    "drop_clearance": 0.15,      # m, lowest link origin above the ground at spawn
    "target_h": 0.68,            # m, base height standing (HiFAR target, our default root)
    "stand_h": 0.58,             # m, "standing" = h > target - 0.1 (HiFAR)
    "stand_upz": 0.9,            # body z axis . world z for "standing"
    "succ_hold_s": 1.0,          # s standing in a row = success
    "head_frac": 0.85,           # head height target = frac * head height standing
    "explode_v2": 1000.0,        # |v|^2 + |w|^2 above this ends the episode (HiFAR; 80 cut 92% of the up0s
                                 # episodes: a 31 kg robot falling/flailing passes 9 rad/s)
    "episode_length": 10.0,      # s
    # task stream
    "w_height": [2.0, 8.0, 16.0],      # AGILE base_height rough / medium / fine
    "height_std": [0.5, 0.25, 0.1],
    "w_upright_head": 2.5,              # HoST task_orientation * task_head_height (task group weight 2.5)
    "w_stand": 2.0,                     # HiFAR stand
    "w_survival": 4.0,                  # HiFAR survival
    "w_post_ang_vel": 10.0,             # HoST target_ang_vel_xy, standing only
    "w_post_lin_vel": 10.0,             # HoST target_lin_vel_xy, standing only
    "w_orientation": -5.0,              # AGILE flat_orientation (trunk), always
    "w_dof_pos_ref": -0.3,              # HiFAR dof_pos_ref, |h - target| < 0.1
    "w_joint_dev": -0.1,                # AGILE joint_deviation_l1_if_standing (-0.05 -> -0.5)
    "w_feet_dist": -50.0,               # AGILE feet_distance_from_ref (0.2 m), standing only
    "feet_dist_ref": 0.2,
    # aux stream (regularization)
    "w_torques": -1e-5,                 # HiFAR / AGILE
    "w_tiredness": -1e-2,               # HiFAR torque_tiredness
    "w_power": -2e-3,                   # HiFAR power
    "w_dof_vel": -1e-3,                 # HiFAR
    "w_dof_acc": -2.5e-7,               # HoST
    "w_action_rate": -0.01,             # AGILE (HoST same)
    "w_smoothness": -0.01,              # HoST regu_smoothness
    "w_dof_limits": -1.0,               # HiFAR dof_pos_limits
    "dof_soft_limit": 0.95,
    "w_hip_roll": 0.0,                  # HiFAR stage 2: -50
    "style_when_down": 0.0,             # AMP style gain while not standing (state-dependent AMP gate)
}


class GetupMode:
    def __init__(self, env, cfg):
        self._env = env
        c = dict(_DEFAULTS)
        c.update(cfg or {})
        self.cfg = c
        self.prob = float(c["prob"])
        self.episode_length = float(c["episode_length"])
        return

    # ------------------------------------------------------------------ build
    def build(self):
        env = self._env
        dev = env._device
        n = env.get_num_envs()
        char_id = env._get_char_id()
        self.mode = torch.zeros(n, device=dev, dtype=torch.bool)
        self.pose_cls = torch.full([n], -1, device=dev, dtype=torch.long)
        self.stand_t = torch.zeros(n, device=dev)
        self.succ = torch.zeros(n, device=dev, dtype=torch.bool)
        self.succ_t = torch.zeros(n, device=dev)
        self.fell_after = torch.zeros(n, device=dev, dtype=torch.bool)
        self.max_h = torch.zeros(n, device=dev)
        num_dof = env._init_dof_pos.shape[-1]
        self.act1 = torch.zeros([n, num_dof], device=dev)
        self.act2 = torch.zeros([n, num_dof], device=dev)
        self.act0 = torch.zeros([n, num_dof], device=dev)
        self.fresh = torch.ones(n, device=dev, dtype=torch.bool)

        lo, hi = env._engine.get_obj_dof_limits(0, char_id)
        self.dof_lo = torch.as_tensor(lo, device=dev, dtype=torch.float)
        self.dof_hi = torch.as_tensor(hi, device=dev, dtype=torch.float)
        mid = 0.5 * (self.dof_lo + self.dof_hi)
        half = 0.5 * (self.dof_hi - self.dof_lo) * float(self.cfg["dof_soft_limit"])
        self.soft_lo = mid - half
        self.soft_hi = mid + half

        self.head_id = int(env._build_body_ids_tensor(["aahead_pitch_link"])[0].item())
        self.feet_ids = env._build_body_ids_tensor(["left_ankle_roll_link", "right_ankle_roll_link"])
        hip_roll = [i for i, nm in enumerate(self._dof_names()) if "hip_roll" in nm]
        self.hip_roll_ids = torch.tensor(hip_roll, device=dev, dtype=torch.long)
        # torque limits: with pd_explicit PhysX reports 1e9 (env._torque_lim), the real effort limit is
        # in the engine's pd_explicit_gains {regex: [kp, kd, effort]}, matched on the joint names
        self.tau_lim = env._torque_lim.clone()
        gains = getattr(env._engine, "_pd_explicit_gains", None)
        if gains:
            for i, nm in enumerate(self._dof_names()):
                for rx, v in gains.items():
                    if re.fullmatch(rx, nm):
                        self.tau_lim[i] = float(v[2])
                        break
        assert bool((self.tau_lim < 1e6).all()), "getup: torque limits not resolved %s" % self.tau_lim.tolist()

        # head height standing: FK of the default pose with the root at the target height
        root_pos = torch.zeros([1, 3], device=dev)
        root_pos[0, 2] = float(self.cfg["target_h"])
        root_rot = torch.zeros([1, 4], device=dev)
        root_rot[0, 3] = 1.0
        jr = env._kin_char_model.dof_to_rot(env._init_dof_pos.unsqueeze(0))
        body_pos, _ = env._kin_char_model.forward_kinematics(root_pos, root_rot, jr)
        foot_z = body_pos[0, self.feet_ids, 2].min()
        self.head_h_stand = float(body_pos[0, self.head_id, 2] - foot_z)
        return

    def _dof_names(self):
        km = self._env._kin_char_model
        names = []
        for j in range(1, km.get_num_joints()):
            jt = km.get_joint(j)
            for _ in range(jt.get_dof_dim()):
                names.append(jt.name)
        return names

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids):
        """Called after the soccer placement (x, y, yaw set). Overwrites the robot state of the
        get-up episodes with a lying pose dropped from just above the ground."""
        env = self._env
        dev = env._device
        n = env_ids.shape[0]
        if (n == 0):
            return
        on = torch.rand(n, device=dev) < self.prob
        self.mode[env_ids] = on
        self.stand_t[env_ids] = 0.0
        self.succ[env_ids] = False
        self.succ_t[env_ids] = 0.0
        self.fell_after[env_ids] = False
        self.max_h[env_ids] = 0.0
        self.act0[env_ids] = 0.0
        self.act1[env_ids] = 0.0
        self.act2[env_ids] = 0.0
        self.fresh[env_ids] = True
        self.pose_cls[env_ids] = -1
        ids = env_ids[on]
        m = int(ids.shape[0])
        if (m == 0):
            return
        c = self.cfg
        char_id = env._get_char_id()
        stand = torch.rand(m, device=dev) < float(c["stand_prob"])
        probs = torch.tensor(c["pose_probs"], device=dev, dtype=torch.float)
        cls = torch.multinomial(probs / probs.sum(), m, replacement=True)
        cls = torch.where(stand, torch.full_like(cls, 4), cls)
        self.pose_cls[ids] = cls

        rn = float(c["rot_noise"])
        roll = rn * (2.0 * torch.rand(m, device=dev) - 1.0)
        pitch = rn * (2.0 * torch.rand(m, device=dev) - 1.0)
        half_pi = 0.5 * math.pi
        # supine: chest up = pitch -pi/2 (body x -> world +z); prone: +pi/2; side: roll +-pi/2
        pitch = torch.where(cls == 0, pitch - half_pi, pitch)
        pitch = torch.where(cls == 1, pitch + half_pi, pitch)
        side_sign = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        roll = torch.where(cls == 2, roll + side_sign * half_pi, roll)
        rnd = cls == 3
        roll = torch.where(rnd, math.pi * (2.0 * torch.rand(m, device=dev) - 1.0), roll)
        pitch = torch.where(rnd, half_pi * (2.0 * torch.rand(m, device=dev) - 1.0), pitch)
        roll = torch.where(stand, torch.zeros_like(roll), roll)
        pitch = torch.where(stand, torch.zeros_like(pitch), pitch)
        yaw = 2.0 * math.pi * torch.rand(m, device=dev) - math.pi
        # yaw applied in the world frame after the tilt
        yaw_axis = torch.zeros([m, 3], device=dev)
        yaw_axis[:, 2] = 1.0
        tilt = torch_util.euler_xyz_to_quat(roll, pitch, torch.zeros_like(yaw))
        root_rot = torch_util.quat_mul(torch_util.axis_angle_to_quat(yaw_axis, yaw), tilt)

        dof0 = env._init_dof_pos.unsqueeze(0).repeat(m, 1)
        half = 0.5 * (self.dof_hi - self.dof_lo)
        noise = float(c["joint_noise"]) * half * (2.0 * torch.rand_like(dof0) - 1.0)
        noise = torch.where(stand.unsqueeze(-1), torch.zeros_like(noise), noise)
        dof = torch.clamp(dof0 + noise, self.dof_lo, self.dof_hi)

        root_pos = env._engine.get_root_pos(char_id)[ids].clone()
        gz = _as_vec(env._ground_z(ids), m, dev)
        jr = env._kin_char_model.dof_to_rot(dof)
        rp0 = root_pos.clone()
        rp0[:, 2] = 0.0
        body_pos, _ = env._kin_char_model.forward_kinematics(rp0, root_rot, jr)
        min_z = body_pos[..., 2].min(dim=-1).values
        lie_z = -min_z + float(c["drop_clearance"]) + gz
        stand_z = float(env._init_root_pos[2]) + gz
        root_pos[:, 2] = torch.where(stand, stand_z, lie_z)

        zero = torch.zeros([m, 3], device=dev)
        env._engine.set_root_pos(ids, char_id, root_pos)
        env._engine.set_root_rot(ids, char_id, root_rot)
        env._engine.set_root_vel(ids, char_id, zero)
        env._engine.set_root_ang_vel(ids, char_id, zero)
        env._engine.set_dof_pos(ids, char_id, dof)
        env._engine.set_dof_vel(ids, char_id, 0.0)
        env._engine.set_body_vel(ids, char_id, 0.0)
        env._engine.set_body_ang_vel(ids, char_id, 0.0)
        return

    def park_ball(self, env_ids):
        """Ball still, in the field corner farthest from the robot (no goal/oob events)."""
        env = self._env
        ids = env_ids[self.mode[env_ids]]
        m = int(ids.shape[0])
        if (m == 0):
            return
        dev = env._device
        ball_id = env._get_ball_id()
        char_id = env._get_char_id()
        rxy = env._engine.get_root_pos(char_id)[ids, 0:2] - env._field_offset[ids]
        hx = 0.5 * env._field_length - 0.6
        hy = 0.5 * env._field_width - 0.6
        bx = torch.where(rxy[:, 0] > 0.0, -hx, hx)
        by = torch.where(rxy[:, 1] > 0.0, -hy, hy)
        pos = torch.zeros([m, 3], device=dev)
        pos[:, 0] = bx
        pos[:, 1] = by
        pos[:, 0:2] += env._field_offset[ids]
        pos[:, 2] = env._ball_radius + _as_vec(env._ground_z(ids), m, dev) + 0.01
        rot = torch.zeros([m, 4], device=dev)
        rot[:, 3] = 1.0
        zero = torch.zeros([m, 3], device=dev)
        env._engine.set_root_pos(ids, ball_id, pos)
        env._engine.set_root_rot(ids, ball_id, rot)
        env._engine.set_root_vel(ids, ball_id, zero)
        env._engine.set_root_ang_vel(ids, ball_id, zero)
        env._prev_ball_pos[ids] = pos
        return

    # ------------------------------------------------------------------ step
    def on_action(self, actions):
        a = actions.detach()
        f = self.fresh.unsqueeze(-1)
        self.act2[:] = torch.where(f, a, self.act1)
        self.act1[:] = torch.where(f, a, self.act0)
        self.act0[:] = a
        self.fresh[:] = False
        return

    def _state(self):
        env = self._env
        char_id = env._get_char_id()
        root_pos = env._engine.get_root_pos(char_id)
        root_rot = env._engine.get_root_rot(char_id)
        gz = _as_vec(env._ground_z(None), root_pos.shape[0], env._device)
        h = root_pos[:, 2] - gz
        ez = torch.zeros_like(root_pos)
        ez[:, 2] = 1.0
        upz = torch_util.quat_rotate(root_rot, ez)[:, 2]
        return char_id, root_pos, root_rot, gz, h, upz

    def compute_rewards(self):
        """Overwrite the task/aux reward and the style gain of the get-up envs."""
        env = self._env
        c = self.cfg
        dt = env._engine.get_timestep()
        char_id, root_pos, root_rot, gz, h, upz = self._state()
        body_pos = env._engine.get_body_pos(char_id)
        q = env._engine.get_dof_pos(char_id)
        qd = env._engine.get_dof_vel(char_id)
        tau = env._engine.get_dof_forces(char_id)
        inv = torch_util.quat_conjugate(root_rot)
        lin_b = torch_util.quat_rotate(inv, env._engine.get_root_vel(char_id))
        ang_b = torch_util.quat_rotate(inv, env._engine.get_root_ang_vel(char_id))
        q0 = env._init_dof_pos.unsqueeze(0)

        standing = (h > float(c["stand_h"])) & (upz > float(c["stand_upz"]))
        sf = standing.float()
        terms = {}
        # ---- task
        err = h - float(c["target_h"])
        hr = torch.zeros_like(h)
        for w, s in zip(c["w_height"], c["height_std"]):
            hr = hr + w * torch.exp(-torch.square(err / s))
        terms["height"] = hr
        head_h = body_pos[:, self.head_id, 2] - gz
        head_tar = float(c["head_frac"]) * self.head_h_stand
        tol_up = _tolerance_lo(upz, 0.99, 1.0)
        tol_head = _tolerance_lo(head_h, head_tar, head_tar)
        terms["upright_head"] = c["w_upright_head"] * tol_up * tol_head
        terms["stand"] = c["w_stand"] * (h > float(c["stand_h"])).float()
        terms["survival"] = torch.full_like(h, float(c["w_survival"]))
        terms["post_ang_vel"] = c["w_post_ang_vel"] * sf * torch.exp(-2.0 * torch.sum(torch.square(ang_b[:, 0:2]), dim=-1))
        terms["post_lin_vel"] = c["w_post_lin_vel"] * sf * torch.exp(-5.0 * torch.sum(torch.square(lin_b[:, 0:2]), dim=-1))
        terms["orientation"] = c["w_orientation"] * (1.0 - torch.square(upz))
        near = (torch.abs(err) < 0.1).float()
        terms["dof_pos_ref"] = c["w_dof_pos_ref"] * near * torch.sum(torch.square(q - q0), dim=-1)
        terms["joint_dev"] = c["w_joint_dev"] * sf * torch.sum(torch.abs(q - q0), dim=-1)
        feet = body_pos[:, self.feet_ids, 0:2]
        fd = torch.linalg.norm(feet[:, 0] - feet[:, 1], dim=-1)
        terms["feet_dist"] = c["w_feet_dist"] * sf * torch.abs(fd - float(c["feet_dist_ref"]))
        task = sum(terms.values()) * dt

        # ---- aux (regularization)
        reg = {}
        reg["torques"] = c["w_torques"] * torch.sum(torch.square(tau), dim=-1)
        ratio = tau / self.tau_lim
        reg["tiredness"] = c["w_tiredness"] * torch.sum(torch.clamp(torch.square(ratio), max=1.0), dim=-1)
        reg["power"] = c["w_power"] * torch.sum(torch.clamp(tau * qd, min=0.0), dim=-1)
        reg["dof_vel"] = c["w_dof_vel"] * torch.sum(torch.square(qd), dim=-1)
        if (env._prev_dof_vel is not None):
            ddq = (qd - env._prev_dof_vel) / dt
            reg["dof_acc"] = c["w_dof_acc"] * torch.sum(torch.square(ddq), dim=-1)
        reg["action_rate"] = c["w_action_rate"] * torch.sum(torch.square(self.act0 - self.act1), dim=-1)
        reg["smoothness"] = c["w_smoothness"] * torch.sum(
            torch.square(self.act0 - 2.0 * self.act1 + self.act2), dim=-1)
        viol = torch.clamp(self.soft_lo - q, min=0.0) + torch.clamp(q - self.soft_hi, min=0.0)
        reg["dof_limits"] = c["w_dof_limits"] * torch.sum(viol, dim=-1)
        if (float(c["w_hip_roll"]) != 0.0):
            reg["hip_roll"] = c["w_hip_roll"] * torch.sum(
                torch.square(q.index_select(-1, self.hip_roll_ids) - q0[:, self.hip_roll_ids]), dim=-1)
        aux = sum(reg.values()) * dt

        m = self.mode
        env._task_reward_buf[:] = torch.where(m, task, env._task_reward_buf)
        env._aux_reward_buf[:] = torch.where(m, aux, env._aux_reward_buf)
        gain = torch.where(standing, torch.ones_like(h), torch.full_like(h, float(c["style_when_down"])))
        env._style_gain_buf[:] = torch.where(m, gain, env._style_gain_buf)

        # ---- success bookkeeping
        self.stand_t[:] = torch.where(standing, self.stand_t + dt, torch.zeros_like(self.stand_t))
        new_succ = m & (~self.succ) & (self.stand_t >= float(c["succ_hold_s"]))
        self.succ_t[:] = torch.where(new_succ, env._time_buf - float(c["succ_hold_s"]), self.succ_t)
        self.succ |= new_succ
        self.fell_after |= self.succ & (h < 0.35)
        self.max_h[:] = torch.maximum(self.max_h, h)

        d = env._diag
        mf = m.float()
        if (mf.sum() > 0):
            for k, v in terms.items():
                d.add_mean("up_r_" + k, v[m] * dt)
            for k, v in reg.items():
                d.add_mean("up_r_" + k, v[m] * dt)
            d.add_mean("up_height", h[m])
            d.add_mean("up_standing", sf[m])
            d.add_mean("up_tau_sat95", torch.mean((torch.abs(ratio[m]) >= 0.95).float(), dim=-1))
        return

    def mask_done(self, done_buf, null_val, fail_val):
        """Body contact does not end a get-up episode. compute_done writes FAIL over TIME, so a
        lying robot at the time limit must get TIME back (else the episode never ends)."""
        env = self._env
        fail = (done_buf == fail_val) & self.mode
        timeout = env._time_buf >= env._episode_length
        repl = torch.where(timeout, torch.full_like(done_buf, base_env.DoneFlags.TIME.value),
                           torch.full_like(done_buf, null_val))
        done_buf[:] = torch.where(fail, repl, done_buf)
        return

    def final_done(self, done_buf, null_val, fail_val):
        """Explosion check (HiFAR) and per-episode success stats; after every other done rule."""
        env = self._env
        char_id = env._get_char_id()
        v2 = torch.sum(torch.square(env._engine.get_root_vel(char_id)), dim=-1) \
            + torch.sum(torch.square(env._engine.get_root_ang_vel(char_id)), dim=-1)
        boom = self.mode & (v2 > float(self.cfg["explode_v2"])) & (done_buf == null_val)
        if bool(self.mode.any()):
            env._diag.add_mean("up_v2", v2[self.mode])
        done_buf[:] = torch.where(boom, torch.full_like(done_buf, fail_val), done_buf)
        ended = self.mode & (done_buf != null_val)
        d = env._diag
        d.add_sum("up_ep_done", ended.float())
        d.add_sum("up_ep_succ", (ended & self.succ).float())
        d.add_sum("up_succ_time", (ended & self.succ).float() * self.succ_t)
        d.add_sum("up_ep_fell_after", (ended & self.succ & self.fell_after).float())
        d.add_sum("up_ep_explode", boom.float())
        for k in range(5):
            ek = ended & (self.pose_cls == k)
            d.add_sum("up_ep_done_c%d" % k, ek.float())
            d.add_sum("up_ep_succ_c%d" % k, (ek & self.succ).float())
        return

    @staticmethod
    def summarize(sums, diags):
        pr = lambda a, b: a / b if b > 0.0 else 0.0
        eps = sums.get("up_ep_done", 0.0)
        succ = sums.get("up_ep_succ", 0.0)
        diags["up_succ_rate"] = pr(succ, eps)
        diags["up_succ_time"] = pr(sums.get("up_succ_time", 0.0), succ)
        diags["up_fell_after_rate"] = pr(sums.get("up_ep_fell_after", 0.0), succ)
        diags["up_explode_rate"] = pr(sums.get("up_ep_explode", 0.0), eps)
        for k, nm in enumerate(["supine", "prone", "side", "random", "stand"]):
            diags["up_succ_" + nm] = pr(sums.get("up_ep_succ_c%d" % k, 0.0), sums.get("up_ep_done_c%d" % k, 0.0))
        return


def _tolerance_lo(x, lo, margin):
    """dm_control-style tolerance with bounds [lo, inf): 1 inside, gaussian to 0.1 at lo - margin."""
    d = torch.clamp(lo - x, min=0.0) / margin
    return torch.exp(-math.log(10.0) * torch.square(d))


def _as_vec(z, n, dev):
    if (torch.is_tensor(z)):
        return z.to(dev).float().reshape(-1).expand(n) if z.numel() == 1 else z.float()
    return torch.full([n], float(z), device=dev)
