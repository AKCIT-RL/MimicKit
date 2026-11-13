"""Deploy-perception emulator (RT-01, docs/t1/37): estimated pose (P1) + 10 Hz ball detections (P2),
batched over N envs, torch only. One implementation for the Lab env and the MuJoCo eval, so train and
eval share the model. Parameters: data/percep/percep_emulator_v1.json.

Per tick call `step(t, pos, yaw, vel, ball, in_fov, scale)` with TRUE robot xy / yaw / xy velocity, the TRUE
ball xy (world) and a bool "ball inside the camera cone". It returns the pose the robot ESTIMATES
(est_pos, est_yaw) and the ball as the deploy delivers it: held in the field frame between detections,
converted with the current estimated pose (ball_rel, robot frame) and a mask (last detection < mask_age).
Goal/other field points use `to_robot_frame(point)` with the same estimated pose.
`scale` in [0, 1.5]: 0 = perfect (est = truth, ball = truth, mask 1); 1 = the measured deploy; 1.5 = stress.
"""
import json
import math
import os

import torch

_DEF = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "percep", "percep_emulator_v1.json")


def load_params(path=None):
    with open(path or _DEF) as f:
        return json.load(f)


def _rot2(yaw, v):
    """Rotate xy vectors v [N,2] by +yaw [N]."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], dim=-1)


class PercepEmulator(object):
    HIST = 5  # ball history ticks (>= delay 0.1 s at 30 Hz + 1)

    def __init__(self, n, device="cpu", params=None, seed=None):
        self.p = params if params is not None else load_params()
        self.pose_mult, self.ball_mult = 1.0, 1.0
        self.bias_correct = False  # deploy-side subtraction of the known detection bias (policy_io)
        self.n, self.dev = n, torch.device(device)
        self.g = torch.Generator(device=self.dev)
        if seed is not None:
            self.g.manual_seed(int(seed))
        b = self.p["ball"]
        self.dt = float(self.p["dt"])
        self._bc = torch.tensor(b["dist_bins_center"], device=self.dev)
        self._bdet = torch.tensor(b["detect_prob"], device=self.dev)
        self._bdir = torch.tensor(b["dir_sigma_deg"], device=self.dev) * math.pi / 180.0
        self._oedge = torch.tensor(b["outlier_speed_edges"], device=self.dev)
        self._oprob = torch.tensor(b["outlier_prob"], device=self.dev)
        z2 = lambda: torch.zeros(n, 2, device=self.dev)
        z1 = lambda: torch.zeros(n, device=self.dev)
        self.bias_p, self.bias_y = z2(), z1()
        self.ou_p, self.ou_y = z2(), z1()
        self.tail_on, self.tail_off, self.tail_v = z1(), z1(), z2()
        self.est_pos, self.est_yaw = z2(), z1()
        self.next_cap, self.miss = z1(), torch.zeros(n, dtype=torch.bool, device=self.dev)
        self.held_field = z2()
        self.last_det_t = torch.full((n,), -1e9, device=self.dev)
        self.hist = torch.zeros(n, self.HIST, 2, device=self.dev)
        self.hist_n = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.last_t = z1()
        self.reset(torch.arange(n, device=self.dev), 0.0)

    # ------------------------------------------------------------------ helpers
    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.g, device=self.dev)

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.dev)

    def _uni(self, rng, n):
        return rng[0] + (rng[1] - rng[0]) * self._rand(n)

    def _interp(self, d, ys):
        """piecewise-linear on the bin centres, flat before the first, `detect_prob_far` decay after the last."""
        x = self._bc
        i = torch.clamp(torch.searchsorted(x, d), 1, len(x) - 1)
        x0, x1, y0, y1 = x[i - 1], x[i], ys[i - 1], ys[i]
        w = torch.clamp((d - x0) / (x1 - x0), 0.0, 1.0)
        return y0 + w * (y1 - y0)

    # -------------------------------------------------------------------- reset
    def reset(self, ids, t):
        n = len(ids)
        if n == 0:
            return
        po, b = self.p["pose"], self.p["ball"]
        self.bias_p[ids] = po["pos_bias_std"] * self._randn(n, 2)
        self.bias_y[ids] = po["yaw_bias_std_deg"] * math.pi / 180.0 * self._randn(n)
        self.ou_p[ids] = po["pos_ou_sigma"] * self._randn(n, 2)
        self.ou_y[ids] = po["yaw_ou_sigma_deg"] * math.pi / 180.0 * self._randn(n)
        # localisation-loss jumps: a renewal process (exponential gap, then a 1-3 m jump held 1-3 s)
        has_tail = self._rand(n) < po["tail_prob"]
        gap = -po["tail_gap_s"] * torch.log(self._rand(n).clamp_min(1e-6))
        self.tail_on[ids] = torch.where(has_tail, t + gap, torch.full_like(gap, 1e9))
        self.tail_off[ids] = self.tail_on[ids] + self._uni(po["tail_dur_s"], n)
        mag = self._uni(po["tail_jump_m"], n)
        ang = 2 * math.pi * self._rand(n)
        self.tail_v[ids] = torch.stack([mag * torch.cos(ang), mag * torch.sin(ang)], dim=-1)
        self.next_cap[ids] = t + self.dt * (b["period_ticks"] * self._rand(n))  # random phase
        self.miss[ids] = False
        self.last_det_t[ids] = -1e9
        self.hist_n[ids] = 0
        self.held_field[ids] = 0.0

    # --------------------------------------------------------------------- pose
    def _pose_err(self, t, scale):
        po = self.p["pose"]
        dt = self.dt
        for ou, tau, sig in ((self.ou_p, po["pos_ou_tau"], po["pos_ou_sigma"]),):
            a = math.exp(-dt / tau)
            ou.mul_(a).add_(sig * math.sqrt(1 - a * a) * self._randn(self.n, 2))
        a = math.exp(-dt / po["yaw_ou_tau"])
        sy = po["yaw_ou_sigma_deg"] * math.pi / 180.0
        self.ou_y.mul_(a).add_(sy * math.sqrt(1 - a * a) * self._randn(self.n))
        renew = (t >= self.tail_off) & (self.tail_on < 1e8)
        if renew.any():
            ids = renew.nonzero(as_tuple=False).flatten()
            k = len(ids)
            self.tail_on[ids] = t + (-po["tail_gap_s"] * torch.log(self._rand(k).clamp_min(1e-6)))
            self.tail_off[ids] = self.tail_on[ids] + self._uni(po["tail_dur_s"], k)
            mag = self._uni(po["tail_jump_m"], k)
            ang = 2 * math.pi * self._rand(k)
            self.tail_v[ids] = torch.stack([mag * torch.cos(ang), mag * torch.sin(ang)], dim=-1)
        tail = ((t >= self.tail_on) & (t < self.tail_off)).unsqueeze(-1).float() * self.tail_v
        sc = scale.unsqueeze(-1) if torch.is_tensor(scale) else scale
        e_pos = sc * (self.bias_p + self.ou_p + tail)
        e_yaw = scale * (self.bias_y + self.ou_y)
        return e_pos, e_yaw

    def to_robot_frame(self, point_xy):
        """world xy [N,2] -> robot frame with the ESTIMATED pose (what the deploy computes)."""
        return _rot2(-self.est_yaw, point_xy - self.est_pos)

    def meas_var(self, d, scale=1.0):
        """isotropic variance (m^2) of one detection at distance d [N] from the noise model (mean of the
        radial and tangential variances); what a deploy-side filter can compute from the same table."""
        b = self.p["ball"]
        sd = torch.clamp(b["dist_sigma_slope"] * d + b["dist_sigma_off"], min=b["dist_sigma_min"])
        st = d * self._interp(d, self._bdir)
        sc = torch.as_tensor(scale, device=self.dev, dtype=d.dtype) * self.ball_mult
        return 0.5 * (sc * sd) ** 2 + 0.5 * (sc * st) ** 2 + 1e-4

    # --------------------------------------------------------------------- step
    def step(self, t, pos, yaw, vel, ball, in_fov, scale=1.0):
        """t float (s); pos [N,2], yaw [N], vel [N,2], ball [N,2] true; in_fov [N] bool."""
        b = self.p["ball"]
        # `scale` is a float or a per-env tensor [N]; rows with scale <= 0 are served the truth
        sc = scale if torch.is_tensor(scale) else torch.full((self.n,), float(scale), device=self.dev)
        # diagnostic split (default 1): pose_mult scales only the pose error, ball_mult only the ball measurement
        sc_pose = sc * self.pose_mult
        sc = sc * self.ball_mult
        zero = sc <= 0.0
        e_pos, e_yaw = self._pose_err(t, sc_pose)
        self.est_pos, self.est_yaw = pos + e_pos, yaw + e_yaw
        if bool(zero.all()) and bool((sc_pose <= 0.0).all()):
            self.held_field = ball.clone()
            self.last_det_t[:] = t
            return dict(est_pos=pos, est_yaw=yaw, ball_rel=_rot2(-yaw, ball - pos),
                        mask=torch.ones(self.n, device=self.dev), detected=torch.zeros(self.n, dtype=torch.bool, device=self.dev))
        # ball history (for the 0-100 ms delay)
        self.hist = torch.roll(self.hist, 1, dims=1)
        self.hist[:, 0] = ball
        self.hist_n = torch.clamp(self.hist_n + 1, max=self.HIST)
        detected = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        cap = t >= self.next_cap
        if cap.any():
            ids = cap.nonzero(as_tuple=False).flatten()
            k = len(ids)
            jit = (self._rand(k) < b["period_jitter_prob"]).float() * (2 * (self._rand(k) < 0.5).float() - 1)
            self.next_cap[ids] = self.next_cap[ids] + self.dt * (b["period_ticks"] + jit)
            # delayed true ball
            delay = sc[ids] * self._uni(b["delay_s"], k)
            lag = delay / self.dt
            i0 = torch.clamp(lag.floor().long(), 0, self.HIST - 2)
            i0 = torch.minimum(i0, torch.clamp(self.hist_n[ids] - 2, min=0))
            w = (lag - i0.float()).clamp(0, 1).unsqueeze(-1)
            bd = (1 - w) * self.hist[ids, i0] + w * self.hist[ids, i0 + 1]
            rel = bd - pos[ids]
            d = rel.norm(dim=-1)
            # detection: Markov chain on top of the per-distance rate
            pdet = self._interp(d, self._bdet)
            pdet = torch.where(d > self._bc[-1], torch.full_like(pdet, b["detect_prob_far"]), pdet)
            m = torch.clamp(sc[ids] * (1 - pdet), 0.0, 0.95)
            stay = b["hole_stay_prob"]
            p_hm = torch.clamp(m * (1 - stay) / (1 - m), 0.0, 1.0)
            u = self._rand(k)
            was = self.miss[ids]
            miss_now = torch.where(was, u < stay, u < p_hm)
            seen = in_fov[ids] & ~miss_now
            self.miss[ids] = miss_now & in_fov[ids]
            # measurement in the robot frame (true yaw: the camera is on the robot)
            r_rel = _rot2(-yaw[ids], rel)
            ang = torch.atan2(r_rel[:, 1], r_rel[:, 0])
            sd = sc[ids] * torch.clamp(b["dist_sigma_slope"] * d + b["dist_sigma_off"], min=b["dist_sigma_min"])
            bias_d = torch.where(d < 1.0, torch.full_like(d, b["dist_bias_near"]),
                                 b["dist_bias_slope"] * d + b["dist_bias_off"]) * sc[ids]
            sdir = sc[ids] * self._interp(d, self._bdir)
            dm = d + bias_d + sd * self._randn(k)
            am = ang + sc[ids] * b["dir_bias_deg"] * math.pi / 180.0 + sdir * self._randn(k)
            if self.bias_correct:
                # deploy-side calibration: subtract the table's mean bias (scale-1 table) at the MEASURED distance
                dm = dm - torch.where(dm < 1.0, torch.full_like(dm, b["dist_bias_near"]),
                                      b["dist_bias_slope"] * dm + b["dist_bias_off"])
                am = am - b["dir_bias_deg"] * math.pi / 180.0
            meas = torch.stack([dm * torch.cos(am), dm * torch.sin(am)], dim=-1)
            # outliers by robot speed
            sp = vel[ids].norm(dim=-1)
            po = self._oprob[torch.bucketize(sp, self._oedge)] * sc[ids]
            out = self._rand(k) < po
            oang = 2 * math.pi * self._rand(k)
            omag = self._uni(b["outlier_mag_m"], k)
            meas = torch.where(out.unsqueeze(-1), meas + omag.unsqueeze(-1) * torch.stack([torch.cos(oang), torch.sin(oang)], -1), meas)
            # false positives (a message with no ball in view still gets a detection somewhere)
            fp = self._rand(k) < b["false_prob"] * sc[ids]
            fr = b["false_range_m"] * torch.sqrt(self._rand(k))
            fa = 2 * math.pi * self._rand(k)
            meas = torch.where(fp.unsqueeze(-1), torch.stack([fr * torch.cos(fa), fr * torch.sin(fa)], -1), meas)
            got = seen | fp
            # deploy: ball in the field frame with the pose estimated NOW (at arrival)
            field = self.est_pos[ids] + _rot2(self.est_yaw[ids], meas)
            gi = ids[got]
            self.held_field[gi] = field[got]
            self.last_det_t[gi] = t
            detected[gi] = True
        if bool(zero.any()):
            pz = sc_pose <= 0.0
            self.est_pos = torch.where(pz.unsqueeze(-1), pos, self.est_pos)
            self.est_yaw = torch.where(pz, yaw, self.est_yaw)
            # true ball in the robot frame, held in the field frame of the (possibly estimated) pose
            true_field = self.est_pos + _rot2(self.est_yaw, _rot2(-yaw, ball - pos))
            self.held_field = torch.where(zero.unsqueeze(-1), true_field, self.held_field)
            self.last_det_t = torch.where(zero, torch.full_like(self.last_det_t, t), self.last_det_t)
        ball_rel = self.to_robot_frame(self.held_field)
        mask = ((t - self.last_det_t) < b["mask_age_s"]).float()
        return dict(est_pos=self.est_pos, est_yaw=self.est_yaw, ball_rel=ball_rel, mask=mask, detected=detected)
