"""MCWAMP agent with encoder-decoder actor (arXiv:2511.03996, Frente F).

Adds a joint reconstruction loss on top of MCWAMPAgent: the decoder predicts
the TRUE planar ball state (position + velocity, heading frame) from the
encoder latent, trained together with the policy (paper coef 1). Trained
from scratch: the measurable actor obs share no layout with the full-state
checkpoints, so there is no warm start in this mode.
"""

import numpy as np
import torch

import learning.mcwamp_agent as mcwamp_agent
import learning.mcwamp_enc_model as mcwamp_enc_model
import learning.normalizer as normalizer
import util.torch_util as torch_util


class MCWAMPEncAgent(mcwamp_agent.MCWAMPAgent):

    def _load_params(self, config):
        super()._load_params(config)
        self._enc_recon_weight = float(config.get("enc_recon_weight", 1.0))
        # recon gating: exponential decay of the per-sample weight by frames
        # since the ball was last seen; <= 0 disables (c5 behavior)
        self._enc_recon_gate_halflife = float(config.get("enc_recon_gate_halflife", 0.0))
        return

    def _build_model(self, config):
        self._model = mcwamp_enc_model.MCWAMPEncModel(config["model"], self._env)
        return

    def _build_normalizers(self):
        super()._build_normalizers()
        assert self._use_critic_obs, \
            "MCWAMPEncAgent requires an env with a privileged critic_obs"
        # the critic obs has its own (larger) layout in measurable mode; the
        # normalizer built by MCWAMPAgent from the actor obs space is resized
        critic_space = self._env.get_critic_obs_space()
        critic_dtype = torch_util.numpy_dtype_to_torch(critic_space.dtype)
        self._critic_obs_norm = normalizer.Normalizer(
            critic_space.shape, clip=10.0, device=self._device, dtype=critic_dtype)
        return

    def load_state_dict(self, state_dict):
        # MCWAMPAgent seeds a missing critic_obs normalizer from the obs
        # normalizer, which is only valid when the two layouts match. Here
        # they never do; keep the fresh normalizer stats instead.
        own_state = self.state_dict()
        for key in own_state.keys():
            if (key.startswith("_critic_obs_norm.") and key not in state_dict):
                state_dict[key] = own_state[key].clone()
        super(mcwamp_agent.MCWAMPAgent, self).load_state_dict(state_dict)
        return

    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)
        self._exp_buffer.record("recon_tar", info["recon_tar"])
        return

    def _recon_gate_weights(self, obs):
        """Per-sample weight by how long ago the ball was last seen, derived
        from the perception mask history in the actor obs (frame_dim-th entry
        of each 87-dim frame; current frame + H history frames, oldest first).
        Ball never seen inside the window -> weight 0 (unrecoverable target,
        pure gradient noise)."""
        frame_dim = self._env.get_measurable_frame_dim()
        n_frames = 1 + self._env.get_measurable_hist_steps()
        mask_idx = torch.arange(0, n_frames, device=obs.device) * frame_dim + frame_dim - 1
        masks = obs[..., mask_idx] > 0.5  # [T, B, F] oldest -> newest
        # age = number of frames since the newest visible frame
        any_vis = masks.any(dim=-1)
        newest_vis = (masks.float() * torch.arange(1, n_frames + 1, device=obs.device)).argmax(dim=-1)
        age = (n_frames - 1 - newest_vis).float()  # 0 = visible now
        half_life = self._enc_recon_gate_halflife
        w = torch.exp(-age * (np.log(2.0) / half_life))
        w = torch.where(any_vis, w, torch.zeros_like(w))
        return w

    def _compute_actor_loss(self, batch):
        info = super()._compute_actor_loss(batch)
        if (self._enc_recon_weight != 0.0):
            # reconstruction on the full batch: every sample carries a valid
            # target, no rand-action mask needed
            norm_obs = self._obs_norm.normalize(batch["obs"])
            pred = self._model.eval_recon(norm_obs)
            sq_err = torch.square(pred - batch["recon_tar"])  # [T, B, 4]
            if (self._enc_recon_gate_halflife > 0):
                w = self._recon_gate_weights(batch["obs"]).unsqueeze(-1)
                recon_loss = (w * sq_err).sum() / torch.clamp(w.sum() * sq_err.shape[-1], min=1.0)
            else:
                recon_loss = torch.mean(sq_err)
            info["actor_loss"] = info["actor_loss"] + self._enc_recon_weight * recon_loss
            info["recon_loss"] = recon_loss.detach()

            # split by the raw perception mask (last entry of the current
            # measurable frame): does the decoder fail on visible or on
            # occluded balls?
            frame_dim = self._env.get_measurable_frame_dim()
            visible = batch["obs"][..., frame_dim - 1] > 0.5
            per_sample = sq_err.detach().mean(dim=-1)
            n_vis = visible.sum()
            n_hid = visible.numel() - n_vis
            info["recon_loss_visible"] = per_sample[visible].sum() / torch.clamp(n_vis, min=1)
            info["recon_loss_hidden"] = per_sample[~visible].sum() / torch.clamp(n_hid, min=1)
        return info
