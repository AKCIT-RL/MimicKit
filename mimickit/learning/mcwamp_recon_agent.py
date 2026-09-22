"""MCWAMP agent with an encoder-decoder actor and a plain reconstruction loss
(arXiv:2511.03996, Table 3 Recon. column).

The decoder predicts privileged state from the encoder latent and is trained
together with the policy. Deliberately separate from MCWAMPEncAgent, which
weights the same loss by how long ago the ball was last seen and splits its
diagnostics on ball visibility: both read a perception mask that exists only in
the soccer observation, and would read an unrelated slot here.

Trained from scratch: the measurable actor obs shares no layout with the
full-state checkpoints, so there is no warm start in this mode.
"""

import torch

import learning.mcwamp_agent as mcwamp_agent
import learning.mcwamp_enc_model as mcwamp_enc_model
import learning.normalizer as normalizer
import util.torch_util as torch_util


class MCWAMPReconAgent(mcwamp_agent.MCWAMPAgent):

    def _load_params(self, config):
        super()._load_params(config)
        # the target here is a velocity in m/s, not the soccer env's ball state
        # in metres, and the MSE is taken on the RAW target (no normalizer), so
        # this weight does not transfer from the soccer configs
        self._recon_weight = float(config.get("recon_weight", 1.0))
        return

    def _build_model(self, config):
        self._model = mcwamp_enc_model.MCWAMPEncModel(config["model"], self._env)
        return

    def _build_normalizers(self):
        super()._build_normalizers()
        assert self._use_critic_obs, \
            "MCWAMPReconAgent requires an env with a privileged critic_obs"
        # the critic obs has its own (different) layout; the normalizer built by
        # MCWAMPAgent from the actor obs space is resized
        critic_space = self._env.get_critic_obs_space()
        critic_dtype = torch_util.numpy_dtype_to_torch(critic_space.dtype)
        self._critic_obs_norm = normalizer.Normalizer(
            critic_space.shape, clip=10.0, device=self._device, dtype=critic_dtype)
        return

    def load_state_dict(self, state_dict):
        # MCWAMPAgent seeds a missing critic_obs normalizer from the obs
        # normalizer, which is only valid when the two layouts match. Here they
        # never do; keep the fresh normalizer stats instead.
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

    def _compute_actor_loss(self, batch):
        info = super()._compute_actor_loss(batch)
        if (self._recon_weight != 0.0):
            # the full batch carries a valid target, so no rand-action mask
            norm_obs = self._obs_norm.normalize(batch["obs"])
            tar = batch["recon_tar"]
            pred = self._model.eval_recon(norm_obs)
            recon_loss = torch.mean(torch.square(pred - tar))
            info["actor_loss"] = info["actor_loss"] + self._recon_weight * recon_loss
            info["recon_loss"] = recon_loss.detach()

            # A falling recon_loss means nothing on its own: predicting the
            # batch mean already gets you the target's variance. Log that floor
            # next to it, so the ratio says whether the decoder learned the
            # signal or just its average.
            tar_var = torch.mean(torch.var(tar.reshape(-1, tar.shape[-1]), dim=0))
            info["recon_tar_var"] = tar_var.detach()
            info["recon_nmse"] = (recon_loss / torch.clamp(tar_var, min=1e-8)).detach()
        return info
