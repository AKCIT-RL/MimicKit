"""MCWAMP agent with encoder-decoder actor (arXiv:2511.03996, Frente F).

Adds a joint reconstruction loss on top of MCWAMPAgent: the decoder predicts
the TRUE planar ball state (position + velocity, heading frame) from the
encoder latent, trained together with the policy (paper coef 1). Trained
from scratch: the measurable actor obs share no layout with the full-state
checkpoints, so there is no warm start in this mode.
"""

import torch

import learning.mcwamp_agent as mcwamp_agent
import learning.mcwamp_enc_model as mcwamp_enc_model
import learning.normalizer as normalizer
import util.torch_util as torch_util


class MCWAMPEncAgent(mcwamp_agent.MCWAMPAgent):

    def _load_params(self, config):
        super()._load_params(config)
        self._enc_recon_weight = float(config.get("enc_recon_weight", 1.0))
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

    def _compute_actor_loss(self, batch):
        info = super()._compute_actor_loss(batch)
        if (self._enc_recon_weight != 0.0):
            # reconstruction on the full batch: every sample carries a valid
            # target, no rand-action mask needed
            norm_obs = self._obs_norm.normalize(batch["obs"])
            pred = self._model.eval_recon(norm_obs)
            recon_loss = torch.mean(torch.square(pred - batch["recon_tar"]))
            info["actor_loss"] = info["actor_loss"] + self._enc_recon_weight * recon_loss
            info["recon_loss"] = recon_loss.detach()
        return info
