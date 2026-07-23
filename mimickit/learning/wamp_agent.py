import torch

import learning.amp_agent as amp_agent
import learning.wgan_util as wgan_util
import util.torch_util as torch_util


class WAMPAgent(amp_agent.AMPAgent):
    """AMP agent with a soft-boundary Wasserstein critic (WGAN-GP).

    Replaces the BCE discriminator objective of AMPAgent with the
    Wasserstein formulation of arXiv:2511.03996 (Sec. 7) / HumanMimic:
    - tanh soft-boundary critic loss,
    - gradient penalty on demo/agent interpolations (target norm 1),
    - bounded style reward increasing in the critic score.
    The paper does not use a policy replay buffer for the critic, so
    replay mixing is disabled.
    """

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _load_params(self, config):
        super()._load_params(config)
        self._disc_score_scale = config["disc_score_scale"]
        return

    def _store_disc_replay_data(self):
        # WGAN critic uses only on-policy samples (no replay mixing).
        return

    def _compute_disc_loss(self, batch):
        disc_obs = batch["disc_obs"]
        disc_demo_obs = batch["disc_obs_demo"]

        norm_disc_obs_demo = self._disc_obs_norm.normalize(disc_demo_obs)
        norm_disc_obs = self._disc_obs_norm.normalize(disc_obs)

        disc_agent_score = self._model.eval_disc(norm_disc_obs).squeeze(-1)
        disc_demo_score = self._model.eval_disc(norm_disc_obs_demo).squeeze(-1)

        disc_loss, w_info = wgan_util.compute_wasserstein_disc_loss(
            disc_demo_score, disc_agent_score, self._disc_score_scale)

        disc_grad_penalty, grad_norm_mean = wgan_util.compute_interp_grad_penalty(
            self._model.eval_disc, norm_disc_obs_demo, norm_disc_obs)
        disc_loss = disc_loss + self._disc_grad_penalty * disc_grad_penalty

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_score, disc_demo_score)

        disc_info = {
            "disc_loss": disc_loss,
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_grad_norm": grad_norm_mean,
            "disc_agent_acc": disc_agent_acc.detach(),
            "disc_demo_acc": disc_demo_acc.detach(),
            "disc_agent_logit": torch.mean(disc_agent_score).detach(),
            "disc_demo_logit": torch.mean(disc_demo_score).detach(),
        }
        disc_info.update(w_info)

        if (self._disc_logit_reg != 0):
            logit_weights = self._model.get_disc_logit_weights()
            disc_logit_loss = torch.sum(torch.square(logit_weights))
            disc_loss = disc_loss + self._disc_logit_reg * disc_logit_loss
            disc_info["disc_logit_loss"] = disc_logit_loss.detach()
            disc_info["disc_loss"] = disc_loss

        return disc_info

    def _calc_disc_rewards(self, norm_disc_obs):
        with torch.no_grad():
            disc_inputs = {"disc_obs": norm_disc_obs}
            disc_scores = torch_util.eval_minibatch(self._model.eval_disc, disc_inputs,
                                                    self._disc_eval_batch_size)
            disc_scores = disc_scores.squeeze(-1)
            disc_r = wgan_util.compute_wamp_disc_rewards(disc_scores, self._disc_score_scale,
                                                         self._disc_reward_scale)
        return disc_r
