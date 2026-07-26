import torch

import learning.amp_agent as amp_agent
import learning.base_agent as base_agent
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

        if (self._disc_reward_norm):
            # Running stats of the raw agent critic scores (EMA, single GPU).
            # Registered as buffers so they persist in checkpoints.
            self.register_buffer("_disc_score_mean",
                                 torch.zeros([1], device=device, dtype=torch.float32))
            self.register_buffer("_disc_score_var",
                                 torch.ones([1], device=device, dtype=torch.float32))
        return

    def _load_params(self, config):
        super()._load_params(config)
        self._disc_score_scale = config["disc_score_scale"]
        # Reward can use a smaller scale than the critic loss so that the
        # tanh mapping stays in its linear regime even when the critic
        # saturates the loss boundary (|score| >> 1/disc_score_scale).
        self._disc_reward_score_scale = config.get("disc_reward_score_scale",
                                                   self._disc_score_scale)
        # Standardize scores with running stats before the reward tanh, making
        # the style reward invariant to critic-score drift (logits running to
        # e.g. -20 flatten the raw tanh reward to a constant ~0).
        self._disc_reward_norm = bool(config.get("disc_reward_norm", False))
        self._disc_reward_norm_alpha = float(config.get("disc_reward_norm_alpha", 0.05))
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

            if (self._disc_reward_norm):
                if (self._mode == base_agent.AgentMode.TRAIN):
                    # One large on-policy batch per iteration; test batches are
                    # small and must not contaminate the stats.
                    new_mean, new_var = wgan_util.update_score_stats_ema(
                        self._disc_score_mean, self._disc_score_var,
                        disc_scores, self._disc_reward_norm_alpha)
                    self._disc_score_mean[:] = new_mean
                    self._disc_score_var[:] = new_var

                disc_scores = wgan_util.normalize_scores(disc_scores,
                                                         self._disc_score_mean,
                                                         self._disc_score_var)

            disc_r = wgan_util.compute_wamp_disc_rewards(disc_scores, self._disc_reward_score_scale,
                                                         self._disc_reward_scale)
        return disc_r
