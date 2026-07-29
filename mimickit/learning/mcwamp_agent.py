import torch

import envs.base_env as base_env
import learning.mcwamp_model as mcwamp_model
import learning.multi_critic_util as multi_critic_util
import learning.rl_util as rl_util
import learning.wamp_agent as wamp_agent
import util.mp_util as mp_util
import util.torch_util as torch_util


class MCWAMPAgent(wamp_agent.WAMPAgent):
    """WAMP agent with multi-critic advantage estimation (arXiv:2511.03996).

    Instead of mixing task and style rewards into a single scalar before PPO,
    each stream keeps its own critic, TD(lambda) return, and advantage. The
    advantages are standardized per stream and combined with fixed weights
    (default 2:1 goal:aux), which makes the relative task:style pressure on
    the policy independent of the raw scale of either reward. Test-mode
    episode returns still use the task/disc reward mixture, so Test_Return
    stays comparable to the single-critic WAMP agent.
    """

    def _load_params(self, config):
        super()._load_params(config)
        critic_weights = config.get("critic_weights", [2.0, 1.0])
        assert len(critic_weights) == 2, \
            "critic_weights must be [goal_weight, aux_weight]"
        self._critic_weights = [float(w) for w in critic_weights]
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = mcwamp_model.MCWAMPModel(model_config, self._env)
        return

    def _compute_rewards(self):
        # Keep the task reward in "reward" and store the style reward in its
        # own stream instead of overwriting the buffer with a mixture.
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")

        norm_disc_obs = self._disc_obs_norm.normalize(disc_obs)
        disc_r = self._calc_disc_rewards(norm_disc_obs)
        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)

        self._exp_buffer.set_data_flat("disc_reward", disc_r)

        info = {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std
        }
        return info

    def _build_train_data(self):
        self._record_disc_demo_data()
        self._store_disc_replay_data()

        reward_info = self._compute_rewards()
        info = self._build_multi_critic_train_data()

        info = {**info, **reward_info}
        return info

    def _build_multi_critic_train_data(self):
        self.eval()

        obs = self._exp_buffer.get_data("obs")
        next_obs = self._exp_buffer.get_data("next_obs")
        task_r = self._exp_buffer.get_data("reward")
        disc_r = self._exp_buffer.get_data("disc_reward")
        done = self._exp_buffer.get_data("done")
        rand_action_mask = self._exp_buffer.get_data("rand_action_mask")

        norm_next_obs = self._obs_norm.normalize(next_obs)
        next_critic_inputs = {"obs": norm_next_obs}
        next_vals = torch_util.eval_minibatch(self._model.eval_critic, next_critic_inputs,
                                              self._critic_eval_batch_size)
        next_vals = next_vals.detach()

        succ_mask = (done == base_env.DoneFlags.SUCC.value)
        fail_mask = (done == base_env.DoneFlags.FAIL.value)

        # Goal stream bootstraps with the env's terminal task rewards. The
        # style stream yields no further reward after termination, and the
        # WAMP style reward is bounded in [0, 1], so 0 is the exact terminal
        # value for both outcomes.
        next_vals[..., 0][succ_mask] = self._compute_succ_val()
        next_vals[..., 0][fail_mask] = self._compute_fail_val()
        next_vals[..., 1][succ_mask] = 0.0
        next_vals[..., 1][fail_mask] = 0.0

        rewards = torch.stack([task_r, disc_r], dim=-1)
        new_vals = torch.stack(
            [rl_util.compute_td_lambda_return(rewards[..., s], next_vals[..., s], done,
                                              self._discount, self._td_lambda)
             for s in range(rewards.shape[-1])], dim=-1)

        norm_obs = self._obs_norm.normalize(obs)
        critic_inputs = {"obs": norm_obs}
        vals = torch_util.eval_minibatch(self._model.eval_critic, critic_inputs,
                                         self._critic_eval_batch_size)
        vals = vals.detach()
        adv = new_vals - vals

        mask = (rand_action_mask == 1.0).flatten()
        norm_adv, adv_info = multi_critic_util.compute_multi_critic_adv(
            adv, self._critic_weights, mask, self._norm_adv_clip,
            calc_mean_std_fn=mp_util.calc_mean_std)

        self._exp_buffer.set_data("tar_val", new_vals)
        self._exp_buffer.set_data("adv", norm_adv)

        info = {
            "adv_mean": adv_info["adv_mean"],
            "adv_std": adv_info["adv_std"],
            "adv_goal_std": adv_info["adv0_std"],
            "adv_aux_std": adv_info["adv1_std"]
        }
        return info

    def _compute_critic_loss(self, batch):
        norm_obs = self._obs_norm.normalize(batch["obs"])
        tar_val = batch["tar_val"]
        pred = self._model.eval_critic(norm_obs)

        diff = tar_val - pred
        loss = torch.mean(torch.square(diff))

        info = {
            "critic_loss": loss
        }
        return info
