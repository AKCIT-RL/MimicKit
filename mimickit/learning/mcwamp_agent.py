import torch

import envs.base_env as base_env
import learning.mcwamp_model as mcwamp_model
import learning.multi_critic_util as multi_critic_util
import learning.normalizer as normalizer
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

        # weight of the AMP style reward inside the aux stream. Envs that
        # publish an "aux_reward" (e.g. soccer regularizations, Table 3)
        # get aux = aux_style_weight * disc_r + aux_reward; envs without it
        # keep the pure style stream.
        self._aux_style_weight = float(config.get("aux_style_weight", 1.0))
        self._has_aux_env_reward = False

        # Per-stream reward scale applied before TD(lambda). Because the
        # advantages are standardized per stream, a uniform scale on a
        # stream leaves the policy gradient unchanged -- this only keeps
        # the critic targets (and hence the critic loss) well-conditioned
        # when the env uses large raw reward weights (e.g. soccer Table 3).
        critic_reward_scales = config.get("critic_reward_scales", [1.0, 1.0])
        assert len(critic_reward_scales) == 2, \
            "critic_reward_scales must be [goal_scale, aux_scale]"
        self._critic_reward_scales = [float(s) for s in critic_reward_scales]
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = mcwamp_model.MCWAMPModel(model_config, self._env)
        return

    def _build_normalizers(self):
        super()._build_normalizers()
        # asymmetric critic (paper Table 2): when the env publishes a
        # privileged critic_obs (same layout as the obs, true ball state),
        # the critics train on it with their own normalizer while the actor
        # keeps the perceived obs.
        self._use_critic_obs = bool(getattr(self._env, "has_critic_obs", lambda: False)())
        if (self._use_critic_obs):
            obs_space = self._env.get_obs_space()
            obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
            self._critic_obs_norm = normalizer.Normalizer(
                obs_space.shape, clip=10.0, device=self._device, dtype=obs_dtype)
        return

    def load_state_dict(self, state_dict):
        # warm-start compat: checkpoints saved before the asymmetric critic
        # lack the critic_obs normalizer; seed it from the obs normalizer
        # (identical layout, so the stats are a valid starting point).
        if (self._use_critic_obs):
            for key in list(self.state_dict().keys()):
                if (key.startswith("_critic_obs_norm.") and key not in state_dict):
                    src = "_obs_norm." + key[len("_critic_obs_norm."):]
                    state_dict[key] = state_dict[src].clone()
        else:
            # eval of an asymmetric-critic checkpoint in an env without a
            # critic_obs: the normalizer does not exist here, drop its keys
            state_dict = {k: v for k, v in state_dict.items()
                          if not k.startswith("_critic_obs_norm.")}
        super().load_state_dict(state_dict)
        return

    def _update_normalizers(self):
        super()._update_normalizers()
        if (self._use_critic_obs):
            self._critic_obs_norm.update()
        return

    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)
        if (self._use_critic_obs):
            critic_obs = info["critic_obs"]
            self._exp_buffer.record("critic_obs", critic_obs)
            if (self._need_normalizer_update()):
                self._critic_obs_norm.record(critic_obs)
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)

        if ("aux_reward" in next_info):
            self._exp_buffer.record("aux_env_reward", next_info["aux_reward"])
            self._has_aux_env_reward = True
        if (self._use_critic_obs):
            self._exp_buffer.record("next_critic_obs", next_info["critic_obs"])
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

        if (self._use_critic_obs):
            obs = self._exp_buffer.get_data("critic_obs")
            next_obs = self._exp_buffer.get_data("next_critic_obs")
            obs_norm = self._critic_obs_norm
        else:
            obs = self._exp_buffer.get_data("obs")
            next_obs = self._exp_buffer.get_data("next_obs")
            obs_norm = self._obs_norm
        task_r = self._exp_buffer.get_data("reward")
        disc_r = self._exp_buffer.get_data("disc_reward")
        done = self._exp_buffer.get_data("done")
        rand_action_mask = self._exp_buffer.get_data("rand_action_mask")

        aux_r = self._aux_style_weight * disc_r
        if (self._has_aux_env_reward):
            aux_r = aux_r + self._exp_buffer.get_data("aux_env_reward")

        # per-stream reward means (pre critic scaling) for diagnostics
        task_reward_mean = task_r.mean().detach()
        aux_reward_mean = aux_r.mean().detach()

        goal_scale = self._critic_reward_scales[0]
        aux_scale = self._critic_reward_scales[1]
        task_r = goal_scale * task_r
        aux_r = aux_scale * aux_r

        norm_next_obs = obs_norm.normalize(next_obs)
        next_critic_inputs = {"obs": norm_next_obs}
        next_vals = torch_util.eval_minibatch(self._model.eval_critic, next_critic_inputs,
                                              self._critic_eval_batch_size)
        next_vals = next_vals.detach()

        succ_mask = (done == base_env.DoneFlags.SUCC.value)
        fail_mask = (done == base_env.DoneFlags.FAIL.value)

        # Goal stream bootstraps with the env's terminal task rewards. The
        # aux stream (style + env regularizations, including any terminal
        # penalty paid on the final step) yields no further reward after
        # termination, so 0 is the exact terminal value for both outcomes.
        next_vals[..., 0][succ_mask] = goal_scale * self._compute_succ_val()
        next_vals[..., 0][fail_mask] = goal_scale * self._compute_fail_val()
        next_vals[..., 1][succ_mask] = 0.0
        next_vals[..., 1][fail_mask] = 0.0

        rewards = torch.stack([task_r, aux_r], dim=-1)
        new_vals = torch.stack(
            [rl_util.compute_td_lambda_return(rewards[..., s], next_vals[..., s], done,
                                              self._discount, self._td_lambda)
             for s in range(rewards.shape[-1])], dim=-1)

        norm_obs = obs_norm.normalize(obs)
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
            "adv_aux_std": adv_info["adv1_std"],
            "task_reward_mean": task_reward_mean,
            "aux_reward_mean": aux_reward_mean
        }
        return info

    def _compute_critic_loss(self, batch):
        if (self._use_critic_obs):
            norm_obs = self._critic_obs_norm.normalize(batch["critic_obs"])
        else:
            norm_obs = self._obs_norm.normalize(batch["obs"])
        tar_val = batch["tar_val"]
        pred = self._model.eval_critic(norm_obs)

        diff = tar_val - pred
        loss = torch.mean(torch.square(diff))

        # per-stream decomposition (diagnostics only; gradients flow through
        # the combined loss above)
        sq = torch.square(diff.detach())
        loss_goal = torch.mean(sq[..., 0])
        loss_aux = torch.mean(sq[..., 1])

        info = {
            "critic_loss": loss,
            "critic_loss_goal": loss_goal,
            "critic_loss_aux": loss_aux
        }
        return info
