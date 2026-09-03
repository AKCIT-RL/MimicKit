import numpy as np
import torch

import envs.base_env as base_env
import learning.adaptive_lr_util as adaptive_lr_util
import learning.base_agent as base_agent
import learning.mirror_util as mirror_util
import learning.mp_optimizer as mp_optimizer
import learning.ppo_model as ppo_model
import learning.rl_util as rl_util
import util.mp_util as mp_util
import util.torch_util as torch_util
from util.logger import Logger

class PPOAgent(base_agent.BaseAgent):
    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _load_params(self, config):
        super()._load_params(config)
        
        self._actor_epochs = config["actor_epochs"]
        self._actor_batch_size = config["actor_batch_size"]
        self._critic_epochs = config["critic_epochs"]
        self._critic_batch_size = config["critic_batch_size"]

        self._td_lambda = config["td_lambda"]
        self._ppo_clip_ratio = config["ppo_clip_ratio"]
        self._norm_adv_clip = config["norm_adv_clip"]

        self._action_bound_weight = config["action_bound_weight"]
        self._action_entropy_weight = config["action_entropy_weight"]
        self._action_reg_weight = config["action_reg_weight"]

        self._critic_eval_batch_size = int(config.get("critic_eval_batch_size", 0))
        
        self._exp_anneal_samples = config.get("exp_anneal_samples", np.inf)
        self._exp_prob_beg = config.get("exp_prob_beg", 1.0)
        self._exp_prob_end = config.get("exp_prob_end", 1.0)

        # KL-adaptive learning rate (arXiv:2511.03996 Table 1, rsl_rl convention).
        # desired_kl None keeps the fixed-lr behaviour of every existing config.
        desired_kl = config.get("desired_kl", None)
        self._desired_kl = None if (desired_kl is None) else float(desired_kl)
        self._lr_min = float(config.get("lr_min", 1e-5))
        self._lr_max = float(config.get("lr_max", 1e-2))
        self._lr_adapt_factor = float(config.get("lr_adapt_factor", 1.5))
        self._adapt_critic_lr = bool(config.get("adapt_critic_lr", True))

        # Mirror symmetry loss (arXiv:2511.03996 Table 1). Weight 0 keeps every
        # existing config bit-identical; log_sym_residual measures the asymmetry
        # without penalising it, which is how a baseline number is obtained.
        self._mirror_sym_weight = float(config.get("mirror_sym_weight", 0.0))
        self._log_sym_residual = bool(config.get("log_sym_residual", False))
        self._mirror_ops = None
        return

    def _get_mirror_ops(self):
        """(obs_perm, obs_signs, act_perm, act_signs), built once from the env.

        Built lazily because it needs both the environment's kinematic model and
        this agent's action normalizer, which are ready only after __init__.
        Everything is read from the live env: deriving the layout from a copy of
        the config instead would let the two drift apart silently.
        """
        if (self._mirror_ops is not None):
            return self._mirror_ops

        env = self._env
        char_model = env._kin_char_model
        key_body_ids = getattr(env, "_key_body_ids", [])
        key_body_names = [char_model.get_body_name(int(i)) for i in key_body_ids]

        obs_perm, obs_signs = mirror_util.build_obs_mirror(
            task_key=type(env).__name__,
            char_model=char_model,
            key_body_names=key_body_names,
            root_height_obs=env._root_height_obs,
            obs_size=int(np.prod(env.get_obs_space().shape)))
        act_perm, act_signs = mirror_util.build_dof_mirror(char_model)

        assert(mirror_util.check_involution(obs_perm, obs_signs)), \
            "observation mirror is not an involution"
        assert(mirror_util.check_involution(act_perm, act_signs)), \
            "action mirror is not an involution"

        # The loss mirrors actions in NORMALIZED space when the normalizer
        # commutes with the mirror (exact for assets with mirror-image joint
        # limits, e.g. the G1). Otherwise (T1: left/right_hip_yaw share the
        # same asymmetric range) actions are mirrored in RAW space, which is
        # exact for any asset at the cost of two extra normalizer hops.
        ok, mean_err, std_err = mirror_util.check_normalizer_equivariance(
            self._a_norm.get_mean(), self._a_norm.get_std(), act_perm, act_signs)
        self._mirror_act_raw = not ok
        if (self._mirror_act_raw):
            Logger.print("Mirror loss: action normalizer does not commute with "
                         "the mirror (mean err {:.3e}, std err {:.3e}); mirroring "
                         "actions in raw space instead".format(mean_err, std_err))

        device = self._device
        self._mirror_ops = (mirror_util.to_tensors(obs_perm, obs_signs, device) +
                            mirror_util.to_tensors(act_perm, act_signs, device))
        return self._mirror_ops

    def _compute_sym_residual(self, raw_obs, a_dist):
        """Mean squared distance between the policy and its own mirror.

        The observation is mirrored BEFORE normalization: the observation
        normalizer tracks running statistics of an asymmetric policy, so it does
        not commute with the mirror. Actions are mirrored in normalized space
        when that commutes (checked in _get_mirror_ops) and in raw space
        otherwise.
        """
        obs_perm, obs_signs, act_perm, act_signs = self._get_mirror_ops()

        mirror_obs = mirror_util.mirror(raw_obs, obs_perm, obs_signs)
        norm_mirror_obs = self._obs_norm.normalize(mirror_obs)
        mirror_dist = self._model.eval_actor(norm_mirror_obs)

        if (self._mirror_act_raw):
            raw_mode = self._a_norm.unnormalize(mirror_dist.mode)
            mirror_a = self._a_norm.normalize(
                mirror_util.mirror(raw_mode, act_perm, act_signs))
        else:
            mirror_a = mirror_util.mirror(mirror_dist.mode, act_perm, act_signs)
        diff = a_dist.mode - mirror_a
        return torch.mean(torch.sum(torch.square(diff), dim=-1))

    def _build_model(self, config):
        model_config = config["model"]
        self._model = ppo_model.PPOModel(model_config, self._env)
        return
    
    def _build_optimizer(self, config):
        actor_config = config["actor_optimizer"]
        actor_params = list(self._model.get_actor_params())
        actor_params = [p for p in actor_params if p.requires_grad]
        self._actor_optimizer = mp_optimizer.MPOptimizer(actor_config, actor_params)
        
        critic_config = config["critic_optimizer"]
        critic_params = list(self._model.get_critic_params())
        critic_params = [p for p in critic_params if p.requires_grad]
        self._critic_optimizer = mp_optimizer.MPOptimizer(critic_config, critic_params)
        return
    
    def _sync_optimizer(self):
        self._actor_optimizer.sync()
        self._critic_optimizer.sync()
        return

    def _get_exp_buffer_length(self):
        return self._steps_per_iter
    
    def _init_iter(self):
        super()._init_iter()
        self._exp_buffer.reset()
        return

    def _decide_action(self, obs, info):
        norm_obs = self._obs_norm.normalize(obs)
        norm_action_dist = self._model.eval_actor(norm_obs)

        if (self._mode == base_agent.AgentMode.TRAIN):
            norm_a_rand = norm_action_dist.sample()
            norm_a_mode = norm_action_dist.mode

            exp_prob = self._get_exp_prob()
            exp_prob = torch.full([norm_a_rand.shape[0], 1], exp_prob, device=self._device, dtype=torch.float)
            rand_action_mask = torch.bernoulli(exp_prob)
            norm_a = torch.where(rand_action_mask == 1.0, norm_a_rand, norm_a_mode)
            rand_action_mask = rand_action_mask.squeeze(-1)

        elif (self._mode == base_agent.AgentMode.TEST):
            norm_a = norm_action_dist.mode
            rand_action_mask = torch.zeros_like(norm_a[..., 0])
            
        else:
            assert(False), "Unsupported agent mode: {}".format(self._mode)
            
        norm_a_logp = norm_action_dist.log_prob(norm_a)

        norm_a = norm_a.detach()
        norm_a_logp = norm_a_logp.detach()
        a = self._a_norm.unnormalize(norm_a)

        a_info = {
            "a_logp": norm_a_logp,
            "rand_action_mask": rand_action_mask
        }
        return a, a_info

    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)
        self._exp_buffer.record("a_logp", action_info["a_logp"])
        self._exp_buffer.record("rand_action_mask", action_info["rand_action_mask"])
        return
    
    def _build_train_data(self):
        self.eval()
        
        obs = self._exp_buffer.get_data("obs")
        next_obs = self._exp_buffer.get_data("next_obs")
        r = self._exp_buffer.get_data("reward")
        done = self._exp_buffer.get_data("done")
        rand_action_mask = self._exp_buffer.get_data("rand_action_mask")
        
        norm_next_obs = self._obs_norm.normalize(next_obs)
        next_critic_inputs = {"obs": norm_next_obs}
        next_vals = torch_util.eval_minibatch(self._model.eval_critic, next_critic_inputs, self._critic_eval_batch_size)
        next_vals = next_vals.squeeze(-1).detach()

        succ_val = self._compute_succ_val()
        succ_mask = (done == base_env.DoneFlags.SUCC.value)
        next_vals[succ_mask] = succ_val

        fail_val = self._compute_fail_val()
        fail_mask = (done == base_env.DoneFlags.FAIL.value)
        next_vals[fail_mask] = fail_val

        new_vals = rl_util.compute_td_lambda_return(r, next_vals, done, self._discount, self._td_lambda)

        norm_obs = self._obs_norm.normalize(obs)
        critic_inputs = {"obs": norm_obs}
        vals = torch_util.eval_minibatch(self._model.eval_critic, critic_inputs, self._critic_eval_batch_size)
        vals = vals.squeeze(-1).detach()
        adv = new_vals - vals
        
        rand_action_mask = (rand_action_mask == 1.0).flatten()
        adv_flat = adv.flatten()
        rand_action_adv = adv_flat[rand_action_mask]
        adv_mean, adv_std = mp_util.calc_mean_std(rand_action_adv)
        norm_adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-5)
        norm_adv = torch.clamp(norm_adv, -self._norm_adv_clip, self._norm_adv_clip)
        
        self._exp_buffer.set_data("tar_val", new_vals)
        self._exp_buffer.set_data("adv", norm_adv)
        
        info = {
            "adv_mean": adv_mean,
            "adv_std": adv_std
        }
        return info
    
    def _get_exp_prob(self):
        if (np.isfinite(self._exp_anneal_samples)):
            samples = self._sample_count
            l = float(samples) / self._exp_anneal_samples
            l = np.clip(l, 0.0, 1.0)
            prob = (1.0 - l) * self._exp_prob_beg + l * self._exp_prob_end
        else:
            prob = self._exp_prob_beg
        return prob

    def _update_model(self):
        self.train()
        
        num_envs = self.get_num_envs()
        num_samples = self._exp_buffer.get_sample_count()

        critic_batch_size = int(np.ceil(self._critic_batch_size * num_envs))
        num_critic_batches = int(np.ceil(float(num_samples) / critic_batch_size))
        num_critic_steps = num_critic_batches * self._critic_epochs
        critic_info = self._update_critic(critic_batch_size, num_critic_steps)
        
        actor_batch_size = int(np.ceil(self._actor_batch_size * num_envs))
        num_actor_batches = int(np.ceil(float(num_samples) / actor_batch_size))
        num_actor_steps = num_actor_batches * self._actor_epochs
        actor_info = self._update_actor(actor_batch_size, num_actor_steps)
        
        train_info = {**critic_info, **actor_info}
        return train_info

    def _update_critic(self, batch_size, steps):
        info = dict()
        device_type = torch.device(self._device).type

        for i in range(steps):
            batch = self._exp_buffer.sample(batch_size)

            with torch.amp.autocast(device_type=device_type, enabled=self._use_mixed_precision, dtype=torch.bfloat16):
                loss_info = self._compute_critic_loss(batch)
                loss = loss_info["critic_loss"]

            self._critic_optimizer.step(loss)

            torch_util.add_torch_dict(loss_info, info)
        
        torch_util.scale_torch_dict(1.0 / steps, info)
        return info

    def _update_actor(self, batch_size, num_steps):
        info = dict()
        device_type = torch.device(self._device).type

        for i in range(num_steps):
            batch = self._exp_buffer.sample(batch_size)

            with torch.amp.autocast(device_type=device_type, enabled=self._use_mixed_precision, dtype=torch.bfloat16):
                loss_info = self._compute_actor_loss(batch)
                loss = loss_info["actor_loss"]

            self._actor_optimizer.step(loss)

            # ASEAgent reimplements _compute_actor_loss instead of extending it,
            # so it reports no KL. Absent key => fixed lr, which is what every
            # config without desired_kl expects anyway.
            approx_kl = loss_info.get("approx_kl", None)
            self._update_lr(approx_kl)

            loss_info["actor_lr"] = torch.tensor(self._actor_optimizer.get_lr(), device=self._device)
            torch_util.add_torch_dict(loss_info, info)

        torch_util.scale_torch_dict(1.0 / num_steps, info)

        # after the scaling, so this stays the lr the next iteration starts from
        # rather than the mean over the minibatches
        info["actor_lr_final"] = torch.tensor(self._actor_optimizer.get_lr(), device=self._device)
        return info

    def _update_lr(self, kl):
        if (self._desired_kl is None):
            return

        assert(kl is not None), \
            "desired_kl is set but {} does not report approx_kl from _compute_actor_loss".format(
                type(self).__name__)

        lr = adaptive_lr_util.adapt_lr(self._actor_optimizer.get_lr(), kl.item(), self._desired_kl,
                                       lr_min=self._lr_min, lr_max=self._lr_max,
                                       factor=self._lr_adapt_factor)
        self._actor_optimizer.set_lr(lr)

        # rsl_rl drives one optimizer for actor+critic; here they are separate,
        # so the critic mirrors the actor lr. The disc optimizer is left alone:
        # its dynamics are not governed by the policy KL.
        if (self._adapt_critic_lr):
            self._critic_optimizer.set_lr(lr)
        return
    
    def _compute_critic_loss(self, batch):
        norm_obs = self._obs_norm.normalize(batch["obs"])
        tar_val = batch["tar_val"]
        pred = self._model.eval_critic(norm_obs)
        pred = pred.squeeze(-1)

        diff = tar_val - pred
        loss = torch.mean(torch.square(diff))

        info = {
            "critic_loss": loss
        }
        return info

    def _compute_actor_loss(self, batch):
        norm_obs = self._obs_norm.normalize(batch["obs"])
        norm_a = self._a_norm.normalize(batch["action"])
        old_a_logp = batch["a_logp"]
        adv = batch["adv"]
        rand_action_mask = batch["rand_action_mask"]

        # loss should only be computed using samples with random actions
        rand_action_mask = (rand_action_mask == 1.0)
        norm_obs = norm_obs[rand_action_mask]
        norm_a = norm_a[rand_action_mask]
        old_a_logp = old_a_logp[rand_action_mask]
        adv = adv[rand_action_mask]

        a_dist = self._model.eval_actor(norm_obs)
        a_logp = a_dist.log_prob(norm_a)

        a_ratio = torch.exp(a_logp - old_a_logp)
        actor_loss0 = adv * a_ratio
        actor_loss1 = adv * torch.clamp(a_ratio, 1.0 - self._ppo_clip_ratio, 1.0 + self._ppo_clip_ratio)
        actor_loss = torch.minimum(actor_loss0, actor_loss1)
        actor_loss = -torch.mean(actor_loss)
        
        clip_frac = (torch.abs(a_ratio - 1.0) > self._ppo_clip_ratio).type(torch.float)
        clip_frac = torch.mean(clip_frac)
        imp_ratio = torch.mean(a_ratio)
        
        # computed on the same masked subset as the loss, so the lr controller
        # reacts to the samples that actually produced the gradient
        approx_kl = adaptive_lr_util.compute_approx_kl(a_logp, old_a_logp)

        info = {
            "actor_loss": actor_loss,
            "clip_frac": clip_frac.detach(),
            "imp_ratio": imp_ratio.detach(),
            "approx_kl": approx_kl.detach()
        }

        if (self._action_bound_weight != 0):
            action_bound_loss = self._compute_action_bound_loss(a_dist)
            if (action_bound_loss is not None):
                action_bound_loss = torch.mean(action_bound_loss)
                actor_loss += self._action_bound_weight * action_bound_loss
                info["action_bound_loss"] = action_bound_loss.detach()

        if (self._action_entropy_weight != 0):
            action_entropy = a_dist.entropy()
            action_entropy = torch.mean(action_entropy)
            actor_loss += -self._action_entropy_weight * action_entropy
            info["action_entropy"] = action_entropy.detach()
        
        if (self._action_reg_weight != 0):
            action_reg_loss = a_dist.param_reg()
            action_reg_loss = torch.mean(action_reg_loss)
            actor_loss += self._action_reg_weight * action_reg_loss
            info["action_reg_loss"] = action_reg_loss.detach()

        if (self._mirror_sym_weight != 0 or self._log_sym_residual):
            # same masked subset as the loss above, so the regularizer does not
            # change the effective batch
            raw_obs = batch["obs"][rand_action_mask]
            sym_residual = self._compute_sym_residual(raw_obs, a_dist)
            info["sym_residual"] = sym_residual.detach()

            if (self._mirror_sym_weight != 0):
                actor_loss += self._mirror_sym_weight * sym_residual

        return info

    def _log_train_info(self, train_info, test_info, env_diag_info, start_time):
        super()._log_train_info(train_info, test_info, env_diag_info, start_time)
        self._logger.log("Exp_Prob", self._get_exp_prob())
        return