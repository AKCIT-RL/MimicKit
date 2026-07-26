"""Pure-torch utilities for the Wasserstein AMP (WAMP) discriminator.

Implements the soft-boundary Wasserstein-1 objective used by
"Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots"
(arXiv:2511.03996, Sec. 7) and originally introduced in HumanMimic
(arXiv:2309.14225, Eq. 9).

Sign convention (kept internally consistent; the two papers disagree in
the rendered reward sign): higher critic score = more demo-like.
- Critic loss:  L_D = -E[tanh(eta * D(x_demo))] + E[tanh(eta * D(x_agent))]
- Gradient penalty on interpolations x_hat = a*x_demo + (1-a)*x_agent,
  a ~ U(0,1):  L_grad = E[(||grad D(x_hat)|| - 1)^2]
- Style reward: r = 0.5 * (1 + tanh(eta * D(x_agent))) in [0, 1],
  monotonically increasing in the score. The affine shift keeps rewards
  non-negative so pure-imitation training does not incentivize early
  termination (HumanMimic uses exp(D) > 0 for the same reason).

These functions are kept free of any simulator/agent dependency so they
can be unit-tested on CPU.
"""

import torch


def compute_wasserstein_disc_loss(demo_scores, agent_scores, score_scale):
    """Soft-boundary Wasserstein critic loss.

    Args:
        demo_scores: [N] raw critic scores for demo (expert) samples.
        agent_scores: [M] raw critic scores for agent (policy) samples.
        score_scale: eta in tanh(eta * D), e.g. 0.4.
    Returns:
        (loss, info) where info holds detached diagnostics.
    """
    demo_tanh = torch.tanh(score_scale * demo_scores)
    agent_tanh = torch.tanh(score_scale * agent_scores)

    demo_mean = torch.mean(demo_tanh)
    agent_mean = torch.mean(agent_tanh)

    loss = -demo_mean + agent_mean
    info = {
        "disc_demo_tanh": demo_mean.detach(),
        "disc_agent_tanh": agent_mean.detach(),
        "disc_w_gap": (demo_mean - agent_mean).detach(),
    }
    return loss, info


def compute_interp_grad_penalty(disc_fn, demo_obs, agent_obs):
    """Two-sided WGAN-GP gradient penalty on random interpolations.

    Args:
        disc_fn: callable mapping [B, obs_dim] -> [B, 1] (or [B]) raw scores.
        demo_obs: [B, obs_dim] normalized demo observations.
        agent_obs: [B, obs_dim] normalized agent observations (same B).
    Returns:
        (penalty, grad_norm_mean) with graph retained for backprop.
    """
    assert demo_obs.shape == agent_obs.shape, \
        "GP interpolation requires matched demo/agent batches, got {} vs {}".format(
            tuple(demo_obs.shape), tuple(agent_obs.shape))

    alpha = torch.rand(demo_obs.shape[0], 1, device=demo_obs.device, dtype=demo_obs.dtype)
    interp_obs = alpha * demo_obs.detach() + (1.0 - alpha) * agent_obs.detach()
    interp_obs.requires_grad_(True)

    interp_scores = disc_fn(interp_obs)
    interp_scores = interp_scores.squeeze(-1)

    grad = torch.autograd.grad(interp_scores, interp_obs,
                               grad_outputs=torch.ones_like(interp_scores),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_norm = torch.linalg.norm(grad, dim=-1)
    penalty = torch.mean(torch.square(grad_norm - 1.0))
    return penalty, torch.mean(grad_norm).detach()


def compute_wamp_disc_rewards(scores, score_scale, reward_scale):
    """Bounded style reward, increasing in the critic score.

    r = reward_scale * 0.5 * (1 + tanh(score_scale * D)) in [0, reward_scale].
    """
    r = 0.5 * (1.0 + torch.tanh(score_scale * scores))
    return reward_scale * r


def update_score_stats_ema(mean, var, scores, alpha):
    """EMA update of running critic-score statistics.

    Tracks the (drifting) distribution of agent scores so the reward can be
    computed on standardized scores. Uses an exponential moving average
    instead of count-weighted running stats because the score distribution
    is non-stationary (the critic and the policy both move); count-weighted
    averages lag by O(num_iters) and cannot track the drift.

    Args:
        mean: [1] current running mean.
        var: [1] current running variance.
        scores: [N] raw critic scores of the current batch.
        alpha: EMA coefficient in (0, 1]; weight of the new batch.
    Returns:
        (new_mean, new_var) detached [1] tensors.
    """
    batch_mean = torch.mean(scores.detach())
    batch_var = torch.var(scores.detach(), unbiased=False)
    new_mean = (1.0 - alpha) * mean + alpha * batch_mean
    new_var = (1.0 - alpha) * var + alpha * batch_var
    return new_mean.reshape(1), new_var.reshape(1)


def normalize_scores(scores, mean, var, min_std=1e-3, z_clip=4.0):
    """Standardize critic scores with running stats, clamped to [-z_clip, z_clip].

    Makes the style reward invariant to additive/multiplicative drift of the
    critic scores: even when the absolute scores saturate the tanh boundary
    (e.g. mean -20), the relative ranking within the agent distribution is
    preserved and remapped into the informative region of the reward tanh.
    """
    std = torch.sqrt(torch.clamp_min(var, min_std * min_std))
    z = (scores - mean) / std
    return torch.clamp(z, -z_clip, z_clip)
