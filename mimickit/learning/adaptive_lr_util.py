"""Pure-torch utilities for KL-adaptive learning rate control.

Implements the training configuration of
"Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots"
(arXiv:2511.03996, Table 1): Adam with a learning rate driven by the
policy KL divergence toward a target of 0.01.

Two independent pieces:
- KL estimation. The k3 estimator of Schulman, "Approximating KL
  Divergence" (2020): KL ~= E[(r - 1) - log r], r = exp(logp - old_logp).
  Unlike E[-log r] it is non-negative for every sample, and it has lower
  variance, which matters because the controller reacts to a single
  minibatch estimate.
- LR adaptation. The rsl_rl convention that the paper references: a
  multiplicative controller with a dead band around the target, applied
  per minibatch and clamped to a fixed range.

The two are kept separate on purpose: `adapt_lr` takes an already
computed KL as a plain number, so it can be exercised with synthetic KL
sequences without building any tensor. Both functions are free of any
simulator/agent dependency so they can be unit-tested on CPU.
"""

import torch


def compute_approx_kl(a_logp, old_a_logp):
    """k3 estimator (Schulman): KL ≈ E[(r-1) - log r], r = exp(logp - old_logp).
    Non-negative by construction, lower variance than E[-log r]."""
    log_ratio = a_logp - old_a_logp
    ratio = torch.exp(log_ratio)
    return torch.mean((ratio - 1.0) - log_ratio)


def adapt_lr(lr, kl, desired_kl, lr_min=1e-5, lr_max=1e-2, factor=1.5):
    """rsl_rl rule: kl > 2*desired -> lr/factor; kl < desired/2 -> lr*factor;
    otherwise unchanged. Always clamped to [lr_min, lr_max]."""
    if kl > 2.0 * desired_kl:
        lr = lr / factor
    elif kl < desired_kl / 2.0:
        lr = lr * factor
    return max(lr_min, min(lr_max, lr))  