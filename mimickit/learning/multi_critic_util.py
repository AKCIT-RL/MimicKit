"""Multi-critic advantage utilities (arXiv:2511.03996).

The paper trains one critic per reward stream (goal/task and auxiliary/style),
normalizes each advantage independently, and combines them with fixed weights:

    A_total = w_goal * norm(A_goal) + w_aux * norm(A_aux)

Because each stream is standardized to unit variance before weighting, the
relative task:style pressure on the policy is fixed by the weights regardless
of the raw scale (or collapse) of either reward signal.

Pure torch, no simulator dependencies.
"""

import torch


def _calc_mean_std(x):
    std, mean = torch.std_mean(x)
    return mean, std


def compute_multi_critic_adv(adv, weights, rand_action_mask, adv_clip,
                             calc_mean_std_fn=None, min_std=1e-5):
    """Combine per-stream advantages into a single normalized advantage.

    Args:
        adv: [..., num_streams] per-stream advantages (e.g. [T, B, S]).
        weights: sequence of num_streams floats (e.g. [2.0, 1.0]).
        rand_action_mask: bool tensor over flattened leading dims; only these
            samples contribute to the normalization statistics (mirrors the
            PPO agent, which normalizes over exploratory actions only).
        adv_clip: symmetric clip applied to the final normalized advantage.
        calc_mean_std_fn: optional (tensor) -> (mean, std); pass
            mp_util.calc_mean_std for multi-process consistency. Defaults to
            plain torch statistics.
        min_std: std floor to keep degenerate (flat) streams finite.

    Returns:
        (norm_adv, info): norm_adv has shape adv.shape[:-1]. The combined
        advantage is re-standardized before clipping so the actor loss scale
        matches the single-critic pipeline; this rescaling preserves the
        weight ratios between streams. info holds per-stream and combined
        statistics (detached scalars).
    """
    if (calc_mean_std_fn is None):
        calc_mean_std_fn = _calc_mean_std

    num_streams = adv.shape[-1]
    assert len(weights) == num_streams, \
        "expected {} critic weights, got {}".format(num_streams, len(weights))

    combined = torch.zeros_like(adv[..., 0])
    info = {}
    for s in range(num_streams):
        stream_adv = adv[..., s]
        masked = stream_adv.flatten()[rand_action_mask]
        mean, std = calc_mean_std_fn(masked)
        norm = (stream_adv - mean) / torch.clamp_min(std, min_std)
        combined = combined + float(weights[s]) * norm

        info["adv{}_mean".format(s)] = mean.detach()
        info["adv{}_std".format(s)] = std.detach()

    masked_combined = combined.flatten()[rand_action_mask]
    c_mean, c_std = calc_mean_std_fn(masked_combined)
    norm_adv = (combined - c_mean) / torch.clamp_min(c_std, min_std)
    norm_adv = torch.clamp(norm_adv, -adv_clip, adv_clip)

    info["adv_mean"] = c_mean.detach()
    info["adv_std"] = c_std.detach()
    return norm_adv, info
