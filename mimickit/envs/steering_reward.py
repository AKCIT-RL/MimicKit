"""Regularization rewards for the steering task (arXiv:2511.03996, Table 4).

Pure torch, batch-first, no simulator state, so each term can be asserted on
its own in CPU tests.

Only the rows of Table 4 that apply to steering land here. Two notes on what
is deliberately absent:

  Stagnation   the paper penalizes standing still (-100 after 1 s nearly
               motionless) because its robot must always chase the ball.
               Steering is commanded to stop, so copying that term would fight
               the task. Not implemented, on purpose.

  Survival /   constant per-step bonus and the action-rate terms are separate
  action rate  variables in the campaign, not part of this change.

Deliberately independent from envs/soccer_util.py, which belongs to the soccer
track, even though the arithmetic is the same.
"""

import torch


@torch.jit.script
def compute_foot_proximity_penalty(left_foot_pos, right_foot_pos, min_dist):
    # type: (Tensor, Tensor, float) -> Tensor
    """Positive magnitude when the feet are closer than min_dist (planar).

    Zero once the feet are at least min_dist apart, so it is a one-sided
    penalty: it pushes the stance open and then stops acting. Paper weight -5.
    """
    d = torch.linalg.norm(left_foot_pos[..., 0:2] - right_foot_pos[..., 0:2], dim=-1)
    return torch.clamp_min(min_dist - d, 0.0)
