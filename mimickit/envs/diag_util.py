import torch


class DiagWindow:
    """Windowed accumulator for per-iteration env diagnostics.

    Two kinds of entries:
      - means: per-step batch means (e.g. reward terms), averaged over the
        number of recorded steps when popped;
      - sums: raw event counts/times (e.g. goals, episode ends), summed over
        the window.

    Everything stays on-device; scalars only cross to the CPU in ``pop``,
    which is called once per training iteration and resets the window.
    """

    def __init__(self, device):
        self._device = device
        self._mean_sums = dict()
        self._event_sums = dict()
        self._steps = 0

    def step(self):
        """Mark one env step recorded in the window. Call exactly once per step."""
        self._steps += 1
        return

    def add_mean(self, name, value):
        """Accumulate the batch mean of ``value`` (tensor [N], scalar tensor or float)."""
        if (not torch.is_tensor(value)):
            value = torch.tensor(float(value), device=self._device)
        value = value.detach().float()
        if (value.dim() > 0):
            value = value.mean()
        acc = self._mean_sums.get(name)
        if (acc is None):
            self._mean_sums[name] = value.clone()
        else:
            acc += value
        return

    def add_sum(self, name, value):
        """Accumulate a raw scalar sum (event counter / accumulated time)."""
        if (not torch.is_tensor(value)):
            value = torch.tensor(float(value), device=self._device)
        value = value.detach().float()
        if (value.dim() > 0):
            value = value.sum()
        acc = self._event_sums.get(name)
        if (acc is None):
            self._event_sums[name] = value.clone()
        else:
            acc += value
        return

    def pop(self):
        """Return (means, sums) as python-float dicts and reset the window."""
        steps = max(self._steps, 1)
        means = {k: (v / steps).item() for k, v in self._mean_sums.items()}
        sums = {k: v.item() for k, v in self._event_sums.items()}
        self._mean_sums = dict()
        self._event_sums = dict()
        self._steps = 0
        return means, sums


def update_reacquisition(prev_valid, valid, lost_since, time_buf):
    """Track how long the perceived ball stays lost before being reacquired.

    Args:
      prev_valid: [N] bool, perception mask on the previous step.
      valid: [N] bool, perception mask on this step.
      lost_since: [N] float, time at which the mask dropped, or -1.0 when the
        ball is not currently lost (or the episode was reset while lost).
      time_buf: [N] float, current episode time.

    Returns (new_lost_since, reacq_time_sum, reacq_count) where the scalar
    sum/count cover only reacquisitions completed on this step.
    """
    lost_now = prev_valid & ~valid
    reacq = (~prev_valid) & valid & (lost_since >= 0.0)

    durations = torch.where(reacq, time_buf - lost_since, torch.zeros_like(time_buf))
    reacq_time_sum = durations.sum()
    reacq_count = reacq.float().sum()

    new_lost_since = torch.where(lost_now, time_buf, lost_since)
    new_lost_since = torch.where(reacq, torch.full_like(lost_since, -1.0), new_lost_since)
    return new_lost_since, reacq_time_sum, reacq_count
