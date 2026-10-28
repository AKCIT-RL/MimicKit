"""Explicit joint PD with Coulomb friction, the actuator model of Booster Gym
(envs/t1.py step(): friction = min(dof_friction, |tau|) * sign(tau); the
result is clipped at the torque limit). Kept free of isaacgym so it can be
unit-tested on the CPU.
"""

import torch


def pd_explicit_torque(kp, kd, tar_dof, dof_pos, dof_vel, friction=None):
    """kp*(tar - q) - kd*qd, minus a Coulomb friction that opposes the drive.

    friction None or all zero returns the plain PD (bit-identical). The friction
    term never reverses the torque: it removes min(friction, |tau|) in the
    direction of tau, so a drive weaker than the friction yields exactly 0.
    Clipping at the torque limit is the caller's (the engine clips every
    actuation torque, with or without friction).
    """
    torque = kp * (tar_dof - dof_pos) - kd * dof_vel
    if (friction is None):
        return torque
    fr = torch.minimum(friction, torch.abs(torque)) * torch.sign(torque)
    return torque - fr
