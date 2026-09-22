"""Domain randomization for the steering task (arXiv:2511.03996, Table 2).

Pure sampling: these functions take a numpy RandomState and the configured
ranges and return plain arrays. No simulator, no env state, so the whole thing
is testable on CPU and the sampled values can be asserted against their ranges.

Sampling is STATIC PER ENV, drawn once at build time, matching the pattern in
task_soccer_env._randomize_env_props: the Isaac Gym API requires mass, CoM and
shape properties to be set before the simulation is initialized. The motor
gains and the action delay could be resampled per episode, but keeping one
regime avoids two different notions of "randomized" inside one experiment.

The ball rows of Table 2 are absent on purpose: the steering task has no ball.

Caveat on `foot_compliance`, measured on this node and unresolved: the field
exists on gymapi.RigidShapeProperties, but writing it and reading it straight
back returns 0.0, while friction and restitution written the same way do come
back with the sampled values. Either the getter does not reflect it and PhysX
still uses it, or the backend drops it. It is sampled and applied anyway, so
this stays faithful to Table 2 -- but do NOT count it as verified randomization
until someone measures a contact difference.

Deliberately independent from envs/soccer_util.py, which belongs to the soccer
track.
"""

import numpy as np

# Table 2, minus the ball block. Motor stiffness/damping and motor bias are
# drawn PER DOF rather than once per robot: the table does not say which, and
# per-actuator variation is both the harder case and the physically likelier
# one (each motor has its own calibration).
DEFAULT_RANGES = {
    "action_delay_ms": (0.0, 20.0),
    "motor_bias_rad": (-0.05, 0.05),
    "motor_stiffness_scale": (0.95, 1.05),
    "motor_damping_scale": (0.95, 1.05),
    "torso_mass_scale": (0.95, 1.05),
    "torso_com_m": (-0.05, 0.05),
    "link_mass_scale": (0.98, 1.02),
    "link_com_m": (-0.005, 0.005),
    "foot_friction": (0.0, 1.0),
    "foot_compliance": (0.5, 1.5),
    "foot_restitution": (0.0, 1.0),
}

# what the critic observes (Table 3, "Mass randomization": mass and CoM of the
# base link) -- one scale plus a 3-vector offset
TORSO_PARAM_DIM = 4


def load_ranges(env_config):
    """Ranges from the env yaml, defaulting to Table 2.

    A key may be overridden with a [lo, hi] pair; an unknown key is an error
    rather than a silent no-op, because a typo in a yaml range would otherwise
    train against the default and nobody would notice.
    """
    ranges = {k: tuple(v) for k, v in DEFAULT_RANGES.items()}
    overrides = env_config.get("dr_ranges", {}) or {}
    for key, value in overrides.items():
        if (key not in ranges):
            raise KeyError(
                "unknown dr_ranges key '{}'. Known keys: {}".format(
                    key, sorted(ranges.keys())))
        assert len(value) == 2 and value[0] <= value[1], \
            "dr_ranges['{}'] must be [lo, hi] with lo <= hi, got {}".format(key, value)
        ranges[key] = (float(value[0]), float(value[1]))
    return ranges


def sample_env_params(rng, ranges, num_bodies, num_dofs, torso_body_id=0):
    """Draw one environment's physical parameters.

    Returns a dict with, per env:
      action_delay_ms  float
      motor_bias       [num_dofs]
      kp_scale         [num_dofs]
      kd_scale         [num_dofs]
      mass_scales      [num_bodies]      torso row from its own range
      com_offsets      [num_bodies, 3]   idem
      foot_friction / foot_compliance / foot_restitution  float
      torso_params     [4]               what the critic observes
    """
    def uniform(key, size=None):
        lo, hi = ranges[key]
        return rng.uniform(lo, hi, size=size)

    mass_scales = uniform("link_mass_scale", size=num_bodies)
    com_offsets = uniform("link_com_m", size=(num_bodies, 3))

    torso_mass_scale = float(uniform("torso_mass_scale"))
    torso_com = uniform("torso_com_m", size=3)
    mass_scales[torso_body_id] = torso_mass_scale
    com_offsets[torso_body_id] = torso_com

    return {
        "action_delay_ms": float(uniform("action_delay_ms")),
        "motor_bias": uniform("motor_bias_rad", size=num_dofs),
        "kp_scale": uniform("motor_stiffness_scale", size=num_dofs),
        "kd_scale": uniform("motor_damping_scale", size=num_dofs),
        "mass_scales": mass_scales,
        "com_offsets": com_offsets,
        "foot_friction": float(uniform("foot_friction")),
        "foot_compliance": float(uniform("foot_compliance")),
        "foot_restitution": float(uniform("foot_restitution")),
        "torso_params": np.concatenate([[torso_mass_scale], torso_com]),
    }


def substep_action_blend(delay_ms, num_substeps, control_dt):
    """Exact discretization of a zero-order-hold delayed by `delay_ms`.

    The engine recomputes the PD torque every substep but holds the joint
    target fixed across the control step, so a delay shorter than one control
    step has to be expressed as a per-substep blend between the previous action
    and the current one. Rounding to whole substeps would cover U(0, 20) ms
    badly: at 30 Hz a substep is 8.33 ms.

    Returns [num_substeps] weights for the CURRENT action; the previous action
    gets 1 - w. A substep fully inside the delay gets 0, one fully past it gets
    1, and the substep straddling the boundary gets the fraction of itself that
    lies past the delay.
    """
    substep_dt = control_dt / num_substeps
    delay = delay_ms * 1e-3

    weights = np.zeros(num_substeps, dtype=np.float64)
    for i in range(num_substeps):
        t0 = i * substep_dt
        t1 = t0 + substep_dt
        if (delay <= t0):
            weights[i] = 1.0
        elif (delay >= t1):
            weights[i] = 0.0
        else:
            weights[i] = (t1 - delay) / substep_dt
    return weights
