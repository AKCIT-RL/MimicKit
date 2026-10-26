"""Ponte de compatibilidade: o modulo agora vive em envs.steering.task_steering_env."""
import sys
import envs.steering.task_steering_env as _m
sys.modules[__name__] = _m
