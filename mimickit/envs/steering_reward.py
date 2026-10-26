"""Ponte de compatibilidade: o modulo agora vive em envs.steering.steering_reward."""
import sys
import envs.steering.steering_reward as _m
sys.modules[__name__] = _m
