"""Ponte de compatibilidade: o modulo agora vive em envs.steering.steering_util."""
import sys
import envs.steering.steering_util as _m
sys.modules[__name__] = _m
