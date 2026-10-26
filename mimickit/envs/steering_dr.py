"""Ponte de compatibilidade: o modulo agora vive em envs.steering.steering_dr."""
import sys
import envs.steering.steering_dr as _m
sys.modules[__name__] = _m
