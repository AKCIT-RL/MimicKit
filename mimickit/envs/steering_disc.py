"""Ponte de compatibilidade: o modulo agora vive em envs.steering.steering_disc."""
import sys
import envs.steering.steering_disc as _m
sys.modules[__name__] = _m
