"""Ponte de compatibilidade: o modulo agora vive em envs.soccer.soccer_util."""
import sys
import envs.soccer.soccer_util as _m
sys.modules[__name__] = _m
