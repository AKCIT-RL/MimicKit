"""Ponte de compatibilidade: o modulo agora vive em envs.soccer.task_soccer_env."""
import sys
import envs.soccer.task_soccer_env as _m
sys.modules[__name__] = _m
