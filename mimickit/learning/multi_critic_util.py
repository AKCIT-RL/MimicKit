"""Ponte de compatibilidade: o modulo agora vive em learning.mcwamp.multi_critic_util."""
import sys
import learning.mcwamp.multi_critic_util as _m
sys.modules[__name__] = _m
