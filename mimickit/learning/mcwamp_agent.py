"""Ponte de compatibilidade: o modulo agora vive em learning.mcwamp.mcwamp_agent."""
import sys
import learning.mcwamp.mcwamp_agent as _m
sys.modules[__name__] = _m
