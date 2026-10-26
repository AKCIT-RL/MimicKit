"""Ponte de compatibilidade: o modulo agora vive em learning.mcwamp.mcwamp_model."""
import sys
import learning.mcwamp.mcwamp_model as _m
sys.modules[__name__] = _m
