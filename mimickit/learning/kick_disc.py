"""Ponte de compatibilidade: o modulo agora vive em learning.mcwamp.kick_disc."""
import sys
import learning.mcwamp.kick_disc as _m
sys.modules[__name__] = _m
