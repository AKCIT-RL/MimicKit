"""Ponte de compatibilidade: o modulo agora vive em learning.mcwamp.mcwamp_enc_model."""
import sys
import learning.mcwamp.mcwamp_enc_model as _m
sys.modules[__name__] = _m
