"""Ponte de compatibilidade: o modulo agora vive em learning.mcwamp.policy_export."""
import sys
import learning.mcwamp.policy_export as _m
sys.modules[__name__] = _m
