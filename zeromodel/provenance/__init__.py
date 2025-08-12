# zeromodel/provenance/__init__.py
"""
ZeroModel Provenance: Universal AI Debugger

This module implements the true vision of ZeroModel as a model-agnostic debugger
that captures and restores the exact state of ANY AI process at ANY point.

Core Principles:
- Intelligence lives in the data structure, not the model
- The medium is the message (McLuhan)
- No model at decision time - the intelligence is in the VPM
- Universal tensor snapshot capability
- Visual debugging of AI state
"""

from .core import (
    tensor_to_vpm,
    vpm_to_tensor,
    create_vpf,
    embed_vpf,
    extract_vpf,
    verify_vpf,
    compare_vpm,
    vpm_logic_and,
    vpm_logic_or,
    vpm_logic_not,
    vpm_logic_xor
)

from .utils import sha3_bytes
__all__ = [
    'tensor_to_vpm',
    'vpm_to_tensor',
    'create_vpf',
    'embed_vpf',
    'extract_vpf',
    'verify_vpf',
    'replay_from_vpf',
    'compare_vpm',
    'vpm_logic_and',
    'vpm_logic_or',
    'vpm_logic_not',
    'vpm_logic_xor',
    'sha3_bytes'
]