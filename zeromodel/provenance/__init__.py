# zeromodel/provenance/__init__.py
"""
ZeroModel Provenance: Universal AI Debugger

This package now focuses on two things:
  1) Universal tensor snapshot/restore to a visual carrier (VPM) and debugging helpers.
  2) A single, canonical VPF implementation (schema + PNG iTXt embed/extract) living in
     `zeromodel.images.vpf`.

There is intentionally NO duplicate VPF code here. Import VPF APIs from this package
and they will be forwarded to the single source of truth under `zeromodel.images`.
"""

# --- VPM (tensor snapshot + visual utilities) -------------------------------
from .core import (
    tensor_to_vpm,
    vpm_to_tensor,
    compare_vpm,
    vpm_logic_and,
    vpm_logic_or,
    vpm_logic_not,
    vpm_logic_xor,
)

# --- VPF (canonical API re-exported from zeromodel.images) ------------------
from zeromodel.images import (
    # Schema/dataclasses (optional to re-export; include if your code uses them)
    VPF, VPFPipeline, VPFModel, VPFDeterminism, VPFParams,
    VPFInputs, VPFMetrics, VPFLineage, VPFAuth,
    # Primary API
    create_vpf,
    embed_vpf,                     # returns PNG bytes with 'vpf' iTXt
    extract_vpf,                   # works from a PIL.Image (round-trips through PNG bytes)
    extract_vpf_from_png_bytes,    # works from raw PNG bytes
    verify_vpf,
    replay_from_vpf,
)

# --- Misc utilities kept here ----------------------------------------------
from .utils import sha3_bytes

__all__ = [
    # VPM / tensor snapshot
    "tensor_to_vpm",
    "vpm_to_tensor",
    "compare_vpm",
    "vpm_logic_and",
    "vpm_logic_or",
    "vpm_logic_not",
    "vpm_logic_xor",

    # VPF schema (optional to expose)
    "VPF", "VPFPipeline", "VPFModel", "VPFDeterminism", "VPFParams",
    "VPFInputs", "VPFMetrics", "VPFLineage", "VPFAuth",

    # VPF API (single source of truth)
    "create_vpf",
    "embed_vpf",
    "extract_vpf",
    "extract_vpf_from_png_bytes",
    "verify_vpf",
    "replay_from_vpf",

    # Utils
    "sha3_bytes",
]
