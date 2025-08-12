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
    extract_vpf as _core_extract_vpf,
    verify_vpf,
    compare_vpm,
    vpm_logic_and,
    vpm_logic_or,
    vpm_logic_not,
    vpm_logic_xor
)

from .utils import sha3_bytes
import hashlib
import json
import zlib
from typing import Tuple
from .core import VPF_FOOTER_MAGIC

# Convenience wrappers for deterministic, model-agnostic provenance demos
def _seed_bytes(seed_hex: str) -> bytes:
    try:
        return bytes.fromhex(seed_hex)
    except Exception:
        return hashlib.sha3_256(seed_hex.encode("utf-8")).digest()

def _artifact_from_seed(seed_hex: str, size: int = 4096) -> bytes:
    chunk = _seed_bytes(seed_hex)
    buf = bytearray()
    while len(buf) < size:
        buf.extend(chunk)
        chunk = hashlib.sha3_256(chunk).digest()
    return bytes(buf[:size])

def embed_vpf(prompt: str) -> Tuple[bytes, dict]:
    """Deterministically generate a binary artifact and embed a minimal VPF footer.

    Returns (artifact_bytes_with_footer, vpf_dict).
    """
    seed_hex = hashlib.sha3_256(prompt.encode("utf-8")).hexdigest()
    core = _artifact_from_seed(seed_hex)
    content_hash = f"sha3:{hashlib.sha3_256(core).hexdigest()}"
    vpf = create_vpf(
        pipeline={"graph_hash": "sha3:demo", "step": "demo.embed"},
        model={"id": "demo", "assets": {}},
        determinism={"seed_global": 0, "rng_backends": ["sha3"]},
        params={"seed_hex": seed_hex, "size_bytes": len(core)},
        inputs={"prompt": prompt, "prompt_hash": hashlib.sha3_256(prompt.encode()).hexdigest()},
        metrics={},
        lineage={"parents": [], "content_hash": content_hash},
        signature=None,
    )
    # Footer payload: zlib(compressed JSON)
    payload = zlib.compress(json.dumps(vpf, sort_keys=True).encode("utf-8"))
    footer = VPF_FOOTER_MAGIC + len(payload).to_bytes(4, "big") + payload
    return core + footer, vpf

def extract_vpf(artifact: bytes) -> dict:
    """Return only the VPF dict (drop metadata) for binary artifacts."""
    vpf, _meta = _core_extract_vpf(artifact)
    return vpf

def replay_from_vpf(vpf: dict) -> bytes:
    """Recreate the deterministic artifact bytes from the VPF.

    Uses params.seed_hex and params.size_bytes; appends the same VPF footer.
    """
    seed_hex = vpf.get("params", {}).get("seed_hex", "00" * 32)
    size = int(vpf.get("params", {}).get("size_bytes", 4096))
    core = _artifact_from_seed(seed_hex, size)
    payload = zlib.compress(json.dumps(vpf, sort_keys=True).encode("utf-8"))
    footer = VPF_FOOTER_MAGIC + len(payload).to_bytes(4, "big") + payload
    return core + footer
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