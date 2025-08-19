# zeromodel/provenance/metadata.py
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

VPF_MAGIC_HEADER = b"VPF1"
VPF_FOOTER_MAGIC = b"ZMVF"


def _sha3_hex(b: bytes) -> str:
    return hashlib.sha3_256(b).hexdigest()


@dataclass
class ProvenanceMetadata:
    vpf: Optional[Dict[str, Any]] = None
    core_sha3: Optional[str] = None
    has_tensor_vpm: bool = False

    @classmethod
    def from_bytes(cls, data: bytes) -> "ProvenanceMetadata":
        meta = cls()
        idx = data.rfind(VPF_FOOTER_MAGIC)
        if idx == -1 or idx + 8 > len(data):
            return meta  # no provenance footer

        total_len = struct.unpack(">I", data[idx + 4 : idx + 8])[0]
        end = idx + 8 + total_len
        if end > len(data):
            return meta  # malformed

        buf = memoryview(data)[idx + 8 : end]

        # Preferred container: VPF1 | u32 | zlib(JSON)
        if len(buf) >= 8 and bytes(buf[:4]) == VPF_MAGIC_HEADER:
            comp_len = struct.unpack(">I", bytes(buf[4:8]))[0]
            comp_end = 8 + comp_len
            vpf_json = zlib.decompress(bytes(buf[8:comp_end]))
            meta.vpf = json.loads(vpf_json)

            # Optional tensor segment
            rest = bytes(buf[comp_end:])
            if len(rest) >= 8 and rest.startswith(b"TNSR"):
                tlen = struct.unpack(">I", rest[4:8])[0]
                meta.has_tensor_vpm = len(rest) >= 8 + tlen

            core = data[:idx]
            meta.core_sha3 = _sha3_hex(core)
            return meta

        # Legacy: footer was just zlib(JSON)
        try:
            vpf_json = zlib.decompress(bytes(buf))
            meta.vpf = json.loads(vpf_json)
            core = data[:idx]
            meta.core_sha3 = _sha3_hex(core)
        except Exception:
            pass
        return meta
