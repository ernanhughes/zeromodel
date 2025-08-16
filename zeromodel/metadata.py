# zeromodel/metadata.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Dict, Optional, Union

# New provenance footer reader (PNG-safe)
from zeromodel.images.metadata import ProvenanceMetadata
# Core (legacy) VPM metadata reader – expects *its own* binary block, not PNG.
from zeromodel.vpm.metadata import VPMMetadata

SrcType = Union[str, Path, bytes, bytearray, IO[bytes]]


@dataclass
class MetadataView:
    vpm: Optional[VPMMetadata]
    provenance: Optional[ProvenanceMetadata]

    def to_dict(self) -> Dict[str, Any]:
        def _maybe_to_dict(obj):
            if obj is None:
                return None
            # Prefer explicit to_dict if available
            if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
                return obj.to_dict()
            # Fall back to __dict__ (best effort)
            try:
                return dict(obj.__dict__)  # type: ignore[attr-defined]
            except Exception:
                return str(obj)
        return {
            "vpm": _maybe_to_dict(self.vpm),
            "provenance": _maybe_to_dict(self.provenance),
        }

    def pretty(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    # --- Convenience constructors ---

    @classmethod
    def from_bytes(cls, data: bytes) -> "MetadataView":
        prov = None
        vpm_meta = None

        # 1) Provenance footer is always safe on PNG bytes (no-op if not present)
        try:
            prov = ProvenanceMetadata.from_bytes(data)
        except Exception:
            prov = None

        # 2) Legacy/core VPM metadata – only if buffer is a VPM block, not a PNG
        try:
            # Heuristic: VPMMetadata.from_bytes raises on non-VPM magic
            vpm_meta = VPMMetadata.from_bytes(data)
        except Exception:
            vpm_meta = None

        return cls(vpm=vpm_meta, provenance=prov)

    @classmethod
    def from_png(cls, path: Union[str, Path]) -> "MetadataView":
        b = _read_all_bytes(path)
        return cls.from_bytes(b)


def read_all_metadata(src: SrcType) -> MetadataView:
    """
    Universal metadata reader.

    Accepts:
      - PNG file path (str/Path)
      - raw bytes / bytearray
      - binary file-like (opened in 'rb')

    Returns:
      MetadataView(vpm=?, provenance=?)

    Behavior:
      - Tries to parse provenance footer first (safe on PNGs; returns None if absent)
      - Attempts legacy VPM block parsing only if bytes match its expected magic
    """
    data = _coerce_to_bytes(src)
    return MetadataView.from_bytes(data)


# ------------------------
# Helpers
# ------------------------

def _coerce_to_bytes(src: SrcType) -> bytes:
    if isinstance(src, (bytes, bytearray)):
        return bytes(src)
    if isinstance(src, (str, Path)):
        return _read_all_bytes(src)
    if hasattr(src, "read"):
        # file-like
        buf = src.read()
        if not isinstance(buf, (bytes, bytearray)):
            raise TypeError("file-like object did not return bytes")
        return bytes(buf)
    raise TypeError(f"Unsupported source type for read_all_metadata: {type(src)!r}")


def _read_all_bytes(path: Union[str, Path]) -> bytes:
    p = Path(path)
    with p.open("rb") as f:
        return f.read()
