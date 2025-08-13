# zeromodel/metadata.py
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from io import BytesIO
from PIL import Image

# Core (legacy) VPM metadata reader – expects *its own* binary block, not PNG.
from zeromodel.vpm.metadata import VPMMetadata
# New provenance footer reader
from zeromodel.provenance.metadata import ProvenanceMetadata

@dataclass
class MetadataView:
    vpm: Optional[VPMMetadata]
    provenance: ProvenanceMetadata
    # Optional: add quick stripe info later if you want

def read_all_metadata(png_bytes: bytes) -> MetadataView:
    """
    Safe multiplexer:
      - Tries to read provenance footer (PNG bytes supported).
      - Tries VPMMetadata only if bytes are in its native block format.
        (Catches exceptions so PNG input won't explode.)
    """
    # 1) Provenance footer (always safe on PNG bytes)
    prov = ProvenanceMetadata.from_bytes(png_bytes)

    # 2) Legacy/core VPM metadata – only if buffer is a VPM block, not a PNG
    vpm_meta = None
    try:
        # Heuristic: VPMMetadata.from_bytes raises ValueError on non-VPM magic.
        vpm_meta = VPMMetadata.from_bytes(png_bytes)
    except Exception:
        vpm_meta = None

    return MetadataView(vpm=vpm_meta, provenance=prov)
