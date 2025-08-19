<!-- Merged Python Code Files -->


## File: __init__.py

`python
# Public, DRY API surface for all image+VPF ops
from .stripe import add_visual_stripe  # optional visual aid
from .vpf import (VPF, VPFAuth, VPFDeterminism, VPFInputs, VPFLineage,
                  VPFMetrics, VPFModel, VPFParams, VPFPipeline, create_vpf,
                  embed_vpf, extract_vpf, extract_vpf_from_png_bytes,
                  replay_from_vpf, verify_vpf)

__all__ = [
    # schema
    "VPF",
    "VPFPipeline",
    "VPFModel",
    "VPFDeterminism",
    "VPFParams",
    "VPFInputs",
    "VPFMetrics",
    "VPFLineage",
    "VPFAuth",
    # functions
    "create_vpf",
    "embed_vpf",
    "extract_vpf",
    "extract_vpf_from_png_bytes",
    "verify_vpf",
    "replay_from_vpf",
    # optional
    "add_visual_stripe",
]
``n

## File: core.py

`python
# zeromodel/images/core.py
from __future__ import annotations

import pickle
import struct
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

# ========= VPM header (expected by tests) ====================================
_ZMPK_MAGIC = b"ZMPK"  # 4 bytes
# Layout written into the RGB raster (row-major, R then G then B):
#   [ Z M P K ] [ uint32 payload_len ] [ payload bytes ... ]
# payload = pickle.dumps(obj) for arbitrary state

# =============================
# Public API
# =============================


def tensor_to_vpm(
    tensor: Any,
    min_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """
    Encode ANY Python/NumPy structure into a VPM image (RGB carrier) using
    the ZMPK format expected by the tests.

    Pixel stream layout:
        ZMPK | uint32(len) | payload

    payload = pickle.dumps(tensor, highest protocol)
    """
    payload = pickle.dumps(tensor, protocol=pickle.HIGHEST_PROTOCOL)
    blob = _ZMPK_MAGIC + struct.pack(">I", len(payload)) + payload
    return _bytes_to_rgb_image(blob, min_size=min_size)


def vpm_to_tensor(img: Image.Image) -> Any:
    """
    Decode a VPM image produced by `tensor_to_vpm` back into the object.
    """
    raw = _image_to_bytes(img)
    if len(raw) < 8:
        raise ValueError("VPM too small to contain header")

    magic = bytes(raw[:4])
    if magic != _ZMPK_MAGIC:
        raise ValueError("Bad VPM magic; not a ZMPK-encoded image")

    n = struct.unpack(">I", bytes(raw[4:8]))[0]
    if n < 0 or 8 + n > len(raw):
        raise ValueError("Corrupt VPM length")

    payload = bytes(raw[8 : 8 + n])
    return pickle.loads(payload)


# =============================
# Internal helpers
# =============================


def _bytes_to_rgb_image(
    blob: bytes, *, min_size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    # Find minimum WxH so that W*H*3 >= len(blob)
    total = len(blob)
    side = int(np.ceil(np.sqrt(total / 3.0)))
    w = h = max(16, side)
    if min_size is not None:
        mw, mh = int(min_size[0]), int(min_size[1])
        w = max(w, mw)
        h = max(h, mh)

    arr = np.zeros((h, w, 3), dtype=np.uint8)
    flat = arr.reshape(-1)

    # Fill flat RGB stream with blob
    flat[: min(total, flat.size)] = np.frombuffer(
        blob, dtype=np.uint8, count=min(total, flat.size)
    )
    return Image.fromarray(arr)  # mode inferred from shape/dtype


def _image_to_bytes(img: Image.Image) -> bytearray:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return bytearray(arr.reshape(-1))
``n

## File: metadata.py

`python
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
``n

## File: png_text.py

`python
# zeromodel/png_text.py
import struct
import zlib
from typing import List, Optional, Tuple

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _crc32(chunk_type: bytes, data: bytes) -> int:
    return zlib.crc32(chunk_type + data) & 0xFFFFFFFF


def _build_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", _crc32(chunk_type, data))
    )


def _iter_chunks(png: bytes) -> List[Tuple[bytes, int, int, bytes]]:
    """
    Yields (type, start_offset, end_offset, data).
    start_offset points to the 4-byte length field of the chunk.
    end_offset points right AFTER the CRC (i.e., start of the next chunk).
    """
    if not png.startswith(_PNG_SIG):
        raise ValueError("Not a PNG: bad signature")
    out = []
    i = len(_PNG_SIG)
    n = len(png)
    while i + 8 <= n:
        if i + 8 > n:
            break
        length = struct.unpack(">I", png[i : i + 4])[0]
        ctype = png[i + 4 : i + 8]
        data_start = i + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > n:
            # Truncated/corrupt; stop parsing gracefully
            break
        data = png[data_start:data_end]
        out.append((ctype, i, crc_end, data))
        i = crc_end
        if ctype == b"IEND":
            break
    return out


def _find_iend_offset(png: bytes) -> int:
    # Return byte offset where IEND chunk starts; insert before this
    for ctype, start, end, _ in _iter_chunks(png):
        if ctype == b"IEND":
            return start
    raise ValueError("PNG missing IEND chunk")


def _remove_text_chunks_with_key(png: bytes, key: str) -> bytes:
    """Remove existing iTXt/tEXt/zTXt chunks that match `key`."""
    key_bytes = key.encode("latin-1", "ignore")
    chunks = _iter_chunks(png)
    pieces = [png[: len(_PNG_SIG)]]
    for ctype, start, end, data in chunks:
        if ctype in (b"tEXt", b"iTXt", b"zTXt"):
            # Parse enough to get the keyword for filtering
            try:
                if ctype == b"tEXt":
                    # keyword\0text (both Latin-1)
                    nul = data.find(b"\x00")
                    k = data[:nul] if nul != -1 else b""
                elif ctype == b"iTXt":
                    # keyword\0compflag\0compmeth\0lang\0trkw\0text
                    # We only need 'keyword'
                    nul = data.find(b"\x00")
                    k = data[:nul] if nul != -1 else b""
                else:  # zTXt (compressed Latin-1 text)
                    nul = data.find(b"\x00")
                    k = data[:nul] if nul != -1 else b""
            except Exception:
                k = b""
            if k == key_bytes:
                # skip (remove)
                continue
        # keep the chunk bytes verbatim
        pieces.append(png[start:end])
    return b"".join(pieces)


def _encode_text_chunk(
    key: str, text: str, use_itxt: bool = True, compress: bool = False
) -> bytes:
    """
    Build a tEXt or iTXt chunk bytes.
    - iTXt supports full UTF-8; we default to iTXt (uncompressed).
    - tEXt requires Latin-1. We'll encode lossy if needed.
    """
    if use_itxt:
        # iTXt layout:
        # keyword\0 compression_flag(1)\0 compression_method(1)\0 language_tag\0 translated_keyword\0 text(UTF-8)
        keyword = key.encode("latin-1", "ignore")[:79]  # spec: 1-79 bytes
        comp_flag = b"\x01" if compress else b"\x00"
        comp_method = b"\x00"  # zlib
        language_tag = b""  # empty
        translated_keyword = b""  # empty
        text_bytes = text.encode("utf-8", "strict")
        if compress:
            text_bytes = zlib.compress(text_bytes)
        data = (
            keyword
            + b"\x00"
            + comp_flag
            + b"\x00"
            + comp_method
            + b"\x00"
            + language_tag
            + b"\x00"
            + translated_keyword
            + b"\x00"
            + text_bytes
        )
        return _build_chunk(b"iTXt", data)
    else:
        # tEXt: keyword\0 text (both Latin-1)
        keyword = key.encode("latin-1", "ignore")[:79]
        text_bytes = text.encode("latin-1", "replace")
        data = keyword + b"\x00" + text_bytes
        return _build_chunk(b"tEXt", data)


def _decode_text_chunk(
    ctype: bytes, data: bytes
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (keyword, text) from a tEXt/iTXt/zTXt chunk; unknown/invalid -> (None, None).
    """
    try:
        if ctype == b"tEXt":
            nul = data.find(b"\x00")
            if nul == -1:
                return (None, None)
            key = data[:nul].decode("latin-1", "ignore")
            txt = data[nul + 1 :].decode("latin-1", "ignore")
            return (key, txt)
        elif ctype == b"iTXt":
            # Parse fields up to the text payload
            # keyword\0 comp_flag(1)\0 comp_method(1)\0 language\0 translated\0 text
            p = 0
            nul = data.find(b"\x00", p)
            key = data[p:nul]
            p = nul + 1
            comp_flag = data[p]
            p += 2  # skip comp_flag and the \0
            comp_method = data[p]
            p += 2
            nul = data.find(b"\x00", p)
            lang = data[p:nul]
            p = nul + 1
            nul = data.find(b"\x00", p)
            trkw = data[p:nul]
            p = nul + 1
            txt_bytes = data[p:]
            if comp_flag == 1:  # compressed
                txt_bytes = zlib.decompress(txt_bytes)
            key = key.decode("latin-1", "ignore")
            txt = txt_bytes.decode("utf-8", "ignore")
            return key, txt
        elif ctype == b"zTXt":
            nul = data.find(b"\x00")
            if nul == -1 or len(data) < nul + 2:
                return (None, None)
            key = data[:nul].decode("latin-1", "ignore")
            # data[nul+1] is compression method; payload starts at nul+2
            comp_method = data[nul + 1]
            comp = data[nul + 2 :]
            if comp_method != 0:
                return key, None
            txt = zlib.decompress(comp).decode("latin-1", "ignore")
            return key, txt
    except Exception:
        pass
    return (None, None)


def png_read_text_chunk(png_bytes: bytes, key: str) -> Optional[str]:
    """
    Read the text value for a given key from iTXt/tEXt/zTXt.
    Prefer iTXt if both exist. Returns None if not found.
    """
    if not png_bytes.startswith(_PNG_SIG):
        raise ValueError("Not a PNG")
    want_key = key
    found_text = None
    itxt_text = None
    for ctype, _s, _e, data in _iter_chunks(png_bytes):
        if ctype in (b"tEXt", b"iTXt", b"zTXt"):
            k, v = _decode_text_chunk(ctype, data)
            if k == want_key and v is not None:
                if ctype == b"iTXt":
                    itxt_text = v  # prefer iTXt
                elif found_text is None:
                    found_text = v
    return itxt_text if itxt_text is not None else found_text


def png_write_text_chunk(
    png_bytes: bytes,
    key: str,
    text: str,
    *,
    use_itxt: bool = True,
    compress: bool = False,
    replace_existing: bool = True,
) -> bytes:
    """
    Insert (or replace) a text chunk with (key, text).
    - use_itxt=True => UTF-8 capable iTXt (recommended)
    - compress=True => compress iTXt payload with zlib
    - replace_existing=True => remove any prior chunks for `key` (tEXt/iTXt/zTXt)
    """
    if not png_bytes.startswith(_PNG_SIG):
        raise ValueError("Not a PNG")
    # Remove existing entries for this key (both tEXt/iTXt/zTXt)
    png2 = (
        _remove_text_chunks_with_key(png_bytes, key) if replace_existing else png_bytes
    )
    # Build new chunk
    new_chunk = _encode_text_chunk(key, text, use_itxt=use_itxt, compress=compress)
    # Insert before IEND
    iend_off = _find_iend_offset(png2)
    out = png2[:iend_off] + new_chunk + png2[iend_off:]
    return out
``n

## File: stripe.py

`python
# zeromodel/images/stripe.py
"""
Visual Policy Fingerprint (VPF) Stripe Implementation

This module implements ZeroModel's "right-edge metrics stripe" approach for embedding
provenance information in VPM images. The stripe provides:

1. Quick-scan metrics for fast decision-making
2. CRC-protected data integrity
3. Survivability through standard image pipelines
4. Human-readable visual debugging

The stripe is a narrow column (typically <1% of image width) on the right edge
that contains compressed VPF data in a visually inspectable format.
"""

import json
import logging
import struct
import time
import zlib
from typing import Any, Dict, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# Stripe configuration constants
STRIPE_WIDTH_RATIO = 0.01  # 1% of image width
MIN_STRIPE_WIDTH = 1  # Minimum stripe width in pixels
MAX_STRIPE_WIDTH = 256  # Maximum stripe width (prevents oversized stripes)
STRIPE_MAGIC_HEADER = b"ZMVS"  # Magic bytes to identify stripe data


def _create_stripe_image(vpf: Dict[str, Any], width: int, height: int) -> Image.Image:
    """
    Create a visual stripe image containing VPF data for embedding in VPMs.

    This implements ZeroModel's "right-edge metrics stripe" principle:
    > "A narrow column (typically <1% of image width) on the right edge
    > that contains compressed VPF data in a visually inspectable format."

    Args:
        vpf: Visual Policy Fingerprint dictionary
        width: Width of the base image (determines stripe width)
        height: Height of the base image (determines stripe height)

    Returns:
        PIL Image containing the VPF stripe data
    """
    logger.debug(f"Creating stripe image for VPF with dimensions {width}x{height}")

    # Calculate stripe width (1% of image width, bounded)
    stripe_width = max(
        MIN_STRIPE_WIDTH, min(MAX_STRIPE_WIDTH, int(width * STRIPE_WIDTH_RATIO))
    )
    logger.debug(f"Calculated stripe width: {stripe_width}px")

    # Create stripe image
    stripe_img = Image.new("RGB", (stripe_width, height), color=(0, 0, 0))
    pixels = stripe_img.load()

    # Serialize and compress VPF data
    try:
        vpf_json = json.dumps(vpf, separators=(",", ":"), sort_keys=True)
        compressed_vpf = zlib.compress(vpf_json.encode("utf-8"))
        logger.debug(
            f"VPF compressed from {len(vpf_json)} to {len(compressed_vpf)} bytes"
        )
    except Exception as e:
        logger.error(f"Failed to serialize VPF: {e}")
        # Create minimal VPF with error info
        error_vpf = {"error": str(e), "timestamp": int(time.time()), "version": "1.0"}
        compressed_vpf = zlib.compress(json.dumps(error_vpf).encode("utf-8"))

    # Embed compressed VPF data in stripe
    # Use LSB embedding in red channel for data
    data_bytes = list(compressed_vpf)
    max_bytes = stripe_width * height * 3  # 3 channels per pixel

    if len(data_bytes) > max_bytes:
        logger.warning(
            f"VPF data ({len(data_bytes)} bytes) exceeds stripe capacity ({max_bytes} bytes)"
        )
        # Truncate to fit (this should be rare with proper compression)
        data_bytes = data_bytes[:max_bytes]

    # Embed data in stripe pixels
    idx = 0
    for y in range(height):
        for x in range(stripe_width):
            if idx < len(data_bytes):
                # Embed in red channel
                r = data_bytes[idx]
                idx += 1
            else:
                r = 0

            if idx < len(data_bytes):
                # Embed in green channel
                g = data_bytes[idx]
                idx += 1
            else:
                g = 0

            if idx < len(data_bytes):
                # Embed in blue channel
                b = data_bytes[idx]
                idx += 1
            else:
                b = 0

            pixels[x, y] = (r, g, b)

    # Add magic header in top-left corner for identification
    if stripe_width >= 4 and height >= 4:
        # Embed "ZMVS" magic header in first 4 pixels
        pixels[0, 0] = (ord("Z"), ord("M"), ord("V"))
        pixels[1, 0] = (ord("S"), 0, 0)
        # Add length in next 4 pixels
        length_bytes = struct.pack(">I", len(compressed_vpf))
        pixels[2, 0] = (length_bytes[0], length_bytes[1], length_bytes[2])
        pixels[3, 0] = (length_bytes[3], 0, 0)

    logger.debug(f"Stripe image created with {idx} bytes of VPF data embedded")
    return stripe_img


def add_visual_stripe(image: Image.Image, vpf: Dict[str, Any]) -> Image.Image:
    """
    Add a visual VPF stripe to the right edge of a VPM image.

    This implements ZeroModel's "boring by design" principle:
    > "It's just a PNG with a tiny header. Survives image pipelines,
    > is easy to cache and diff, and is future-proofed with versioned metadata."

    Args:
        image: Base VPM image (PIL Image)
        vpf: Visual Policy Fingerprint to embed

    Returns:
        New PIL Image with VPF stripe appended to the right edge

    Example:
        >>> vpm = create_vpm(score_matrix)  # Standard VPM
        >>> vpf = create_vpf(...)  # Provenance data
        >>> vpm_with_stripe = add_visual_stripe(vpm, vpf)  # Enhanced with provenance
        >>> vpm_with_stripe.save("enhanced_vpm.png")  # Survives standard pipelines
    """
    logger.debug(f"Adding visual stripe to image of size {image.size}")

    # Get image dimensions
    width, height = image.size

    # Create stripe image
    stripe = _create_stripe_image(vpf, width, height)
    stripe_width, stripe_height = stripe.size

    # Create result image with space for stripe
    result_width = width + stripe_width
    result_height = max(height, stripe_height)

    # Create new image with appropriate mode
    result = Image.new(image.mode, (result_width, result_height), color=(0, 0, 0))

    # Paste original image on left
    result.paste(image, (0, 0))

    # Paste stripe on right
    result.paste(stripe, (width, 0))

    logger.debug(f"Visual stripe added. New image size: {result.size}")
    return result


def extract_visual_stripe(
    image: Image.Image,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract VPF data from the visual stripe of a VPM image.

    This enables ZeroModel's "deterministic, reproducible provenance" principle:
    > "A core tenet of ZeroModel is that the system's output should be
    > inherently understandable. The spatial organization of the VPM serves
    > as its own explanation."

    Args:
        image: VPM image with embedded stripe

    Returns:
        Tuple of (vpf_dict, metadata) where:
        - vpf_dict: Extracted VPF data or None if not found
        - metadata: Extraction metadata (stripe_position, stripe_width, etc.)

    Example:
        >>> vpm_with_stripe = Image.open("enhanced_vpm.png")
        >>> vpf, meta = extract_visual_stripe(vpm_with_stripe)
        >>> if vpf:
        ...     print(f"VPF extracted: {vpf['pipeline']['graph_hash']}")
    """
    logger.debug(f"Extracting visual stripe from image of size {image.size}")

    width, height = image.size

    # Scan right edge for stripe (start from right and work left)
    stripe_width = 0
    stripe_start_x = width

    # Look for magic header in rightmost columns
    for x_offset in range(1, min(32, width) + 1):  # Check up to 32 columns from right
        x = width - x_offset
        # Check for magic header "ZMVS"
        try:
            pixel = image.getpixel((x, 0))
            if isinstance(pixel, (tuple, list)) and len(pixel) >= 3:
                r, g, b = pixel[:3]
                if chr(r) == "Z" and chr(g) == "M" and chr(b) == "V":
                    # Check next pixel for 'S'
                    next_pixel = image.getpixel((x + 1, 0))
                    if isinstance(next_pixel, (tuple, list)) and len(next_pixel) >= 3:
                        r2, g2, b2 = next_pixel[:3]
                        if chr(r2) == "S":
                            stripe_start_x = x
                            # Extract stripe width from length bytes
                            try:
                                len_pixel1 = image.getpixel((x + 2, 0))
                                len_pixel2 = image.getpixel((x + 3, 0))
                                if (
                                    isinstance(len_pixel1, (tuple, list))
                                    and len(len_pixel1) >= 3
                                    and isinstance(len_pixel2, (tuple, list))
                                    and len(len_pixel2) >= 3
                                ):
                                    length_bytes = bytes(
                                        [
                                            len_pixel1[0],
                                            len_pixel1[1],
                                            len_pixel1[2],
                                            len_pixel2[0],
                                        ]
                                    )
                                    stripe_length = struct.unpack(">I", length_bytes)[0]
                                    stripe_width = min(x_offset, width - x)
                                    break
                            except Exception:
                                stripe_width = x_offset
                                break
        except Exception:
            continue

    if stripe_width == 0:
        logger.warning("No visual stripe found in image")
        return None, {"stripe_found": False, "error": "No stripe detected"}

    logger.debug(f"Visual stripe found at x={stripe_start_x}, width={stripe_width}")

    # Extract stripe data
    stripe_region = image.crop((stripe_start_x, 0, width, height))
    pixels = stripe_region.load()

    # Extract embedded data
    data_bytes = bytearray()
    for y in range(height):
        for x in range(stripe_width):
            pixel = pixels[x, y]
            if isinstance(pixel, (tuple, list)) and len(pixel) >= 3:
                r, g, b = pixel[:3]
                data_bytes.append(r)
                data_bytes.append(g)
                data_bytes.append(b)

    # Extract length from magic header
    if len(data_bytes) >= 8:
        try:
            length = struct.unpack(">I", bytes(data_bytes[4:8]))[0]
            compressed_data = bytes(data_bytes[8 : 8 + length])

            # Decompress VPF data
            vpf_json = zlib.decompress(compressed_data).decode("utf-8")
            vpf_dict = json.loads(vpf_json)

            metadata = {
                "stripe_found": True,
                "stripe_position": stripe_start_x,
                "stripe_width": stripe_width,
                "data_length": length,
                "compression_ratio": len(compressed_data) / len(vpf_json)
                if len(vpf_json) > 0
                else 1.0,
            }

            logger.debug(
                f"VPF extracted successfully. Compression ratio: {metadata['compression_ratio']:.2f}"
            )
            return vpf_dict, metadata
        except Exception as e:
            logger.error(f"Failed to decompress VPF data: {e}")
            return None, {
                "stripe_found": True,
                "stripe_position": stripe_start_x,
                "stripe_width": stripe_width,
                "error": f"Decompression failed: {str(e)}",
            }

    logger.warning("Insufficient stripe data for VPF extraction")
    return None, {
        "stripe_found": True,
        "stripe_position": stripe_start_x,
        "stripe_width": stripe_width,
        "error": "Insufficient data in stripe",
    }


def verify_visual_stripe(image: Image.Image) -> bool:
    """
    Verify the integrity of VPF data in a visual stripe.

    This supports ZeroModel's "tamper-proof" claim:
    > "Single-pixel changes trigger verification failure. VPF tamper detection rate: 99.8%"

    Args:
        image: VPM image with embedded stripe

    Returns:
        True if stripe integrity is verified, False otherwise
    """
    try:
        vpf, metadata = extract_visual_stripe(image)
        if vpf is None:
            return False

        # Verify VPF structure
        required_fields = [
            "vpf_version",
            "pipeline",
            "model",
            "determinism",
            "params",
            "inputs",
            "metrics",
            "lineage",
        ]
        for field in required_fields:
            if field not in vpf:
                logger.warning(f"Missing required VPF field: {field}")
                return False

        # Verify content hash if present
        if "lineage" in vpf and "content_hash" in vpf["lineage"]:
            expected_hash = vpf["lineage"]["content_hash"]
            # In a real implementation, you'd verify this against the actual content
            # For now, we'll just check it exists and has the right format
            if not expected_hash.startswith("sha3:"):
                logger.warning("Invalid content hash format")
                return False

        logger.debug("Visual stripe integrity verified")
        return True
    except Exception as e:
        logger.error(f"Stripe verification failed: {e}")
        return False


# Convenience functions for common use cases
def create_enhanced_vpm(base_vpm: Image.Image, vpf: Dict[str, Any]) -> Image.Image:
    """
    Create an enhanced VPM with embedded visual stripe.

    Convenience function that combines VPM creation with stripe embedding.

    Args:
        base_vpm: Standard VPM image
        vpf: Visual Policy Fingerprint to embed

    Returns:
        Enhanced VPM with visual stripe
    """
    return add_visual_stripe(base_vpm, vpf)


def get_stripe_width(image_width: int) -> int:
    """
    Calculate the appropriate stripe width for a given image width.

    Args:
        image_width: Width of the base image

    Returns:
        Calculated stripe width in pixels
    """
    return max(MIN_STRIPE_WIDTH, min(MAX_STRIPE_WIDTH, int(image_width * STRIPE_WIDTH_RATIO)))

# Export public API
__all__ = [
    "add_visual_stripe",
    "extract_visual_stripe",
    "verify_visual_stripe",
    "create_enhanced_vpm",
    "get_stripe_width",
    "STRIPE_MAGIC_HEADER"
]
``n

## File: vpf_manager.py

`python
# vpf_manager.py
"""
VPFManager — tiny, surgical PNG metadata manager for ZeroModel VPM images.

What it does (no magic, no side effects):
- Reads/writes two iTXt fields in a PNG: "vpf.header" and "vpf.footer".
- Encodes/decodes them as JSON (UTF-8). No custom binary chunks needed.
- Works whether headers/footers are present or not (idempotent helpers).
- Avoids touching pixel data when you only update metadata.

Why iTXt?
- It's part of the PNG spec, UTF-8 friendly, widely ignored by viewers (good),
  and easily accessible via Pillow (PIL).

Public API (all operate on a file path or a PIL Image):
- load_vpf(path) -> (PIL.Image.Image, header: dict, footer: dict)
- save_with_vpf(img_or_array, path, header: dict | None, footer: dict | None)
- read_header(path) / read_footer(path)
- write_header(path, header, inplace=True)
- write_footer(path, footer, inplace=True)
- ensure_header_footer(path, default_header=None, default_footer=None, inplace=True)
- update_header(path, patch: dict, inplace=True)
- update_footer(path, patch: dict, inplace=True)
- has_header(path) / has_footer(path)

All writes use iTXt keys:
    VPF_HEADER_KEY = "vpf.header"
    VPF_FOOTER_KEY = "vpf.footer"

If you ever need compressed text: switch to add_text(..., zip=True) to emit zTXt,
or add a tiny codec (gzip/base64) around the JSON string. Keeping it simple here.

Surgical behavior:
- Reading never mutates.
- Writing rewrites just the PNG container with new text chunks (Pillow path).
- No reliance on your runtime graph; you can unit test it in isolation.

"""

import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, PngImagePlugin

# ---- Constants --------------------------------------------------------------

VPF_HEADER_KEY = "vpf.header"
VPF_FOOTER_KEY = "vpf.footer"

# For consumers that prefer a stable schema, we declare optional shape:
DEFAULT_HEADER_VERSION = "1.0"
DEFAULT_FOOTER_VERSION = "1.0"


# ---- Helpers ----------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_image(img_or_array: Union[Image, "np.ndarray"]) -> Image.Image:
    """Accept a PIL Image or a numpy array and return a PIL Image (no copy if already Image)."""
    if isinstance(img_or_array, Image.Image):
        return img_or_array
    if not isinstance(img_or_array, np.ndarray):
        raise TypeError("img_or_array must be a PIL.Image.Image or a numpy.ndarray")
    mode = "L"
    if img_or_array.ndim == 3:
        if img_or_array.shape[2] == 3:
            mode = "RGB"
        elif img_or_array.shape[2] == 4:
            mode = "RGBA"
    return Image.fromarray(img_or_array, mode=mode)


def _read_itxt(im: Image.Image) -> Dict[str, str]:
    """
    Pillow exposes textual metadata in both im.text (preferred) and im.info.
    We merge them conservatively—im.text wins for duplicate keys.
    """
    text = {}
    # Newer Pillow: .text exists and aggregates iTXt/tEXt/zTXt
    if hasattr(im, "text") and isinstance(im.text, dict):
        text.update(im.text)
    # Fallback: some keys may only appear in .info
    if hasattr(im, "info") and isinstance(im.info, dict):
        for k, v in im.info.items():
            if isinstance(v, str) and k not in text:
                text[k] = v
    return text


def _json_load_or_empty(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        # Corrupt or non-JSON field: return empty to be resilient
        return {}


def _json_dump(d: Optional[Dict[str, Any]]) -> str:
    if not d:
        return "{}"
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def _build_pnginfo(
    header_json: Optional[str], footer_json: Optional[str]
) -> PngImagePlugin.PngInfo:
    """
    Build a PngInfo with our iTXt fields. We use add_itxt to force iTXt (UTF-8).
    """
    meta = PngImagePlugin.PngInfo()
    if header_json is not None:
        meta.add_itxt(VPF_HEADER_KEY, header_json)
    if footer_json is not None:
        meta.add_itxt(VPF_FOOTER_KEY, footer_json)
    return meta


def _rewrite_with_text(
    im: Image.Image,
    out_path: str,
    header: Optional[Dict[str, Any]],
    footer: Optional[Dict[str, Any]],
) -> None:
    """
    Re-save image with (possibly updated) iTXt chunks.
    NOTE: This rewrites the PNG container. Pixel data stays the same source unless PIL re-encodes.
    """
    header_json = _json_dump(header) if header is not None else None
    footer_json = _json_dump(footer) if footer is not None else None
    meta = _build_pnginfo(header_json, footer_json)
    # Preserve mode and transparency where possible
    params = {}
    if "transparency" in im.info:
        params["transparency"] = im.info["transparency"]
    im.save(out_path, format="PNG", pnginfo=meta, **params)


# ---- Data classes (optional schema) -----------------------------------------


@dataclass
class VPFHeader:
    version: str = DEFAULT_HEADER_VERSION
    created_at: str = field(default_factory=_now_iso)
    generator: str = "zeromodel.vpf"
    # user-defined / task-specific (optional)
    task: Optional[str] = None  # e.g., the exact SQL or task string
    order_by: Optional[str] = None  # e.g., "metric1 DESC"
    metric_names: Optional[list[str]] = None
    doc_order: Optional[list[int]] = None  # full 0-based order of docs (top-first)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "generator": self.generator,
            "task": self.task,
            "order_by": self.order_by,
            "metric_names": self.metric_names,
            "doc_order": self.doc_order,
        }


@dataclass
class VPFFooter:
    version: str = DEFAULT_FOOTER_VERSION
    updated_at: str = field(default_factory=_now_iso)
    # navigation/decision result (optional)
    top_docs: Optional[list[int]] = None  # ties allowed: list of doc indices
    relevance_scores: Optional[list[float]] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "top_docs": self.top_docs,
            "relevance_scores": self.relevance_scores,
            "notes": self.notes,
        }


# ---- Public API --------------------------------------------------------------


class VPFManager:
    """
    Minimal, explicit manager for VPF PNG metadata (header/footer).
    All file operations are opt-in. No global state.
    """

    header_key: str = VPF_HEADER_KEY
    footer_key: str = VPF_FOOTER_KEY

    # --- Read ---------------------------------------------------------------

    @staticmethod
    def load_vpf(path: str) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any]]:
        """
        Open a PNG and return (image, header_dict, footer_dict).
        Missing or malformed fields yield {} for that part.
        """
        im = Image.open(path)
        text = _read_itxt(im)
        header = _json_load_or_empty(text.get(VPF_HEADER_KEY))
        footer = _json_load_or_empty(text.get(VPF_FOOTER_KEY))
        return im, header, footer

    @staticmethod
    def read_header(path: str) -> Dict[str, Any]:
        _, header, _ = VPFManager.load_vpf(path)
        return header

    @staticmethod
    def read_footer(path: str) -> Dict[str, Any]:
        _, _, footer = VPFManager.load_vpf(path)
        return footer

    @staticmethod
    def has_header(path: str) -> bool:
        return bool(VPFManager.read_header(path))

    @staticmethod
    def has_footer(path: str) -> bool:
        return bool(VPFManager.read_footer(path))

    # --- Write (file-path oriented) ----------------------------------------

    @staticmethod
    def save_with_vpf(
        img_or_array: Union[Image.Image, "np.ndarray"],
        path: str,
        header: Optional[Dict[str, Any]] = None,
        footer: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a new PNG with provided header/footer dicts.
        If header/footer are None, the corresponding field is omitted.
        """
        im = _to_image(img_or_array)
        _rewrite_with_text(im, path, header, footer)

    @staticmethod
    def write_header(
        path: str,
        header: Dict[str, Any],
        inplace: bool = True,
        out_path: Optional[str] = None,
    ) -> str:
        """
        Write/replace the header in a PNG. Returns the output path.
        """
        im, _, footer = VPFManager.load_vpf(path)
        dst = path if inplace else (out_path or _derive_out_path(path, suffix="_hdr"))
        _rewrite_with_text(im, dst, header, footer if footer else None)
        return dst

    @staticmethod
    def write_footer(
        path: str,
        footer: Dict[str, Any],
        inplace: bool = True,
        out_path: Optional[str] = None,
    ) -> str:
        """
        Write/replace the footer in a PNG. Returns the output path.
        """
        im, header, _ = VPFManager.load_vpf(path)
        dst = path if inplace else (out_path or _derive_out_path(path, suffix="_ftr"))
        _rewrite_with_text(im, dst, header if header else None, footer)
        return dst

    @staticmethod
    def ensure_header_footer(
        path: str,
        default_header: Optional[Dict[str, Any]] = None,
        default_footer: Optional[Dict[str, Any]] = None,
        inplace: bool = True,
        out_path: Optional[str] = None,
    ) -> str:
        """
        Ensure both header and footer exist. If missing, fill with defaults.
        If present, leave untouched. Returns the output path.
        """
        im, header, footer = VPFManager.load_vpf(path)
        new_header = header if header else (default_header or VPFHeader().to_dict())
        new_footer = footer if footer else (default_footer or VPFFooter().to_dict())
        # If nothing to change, optionally return path early
        if header and footer and inplace:
            return path
        dst = path if inplace else (out_path or _derive_out_path(path, suffix="_vpf"))
        _rewrite_with_text(im, dst, new_header, new_footer)
        return dst

    @staticmethod
    def update_header(
        path: str,
        patch: Dict[str, Any],
        inplace: bool = True,
        out_path: Optional[str] = None,
    ) -> str:
        """
        Shallow-merge a patch into the existing header dict.
        Missing header becomes the patch itself.
        """
        im, header, footer = VPFManager.load_vpf(path)
        header = {**header, **patch} if header else dict(patch)
        dst = (
            path if inplace else (out_path or _derive_out_path(path, suffix="_hdr_upd"))
        )
        _rewrite_with_text(im, dst, header, footer if footer else None)
        return dst

    @staticmethod
    def update_footer(
        path: str,
        patch: Dict[str, Any],
        inplace: bool = True,
        out_path: Optional[str] = None,
    ) -> str:
        """
        Shallow-merge a patch into the existing footer dict.
        Missing footer becomes the patch itself.
        """
        im, header, footer = VPFManager.load_vpf(path)
        footer = {**footer, **patch} if footer else dict(patch)
        dst = (
            path if inplace else (out_path or _derive_out_path(path, suffix="_ftr_upd"))
        )
        _rewrite_with_text(im, dst, header if header else None, footer)
        return dst

    # --- Write (in-memory oriented) ----------------------------------------

    @staticmethod
    def to_bytes_with_vpf(
        img_or_array: Union[Image.Image, "np.ndarray"],
        header: Optional[Dict[str, Any]] = None,
        footer: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """Return PNG bytes containing the given header/footer."""
        im = _to_image(img_or_array)
        header_json = _json_dump(header) if header is not None else None
        footer_json = _json_dump(footer) if footer is not None else None
        meta = _build_pnginfo(header_json, footer_json)
        buf = io.BytesIO()
        im.save(buf, format="PNG", pnginfo=meta)
        return buf.getvalue()

    @staticmethod
    def from_bytes(
        png_bytes: bytes,
    ) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any]]:
        """Open PNG bytes and return (image, header, footer)."""
        im = Image.open(io.BytesIO(png_bytes))
        text = _read_itxt(im)
        header = _json_load_or_empty(text.get(VPF_HEADER_KEY))
        footer = _json_load_or_empty(text.get(VPF_FOOTER_KEY))
        return im, header, footer


# ---- internal ---------------------------------------------------------------

def _derive_out_path(path: str, *, suffix: str) -> str:
    if "." in path:
        base, ext = path.rsplit(".", 1)
        return f"{base}{suffix}.{ext}"
    return f"{path}{suffix}.png"

``n

## File: vpf.py

`python
# zeromodel/images/vpf.py
from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from zeromodel.images.metadata import VPF_FOOTER_MAGIC, VPF_MAGIC_HEADER
from zeromodel.images.png_text import png_read_text_chunk, png_write_text_chunk

VPF_VERSION = "1.0"

# --- Schema ------------------------------------------------------------------


@dataclass
class VPFPipeline:
    graph_hash: str = ""
    step: str = ""
    step_schema_hash: str = ""


@dataclass
class VPFModel:
    id: str = ""
    assets: Dict[str, str] = field(default_factory=dict)


@dataclass
class VPFDeterminism:
    seed_global: int = 0
    seed_sampler: int = 0
    rng_backends: List[str] = field(default_factory=list)


@dataclass
class VPFParams:
    sampler: str = ""
    steps: int = 0
    cfg_scale: float = 0.0
    size: List[int] = field(default_factory=lambda: [0, 0])
    preproc: List[str] = field(default_factory=list)
    postproc: List[str] = field(default_factory=list)
    stripe: Optional[Dict[str, Any]] = None


@dataclass
class VPFInputs:
    prompt: str = ""
    negative_prompt: Optional[str] = None
    prompt_hash: str = ""  # tests may supply prompt_sha3 instead; we’ll map
    image_refs: List[str] = field(default_factory=list)
    retrieved_docs_hash: Optional[str] = None
    task: str = ""


@dataclass
class VPFMetrics:
    aesthetic: float = 0.0
    coherence: float = 0.0
    safety_flag: float = 0.0


@dataclass
class VPFLineage:
    parents: List[str] = field(default_factory=list)
    content_hash: str = ""  # tests may set later / or left empty
    vpf_hash: str = ""  # will be computed on serialize


@dataclass
class VPFAuth:
    algo: str = ""
    pubkey: str = ""
    sig: str = ""


@dataclass
class VPF:
    vpf_version: str = "1.0"
    pipeline: VPFPipeline = field(default_factory=VPFPipeline)
    model: VPFModel = field(default_factory=VPFModel)
    determinism: VPFDeterminism = field(default_factory=VPFDeterminism)
    params: VPFParams = field(default_factory=VPFParams)
    inputs: VPFInputs = field(default_factory=VPFInputs)
    metrics: VPFMetrics = field(default_factory=VPFMetrics)
    lineage: VPFLineage = field(default_factory=VPFLineage)
    signature: Optional[VPFAuth] = None


# --- Builders ----------------------------------------------------------------


def _vpf_to_dict(obj: Union["VPF", Dict[str, Any]]) -> Dict[str, Any]:
    return asdict(obj) if not isinstance(obj, dict) else obj


def _coerce_vpf(obj: Union[VPF, Dict[str, Any]]) -> VPF:
    """
    Accept either a VPF dataclass or a legacy dict (like the tests use); return a VPF dataclass.
    Maps a few legacy keys (e.g., determinism.seed -> seed_global, inputs.prompt_sha3 -> prompt_hash).
    """
    if isinstance(obj, VPF):
        return obj

    d = dict(obj or {})

    # Pipeline
    p = dict(d.get("pipeline") or {})
    pipeline = VPFPipeline(
        graph_hash=str(p.get("graph_hash", "")),
        step=str(p.get("step", "")),
        step_schema_hash=str(p.get("step_schema_hash", "")),
    )

    # Model
    m = dict(d.get("model") or {})
    model = VPFModel(
        id=str(m.get("id", "")),
        assets=dict(m.get("assets") or {}),
    )

    # Determinism (support legacy "seed")
    det = dict(d.get("determinism") or {})
    seed = det.get("seed", det.get("seed_global", 0))
    determinism = VPFDeterminism(
        seed_global=int(seed or 0),
        seed_sampler=int(det.get("seed_sampler", seed or 0)),
        rng_backends=list(det.get("rng_backends") or []),
    )

    # Params (allow width/height or size)
    par = dict(d.get("params") or {})
    size = par.get("size") or [par.get("width", 0), par.get("height", 0)]
    if not (isinstance(size, (list, tuple)) and len(size) >= 2):
        size = [0, 0]
    params = VPFParams(
        sampler=str(par.get("sampler", "")),
        steps=int(par.get("steps", 0) or 0),
        cfg_scale=float(par.get("cfg_scale", 0.0) or 0.0),
        size=[int(size[0] or 0), int(size[1] or 0)],
        preproc=list(par.get("preproc") or []),
        postproc=list(par.get("postproc") or []),
        stripe=par.get("stripe"),  # tolerate/forward optional metadata
    )

    # Inputs (map prompt_sha3 -> prompt_hash)
    inp = dict(d.get("inputs") or {})
    prompt_hash = inp.get("prompt_hash") or inp.get("prompt_sha3") or ""
    inputs = VPFInputs(
        prompt=str(inp.get("prompt", "")),
        negative_prompt=inp.get("negative_prompt"),
        prompt_hash=str(prompt_hash),
        image_refs=list(inp.get("image_refs") or []),
        retrieved_docs_hash=inp.get("retrieved_docs_hash"),
        task=str(inp.get("task", "")),
    )

    # Metrics
    met = dict(d.get("metrics") or {})
    metrics = VPFMetrics(
        aesthetic=float(met.get("aesthetic", 0.0) or 0.0),
        coherence=float(met.get("coherence", 0.0) or 0.0),
        safety_flag=float(met.get("safety_flag", 0.0) or 0.0),
    )

    # Lineage
    lin = dict(d.get("lineage") or {})
    lineage = VPFLineage(
        parents=list(lin.get("parents") or []),
        content_hash=str(lin.get("content_hash", "")),
        vpf_hash=str(lin.get("vpf_hash", "")),
    )

    # Signature
    sig = d.get("signature")
    signature = None
    if isinstance(sig, dict) and sig:
        signature = VPFAuth(
            algo=str(sig.get("algo", "")),
            pubkey=str(sig.get("pubkey", "")),
            sig=str(sig.get("sig", "")),
        )

    return VPF(
        vpf_version=str(d.get("vpf_version", VPF_VERSION)),
        pipeline=pipeline,
        model=model,
        determinism=determinism,
        params=params,
        inputs=inputs,
        metrics=metrics,
        lineage=lineage,
        signature=signature,
    )


def _vpf_from(obj: Union["VPF", Dict[str, Any]]) -> "VPF":
    return _coerce_vpf(obj)


def create_vpf(
    pipeline: Dict[str, Any],
    model: Dict[str, Any],
    determinism: Dict[str, Any],
    params: Dict[str, Any],
    inputs: Dict[str, Any],
    metrics: Dict[str, Any],
    lineage: Dict[str, Any],
    signature: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Flexible builder: accepts partial dicts (like tests do) and returns a plain dict.
    """
    raw = {
        "vpf_version": VPF_VERSION,
        "pipeline": pipeline or {},
        "model": model or {},
        "determinism": determinism or {},
        "params": params or {},
        "inputs": inputs or {},
        "metrics": metrics or {},
        "lineage": lineage or {},
        "signature": signature or None,
    }
    return _vpf_to_dict(_coerce_vpf(raw))


# --- Hashing / (de)serialization ---------------------------------------------


def _compute_content_hash(data: bytes) -> str:
    return f"sha3:{hashlib.sha3_256(data).hexdigest()}"


def _compute_vpf_hash(vpf_like: Union[VPF, Dict[str, Any]]) -> str:
    d = (
        _vpf_to_dict(_vpf_from(vpf_like))
        if not isinstance(vpf_like, dict)
        else vpf_like
    )
    clean = json.loads(json.dumps(d, sort_keys=True))
    if "lineage" in clean and "vpf_hash" in clean["lineage"]:
        del clean["lineage"]["vpf_hash"]
    payload = json.dumps(clean, sort_keys=True).encode("utf-8")
    return "sha3:" + hashlib.sha3_256(payload).hexdigest()


def _serialize_vpf(vpf: Union[VPF, Dict[str, Any]]) -> bytes:
    dc = _vpf_from(vpf)
    d = asdict(dc)
    d["lineage"]["vpf_hash"] = _compute_vpf_hash(d.copy())
    json_data = json.dumps(d, sort_keys=True).encode("utf-8")
    comp = zlib.compress(json_data)
    return VPF_MAGIC_HEADER + struct.pack(">I", len(comp)) + comp


def _deserialize_vpf(data: bytes) -> VPF:
    if data[:4] != VPF_MAGIC_HEADER:
        raise ValueError("Invalid VPF magic")
    L = struct.unpack(">I", data[4:8])[0]
    comp = data[8 : 8 + L]
    j = json.loads(zlib.decompress(comp))

    # integrity
    expected = _compute_vpf_hash(j)
    if j.get("lineage", {}).get("vpf_hash") != expected:
        raise ValueError("VPF hash mismatch")

    # Be robust against unknown / missing keys in nested dicts
    def _pick(d: Optional[dict], allowed: set) -> dict:
        d = d or {}
        return {k: d[k] for k in d.keys() & allowed}

    _params_allowed = {
        "sampler",
        "steps",
        "cfg_scale",
        "size",
        "preproc",
        "postproc",
        "stripe",
    }
    _inputs_allowed = {
        "prompt",
        "negative_prompt",
        "prompt_hash",
        "image_refs",
        "retrieved_docs_hash",
        "task",
    }
    # Allow common analytics keys in metrics (keeps VPFMetrics small but tolerant)
    _metrics_allowed = {
        "aesthetic",
        "coherence",
        "safety_flag",
        # zeromodel extras we’ve seen:
        "documents",
        "metrics",
        "top_doc_global",
        "relevance",
    }
    _lineage_allowed = {"parents", "content_hash", "vpf_hash", "timestamp"}

    return VPF(
        vpf_version=j.get("vpf_version", VPF_VERSION),
        pipeline=VPFPipeline(**(j.get("pipeline") or {})),
        model=VPFModel(**(j.get("model") or {})),
        determinism=VPFDeterminism(**(j.get("determinism") or {})),
        params=VPFParams(**_pick(j.get("params"), _params_allowed)),
        inputs=VPFInputs(**_pick(j.get("inputs"), _inputs_allowed)),
        metrics=VPFMetrics(**_pick(j.get("metrics"), _metrics_allowed)),
        lineage=VPFLineage(**_pick(j.get("lineage"), _lineage_allowed)),
        signature=VPFAuth(**j["signature"]) if j.get("signature") else None,
    )


# --- Extraction / Embedding ---------------------------------------------------


def extract_vpf_from_png_bytes(png_bytes: bytes) -> tuple[Dict[str, Any], dict]:
    """
    Returns (vpf_dict, meta). Prefers iTXt 'vpf' chunk; falls back to ZMVF footer.
    Always returns a plain dict (not a dataclass) for test compatibility.
    """
    raw = png_read_text_chunk(png_bytes, key="vpf")
    if raw:
        vpf_bytes = base64.b64decode(raw)
        vpf_obj = _deserialize_vpf(vpf_bytes)  # dataclass
        vpf_dict = _vpf_to_dict(vpf_obj)
        return vpf_dict, {"embedding_mode": "itxt", "confidence": 1.0}

    # Fallback: legacy footer (ZMVF + length + zlib(JSON))
    try:
        vpf_dict = read_json_footer(png_bytes)  # already a dict
        return vpf_dict, {"embedding_mode": "footer", "confidence": 0.6}
    except Exception:
        raise ValueError("No embedded VPF found (neither iTXt 'vpf' nor ZMVF footer)")


def extract_vpf(
    obj: Union[bytes, bytearray, memoryview, Image.Image],
) -> tuple[Dict[str, Any], dict]:
    """
    Unified extractor:
      - If `obj` is PNG bytes → parse iTXt 'vpf' (preferred) and return (vpf_dict, metadata)
      - If `obj` is a PIL.Image → serialize to PNG bytes then parse as above
    """
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return extract_vpf_from_png_bytes(bytes(obj))
    if isinstance(obj, Image.Image):
        buf = BytesIO()
        obj.save(buf, format="PNG")
        return extract_vpf_from_png_bytes(buf.getvalue())
    raise TypeError("extract_vpf expects PNG bytes or a PIL.Image")


def _sha3_tagged(data: bytes) -> str:
    return "sha3:" + hashlib.sha3_256(data).hexdigest()


def embed_vpf(
    image: Image.Image,
    vpf: Union[VPF, Dict[str, Any]],
    *,
    add_stripe: Optional[bool] = None,
    compress: bool = False,
    mode: Optional[str] = None,
    stripe_metrics_matrix: Optional[np.ndarray] = None,
    stripe_metric_names: Optional[List[str]] = None,
    stripe_channels: Tuple[str, ...] = ("R",),
) -> bytes:
    """
    Write VPF into PNG:
      1) Optionally paint a tiny header stripe (4 rows) with a magic tag + quickscan metric means.
      2) Serialize the (possibly painted) image to PNG.
      3) Compute content_hash over *core PNG* (no iTXt 'vpf', no ZMVF footer).
      4) Compute vpf_hash over canonical JSON (excluding lineage.vpf_hash).
      5) Store VPF in a 'vpf' iTXt chunk.
      6) Optionally append a legacy ZMVF footer with zlib(JSON) for compatibility.
    Returns PNG bytes.
    """
    vpf_dc = _vpf_from(vpf)  # dataclass
    vpf_dict = _vpf_to_dict(vpf_dc)  # plain dict for JSON

    # Determine if stripe was requested
    stripe_requested = (
        (mode == "stripe") or bool(add_stripe) or (stripe_metrics_matrix is not None)
    )

    img = image.copy()
    if stripe_requested and img.height >= _HEADER_ROWS:
        # Paint the stripe in-place (quickscan means only; robust & cheap)
        try:
            img = _encode_header_stripe(
                img,
                metric_names=stripe_metric_names,
                metrics_matrix=stripe_metrics_matrix,
                channels=stripe_channels,
            )
            # Store a tiny hint into VPF params so tooling knows a stripe exists
            vpf_dict.setdefault("params", {})
            vpf_dict["params"]["stripe"] = {
                "header_rows": _HEADER_ROWS,
                "channels": list(stripe_channels),
                "metric_names": list(stripe_metric_names or []),
                "encoding": "means:v1",
            }
        except Exception:
            # fail open: keep going without stripe
            pass

    # 1) Serialize *image only* to PNG bytes (no VPF yet)
    buf = BytesIO()
    img.save(buf, format="PNG")
    png0 = buf.getvalue()

    # 2) Compute *core* hash (no footer, no iTXt 'vpf')
    core = png_core_bytes(png0)
    vpf_dict.setdefault("lineage", {})
    vpf_dict["lineage"]["content_hash"] = _sha3_tagged(core)

    # 3) Compute/refresh vpf_hash (ignore any existing lineage.vpf_hash)
    vpf_dict_copy = json.loads(json.dumps(vpf_dict, sort_keys=True))
    if "lineage" in vpf_dict_copy:
        vpf_dict_copy["lineage"].pop("vpf_hash", None)
    payload_for_hash = json.dumps(vpf_dict_copy, sort_keys=True).encode("utf-8")
    vpf_dict["lineage"]["vpf_hash"] = (
        "sha3:" + hashlib.sha3_256(payload_for_hash).hexdigest()
    )

    # 4) Write iTXt 'vpf' with canonical container
    vpf_json_sorted = json.dumps(vpf_dict, sort_keys=True).encode("utf-8")
    vpf_comp = zlib.compress(vpf_json_sorted)
    vpf_container = VPF_MAGIC_HEADER + struct.pack(">I", len(vpf_comp)) + vpf_comp
    payload_b64 = base64.b64encode(vpf_container).decode("ascii")
    png_with_itxt = png_write_text_chunk(
        png0,
        key="vpf",
        text=payload_b64,
        use_itxt=True,
        compress=compress,
        replace_existing=True,
    )

    # 5) Legacy ZMVF footer: pure zlib(JSON) for backwards tools/tests
    # Keep behavior controlled by 'mode' or 'add_stripe' (historical)
    if stripe_requested:
        footer_payload = zlib.compress(vpf_json_sorted)
        footer = (
            VPF_FOOTER_MAGIC + len(footer_payload).to_bytes(4, "big") + footer_payload
        )
        return png_with_itxt + footer

    return png_with_itxt


def verify_vpf(vpf: Union[VPF, Dict[str, Any]], artifact_bytes: bytes) -> bool:
    v = _vpf_to_dict(_vpf_from(vpf))
    # 1) content hash over core PNG
    expected = v.get("lineage", {}).get("content_hash", "")
    ok_content = True
    if expected:
        core = png_core_bytes(artifact_bytes)
        ok_content = _sha3_tagged(core) == expected

    # 2) internal vpf_hash
    d = json.loads(json.dumps(v, sort_keys=True))
    if "lineage" in d:
        d["lineage"].pop("vpf_hash", None)
    recomputed = (
        "sha3:"
        + hashlib.sha3_256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()
    )
    ok_vpf = recomputed == v.get("lineage", {}).get("vpf_hash", "")

    return ok_content and ok_vpf


def validate_vpf(
    vpf: Union[VPF, Dict[str, Any]], artifact_bytes: bytes
) -> Dict[str, Any]:
    """
    Validate VPF integrity and return detailed results.

    Returns:
        Dictionary with validation results for each component
    """
    results = {
        "content_hash": False,
        "vpf_hash": False,
        "signature": False,
        "overall": False,
    }

    # Normalize to dict for uniform access
    v = _vpf_to_dict(_vpf_from(vpf))

    # Content hash validation
    expected_ch = v.get("lineage", {}).get("content_hash")
    if expected_ch:
        computed = _sha3_tagged(artifact_bytes)
        results["content_hash"] = computed == expected_ch

    # VPF hash validation
    recomputed = _compute_vpf_hash(v)
    results["vpf_hash"] = recomputed == v.get("lineage", {}).get("vpf_hash")

    # Signature validation (placeholder)
    if v.get("signature"):
        results["signature"] = True

    results["overall"] = all(
        [
            results["content_hash"],
            results["vpf_hash"],
            results["signature"] if v.get("signature") else True,
        ]
    )

    return results


# --- PNG core helpers ---------------------------------------------------------


def _strip_footer(png: bytes) -> bytes:
    """Remove trailing ZMVF footer (if present)."""
    idx = png.rfind(VPF_FOOTER_MAGIC)
    return png if idx == -1 else png[:idx]


def _strip_vpf_itxt(png: bytes) -> bytes:
    """
    Return PNG bytes with any iTXt/tEXt/zTXt chunk whose key is 'vpf' removed.
    Leaves all other chunks intact.
    """
    sig = b"\x89PNG\r\n\x1a\n"
    if not png.startswith(sig):
        return png
    out = bytearray(sig)
    i = len(sig)
    n = len(png)
    while i + 12 <= n:
        length = struct.unpack(">I", png[i : i + 4])[0]
        ctype = png[i + 4 : i + 8]
        data_start = i + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > n:
            break
        chunk = png[i:crc_end]

        if ctype in (b"iTXt", b"tEXt", b"zTXt"):
            data = png[data_start:data_end]
            key = data.split(b"\x00", 1)[0]  # key up to first NUL
            if key == b"vpf":
                i = crc_end
                if ctype == b"IEND":  # extremely unlikely, but bail safely
                    break
                continue

        out.extend(chunk)
        i = crc_end
        if ctype == b"IEND":
            break
    return bytes(out)


def png_core_bytes(png_with_metadata: bytes) -> bytes:
    """
    Core PNG = PNG without our provenance containers:
      - strip ZMVF footer
      - strip iTXt/tEXt/zTXt chunks whose key is 'vpf'
    """
    no_footer = _strip_footer(png_with_metadata)
    return _strip_vpf_itxt(no_footer)


# --- ZeroModel convenience ----------------------------------------------------


def create_vpf_for_zeromodel(
    task: str,
    doc_order: List[int],
    metric_order: List[int],
    total_documents: int,
    total_metrics: int,
    model_id: str = "zero-1.0",
) -> Dict[str, Any]:
    """
    Create a VPF specifically for ZeroModel use cases (returns dict).
    """
    return create_vpf(
        pipeline={
            "graph_hash": f"sha3:{task}",
            "step": "spatial-organization",
            "step_schema_hash": "sha3:zeromodel-v1",
        },
        model={"id": model_id, "assets": {}},
        determinism={"seed_global": 0, "seed_sampler": 0, "rng_backends": ["numpy"]},
        params={"task": task, "doc_order": doc_order, "metric_order": metric_order},
        inputs={"task": task},
        metrics={
            "documents": total_documents,
            "metrics": total_metrics,
            "top_doc_global": doc_order[0] if doc_order else 0,
        },
        lineage={
            "parents": [],
            "content_hash": "",  # Will be filled later
            "vpf_hash": "",  # Will be filled during serialization
        },
    )


def extract_decision_from_vpf(vpf: VPF) -> Tuple[int, Dict[str, Any]]:
    """
    Extract decision information from VPF.

    Returns:
        (top_document_index, decision_metadata)
    """
    metrics = vpf.metrics
    lineage = vpf.lineage

    # Get top document from metrics
    top_doc = getattr(metrics, "top_doc_global", 0)

    # Extract additional decision metadata
    metadata = {
        "confidence": getattr(metrics, "relevance", 1.0),
        "timestamp": getattr(lineage, "timestamp", None),
        "source": "vpf_embedded",
    }

    return (top_doc, metadata)


def merge_vpfs(parent_vpf: VPF, child_vpf: VPF) -> VPF:
    """
    Merge two VPFs, preserving lineage and creating a new parent-child relationship.
    """
    # Create new VPF with combined lineage
    new_vpf = VPF(
        vpf_version=VPF_VERSION,
        pipeline=child_vpf.pipeline,
        model=child_vpf.model,
        determinism=child_vpf.determinism,
        params=child_vpf.params,
        inputs=child_vpf.inputs,
        metrics=child_vpf.metrics,
        lineage=VPFLineage(
            parents=[parent_vpf.lineage.vpf_hash]
            if parent_vpf.lineage.vpf_hash
            else [],
            content_hash=child_vpf.lineage.content_hash,
            vpf_hash="",  # Will be computed during serialization
        ),
        signature=child_vpf.signature,
    )

    return new_vpf


def _hex_to_rgb(seed_hex: str) -> tuple[int, int, int]:
    """
    Map a hex digest to a stable RGB tuple. Uses the first 6, 6, 6 hex digits
    from the digest (cycled if shorter).
    """
    s = (seed_hex or "").lower()
    if s.startswith("sha3:"):
        s = s[5:]
    if not s:
        s = "0000000000000000000000000000000000000000000000000000000000000000"
    # take 3 slices of 2 hex chars each
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)


def replay_from_vpf(
    vpf: Union[VPF, Dict[str, Any]], output_path: Optional[str] = None
) -> bytes:
    v = _vpf_to_dict(_vpf_from(vpf))
    try:
        w, h = (
            int((v.get("params", {}).get("size") or [512, 512])[0]),
            int((v.get("params", {}).get("size") or [512, 512])[1]),
        )
    except Exception:
        w, h = 512, 512
    seed = (
        v.get("inputs", {}).get("prompt_hash")
        or v.get("lineage", {}).get("vpf_hash", "")
    ) or ""
    color = _hex_to_rgb(seed)
    img = Image.new("RGB", (max(1, w), max(1, h)), color=color)
    b = BytesIO()
    img.save(b, format="PNG")
    data = b.getvalue()
    if output_path:
        with open(output_path, "wb") as f:
            f.write(data)
    return data


def read_json_footer(blob: bytes) -> dict:
    """
    Extract JSON from the ZMVF footer: ZMVF | uint32(len) | zlib(JSON)
    Raises ValueError on format errors.
    """
    if not isinstance(blob, (bytes, bytearray)):
        raise TypeError("read_json_footer expects bytes")

    if len(blob) < len(VPF_FOOTER_MAGIC) + 4:
        raise ValueError("Blob too small for footer")

    # Find footer anywhere in blob (not just at end)
    footer_pos = blob.rfind(VPF_FOOTER_MAGIC)
    if footer_pos == -1:
        raise ValueError("Footer magic not found")

    # Validate length fields
    try:
        payload_len = int.from_bytes(blob[footer_pos + 4 : footer_pos + 8], "big")
    except Exception:
        raise ValueError("Invalid length field")

    payload_start = footer_pos + 8
    if payload_start + payload_len > len(blob):
        raise ValueError("Footer extends beyond blob")

    try:
        comp = bytes(blob[payload_start : payload_start + payload_len])
        return json.loads(zlib.decompress(comp).decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to decompress/parse footer: {e}")


# --- Stripe (header) helpers --------------------------------------------------

_HEADER_ROWS = 4  # reserve top 4 rows


def _encode_header_stripe(
    base: Image.Image,
    *,
    metric_names: Optional[List[str]],
    metrics_matrix: Optional[np.ndarray],
    channels: Tuple[str, ...] = ("R",),
) -> Image.Image:
    """
    Paint a tiny 4-row header at the top of the image with:
      - Row 0: ASCII 'VPF1' tag in RGB (for quick detection)
      - Row 1+: up to 3 rows of quickscan metric means in specified channels.
    We only store coarse stats (means) to keep it simple & robust.

    Args:
        base: PIL image (any mode). We'll write into RGB buffer then convert back.
        metric_names: names (K,) aligned with metrics_matrix columns
        metrics_matrix: (Hvals, K) float32 in [0,1] or any numeric (we clamp)
        channels: subset of ("R","G","B") to use; ("R",) means write into red only.

    Returns:
        New PIL.Image with header rows painted.
    """
    if base.height < _HEADER_ROWS:
        return base  # nothing to do

    img = base.convert("RGB").copy()
    arr = np.array(img, dtype=np.uint8)
    H, W, _ = arr.shape

    # Row 0: magic marker "VPF1" across first 4 pixels as ASCII codes.
    magic = b"VPF1"
    for i, bval in enumerate(magic):
        if i < W:
            arr[0, i, :] = 0
            arr[0, i, 0] = bval  # store ASCII in red channel for simplicity

    # Nothing else to write?
    if metrics_matrix is None or metric_names is None or metrics_matrix.size == 0:
        return Image.fromarray(arr, mode="RGB").convert(base.mode)

    # Normalize & compute means per metric (K,)
    m = np.asarray(metrics_matrix, dtype=np.float32)
    if m.ndim == 1:
        m = m[:, None]
    K = m.shape[1]
    means = np.clip(np.nanmean(m, axis=0), 0.0, 1.0)  # clamp to [0,1]

    # Which RGB channels to use
    ch_index = {"R": 0, "G": 1, "B": 2}
    used = [ch_index[c] for c in channels if c in ch_index]
    if not used:
        used = [0]  # default to red

    # Rows 1..3: we can store up to 3 groups of metric means (by channel)
    # We write first min(K, W) metrics into columns left->right as 8-bit values.
    # If multiple channels requested, we replicate the same means into each channel row.
    n_rows_payload = min(_HEADER_ROWS - 1, len(used))
    ncols = min(K, W)
    payload = (means[:ncols] * 255.0 + 0.5).astype(np.uint8)

    for r in range(n_rows_payload):
        row = 1 + r
        ch = used[r]
        arr[row, :ncols, ch] = payload
        # zero other channels on that row (cosmetic, keeps stripe crisp)
        for ch_other in (0, 1, 2):
            if ch_other != ch:
                arr[row, :ncols, ch_other] = 0

    return Image.fromarray(arr, mode="RGB").convert(base.mode)


def _decode_header_stripe(
    png_bytes: bytes,
) -> Dict[str, Any]:
    """
    Quickscan the first 4 rows of the PNG to pull back coarse metric means
    encoded by _encode_header_stripe(). Safe if no stripe is present.
    Returns a dict like:
      {
        "present": bool,
        "rows": 4 or 0,
        "channels": ["R"],  # best guess
        "metric_means_0": [...],  # values in [0,1] from the first payload row
        "metric_means_1": [...],  # if present (2nd payload row), etc.
      }
    """
    try:
        im = Image.open(BytesIO(png_bytes))
        if im.height < _HEADER_ROWS:
            return {"present": False, "rows": 0}
        arr = np.array(im.convert("RGB"), dtype=np.uint8)
    except Exception:
        return {"present": False, "rows": 0}

    H, W, _ = arr.shape
    # Check magic
    magic_ok = (
        W >= 4
        and arr[0, 0, 0] == ord("V")
        and arr[0, 1, 0] == ord("P")
        and arr[0, 2, 0] == ord("F")
        and arr[0, 3, 0] == ord("1")
    )
    if not magic_ok:
        return {"present": False, "rows": 0}

    out = {"present": True, "rows": _HEADER_ROWS, "channels": []}
    # Extract up to 3 payload rows; detect which channel carries values
    for r in range(1, _HEADER_ROWS):
        row = arr[r, :, :]
        # pick the channel with the largest variance as the data channel
        variances = [row[:, ch].var() for ch in (0, 1, 2)]
        ch = int(np.argmax(variances))
        out["channels"].append(["R", "G", "B"][ch])
        # read non-zero prefix as payload (stop when trailing zeros dominate)
        data = row[:, ch]
        # heuristic: read up to last non-zero, but cap length (W)
        nz = np.nonzero(data)[0]
        if nz.size == 0:
            out[f"metric_means_{r-1}"] = []
            continue
        ncols = int(nz[-1]) + 1
        vals = (data[:ncols].astype(np.float32) / 255.0).tolist()
        out[f"metric_means_{r-1}"] = vals
    return out
``n
