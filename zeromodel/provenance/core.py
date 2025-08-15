# zeromodel/provenance/core.py
"""
ZeroModel Provenance Core: Universal Tensor Snapshot + Visual Debug Utilities

This module now focuses solely on the VPM (image-backed) tensor snapshot/restore
and simple visualization helpers.

All Visual Policy Fingerprint (VPF) schema, serialization, and PNG embedding/
extraction live in the single canonical module:

    zeromodel.images.vpf

Usage (VPF):
    from zeromodel.images import create_vpf, embed_vpf, extract_vpf, verify_vpf

This file intentionally contains **no** VPF logic to keep things DRY.
"""

from __future__ import annotations

import struct
import pickle
from io import BytesIO
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

# =============================
# Tensor snapshot configuration
# =============================

TENSOR_DTYPE = np.float32

# On-image payload header tags:
FORMAT_NUMERIC = b"F32"  # legacy numeric-only format (float32 vector)
FORMAT_PICKLE = b"PKL"   # canonical format: Python pickle payload


# =============================
# Public API
# =============================

def tensor_to_vpm(
    tensor: Any,
    quality: int = 95,                       # kept for API symmetry; unused in RGB packing
    min_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """
    Encode ANY Python/NumPy structure into a VPM image (RGB carrier).

    Layout in pixels:
        [ 4 bytes length | payload bytes ... ]  // written across R,G,B in raster order

    - Payload uses a tiny header (FORMAT + uint32 length) followed by data.
    - FORMAT = 'PKL' (pickle) for arbitrary objects (recommended).
      Legacy FORMAT = 'F32' encodes a flat float32 vector.

    Args:
        tensor: Arbitrary Python/NumPy object to snapshot.
        quality: Unused (reserved for potential future encoders).
        min_size: Optional (width, height) lower bound for the output image.

    Returns:
        PIL.Image (RGB) carrying the serialized bytes.
    """
    binary_data = _serialize_tensor_with_header(tensor)
    return _binary_to_vpm(binary_data, quality=quality, min_size=min_size)


def vpm_to_tensor(vpm: Image.Image) -> Any:
    """
    Decode a VPM image produced by `tensor_to_vpm` back to the original object.

    Returns:
        The exact original object (bit-for-bit) if it was pickled,
        or a legacy float32 vector for FORMAT_NUMERIC,
        or raw bytes if no valid header is found.
    """
    binary_data = _vpm_to_binary(vpm)
    return _deserialize_tensor_with_header(binary_data)


# =============================
# Internal snapshot helpers
# =============================

def _serialize_tensor_with_header(tensor: Any) -> bytes:
    """
    Serialize an arbitrary object with a tiny format header:

        3-byte FORMAT  |  4-byte big-endian length  |  payload

    - FORMAT 'PKL': payload = pickle.dumps(obj, highest protocol)
    - FORMAT 'F32': legacy float32 vector (shape not stored)
    """
    try:
        data = pickle.dumps(tensor, protocol=pickle.HIGHEST_PROTOCOL)
        return FORMAT_PICKLE + struct.pack(">I", len(data)) + data
    except Exception:
        # As a last resort, try to coerce numeric types to float32 legacy format
        arr = np.asarray(tensor, dtype=TENSOR_DTYPE).ravel()
        raw = arr.tobytes(order="C")
        length_as_floats = len(raw) // 4
        return FORMAT_NUMERIC + struct.pack(">I", length_as_floats) + raw


def _deserialize_tensor_with_header(binary_data: bytes) -> Any:
    """
    Restore an object serialized by `_serialize_tensor_with_header`.

    Primary path:
      - Read 3-byte format + 4-byte length
      - If 'PKL', unpickle the next `length` bytes
      - If 'F32', reconstruct legacy float32 numeric vector (shape info not stored)

    Fallback:
      - If header is missing/invalid, attempt pickle.loads(...)
      - Otherwise return the raw bytes.
    """
    if len(binary_data) >= 7:
        fmt = binary_data[:3]
        try:
            length = struct.unpack(">I", binary_data[3:7])[0]
        except Exception:
            length = None

        if fmt == FORMAT_PICKLE and length is not None and len(binary_data) >= 7 + length:
            return pickle.loads(binary_data[7:7 + length])

        if fmt == FORMAT_NUMERIC and length is not None and len(binary_data) >= 7 + 4 * length:
            vec = np.frombuffer(binary_data[7:7 + 4 * length], dtype=TENSOR_DTYPE)
            return vec.reshape(-1)

    # Compatibility fallbacks
    try:
        return pickle.loads(binary_data)
    except Exception:
        return binary_data


def _binary_to_vpm(
    payload_bytes: bytes,
    *,
    quality: int = 95,                        # reserved; not used
    min_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """
    Pack an arbitrary byte payload into an RGB image with a 4-byte length prefix.

    The bytes are written across R, then G, then B for each pixel in raster order.
    If `min_size` is provided, the image is expanded (never shrunk) to satisfy it.
    """
    payload = struct.pack(">I", len(payload_bytes)) + payload_bytes
    total = len(payload)

    # Minimum square to hold payload across 3 channels
    side = int(np.ceil(np.sqrt(total / 3.0)))
    w = h = max(16, side)

    if min_size is not None:
        min_w, min_h = int(min_size[0]), int(min_size[1])
        w = max(w, min_w)
        h = max(h, min_h)

    img = Image.new("RGB", (w, h))
    idx = 0
    for y in range(h):
        for x in range(w):
            r = payload[idx] if idx < total else 0; idx += 1
            g = payload[idx] if idx < total else 0; idx += 1
            b = payload[idx] if idx < total else 0; idx += 1
            img.putpixel((x, y), (r, g, b))
    return img


def _vpm_to_binary(vpm: Image.Image) -> bytes:
    """
    Recover the raw payload from an RGB VPM image written by `_binary_to_vpm`.

    Reads pixels in raster order, then uses the 4-byte big-endian length prefix to
    return exactly the payload bytes (trims any tail padding). If the prefix is
    missing/invalid, a legacy heuristic trims at the first RGB (0,0,0) triple.
    """
    w, h = vpm.size
    data = bytearray()
    px = vpm.convert("RGB")
    for y in range(h):
        for x in range(w):
            r, g, b = px.getpixel((x, y))
            data.extend((r, g, b))

    # Preferred path: 4-byte length prefix
    if len(data) >= 4:
        try:
            payload_len = struct.unpack(">I", bytes(data[:4]))[0]
            total = 4 + payload_len
            if 0 <= payload_len <= len(data) - 4:
                return bytes(data[4:total])
        except Exception:
            pass

    # Legacy fallback: stop at first (0,0,0)
    for i in range(0, len(data) - 2, 3):
        if data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 0:
            return bytes(data[:i])
    return bytes(data)


# =============================
# Visual debugging helpers
# =============================

def compare_vpm(vpm1: Image.Image, vpm2: Image.Image) -> Image.Image:
    """
    Visualize absolute per-pixel channel differences between two VPM images.

    Red channel carries the magnitude of the difference (simple, fast cue).
    """
    arr1 = np.array(vpm1.convert("RGB"))
    arr2 = np.array(vpm2.convert("RGB"))

    h = min(arr1.shape[0], arr2.shape[0])
    w = min(arr1.shape[1], arr2.shape[1])

    diff = np.abs(arr1[:h, :w].astype(np.float32) - arr2[:h, :w].astype(np.float32))
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:, :, 0] = diff[:, :, 0]  # red
    return Image.fromarray(vis)


def vpm_logic_and(vpm1: Image.Image, vpm2: Image.Image) -> Image.Image:
    """Pixel-wise minimum (AND-like) composition of two VPM images."""
    a = np.array(vpm1.convert("RGB")).astype(np.float32) / 255.0
    b = np.array(vpm2.convert("RGB")).astype(np.float32) / 255.0
    out = (np.minimum(a, b) * 255.0).astype(np.uint8)
    return Image.fromarray(out)


def vpm_logic_or(vpm1: Image.Image, vpm2: Image.Image) -> Image.Image:
    """Pixel-wise maximum (OR-like) composition of two VPM images."""
    a = np.array(vpm1.convert("RGB")).astype(np.float32) / 255.0
    b = np.array(vpm2.convert("RGB")).astype(np.float32) / 255.0
    out = (np.maximum(a, b) * 255.0).astype(np.uint8)
    return Image.fromarray(out)


def vpm_logic_not(vpm: Image.Image) -> Image.Image:
    """Per-pixel inversion (NOT-like) of a VPM image."""
    arr = np.array(vpm.convert("RGB")).astype(np.float32)
    out = (255.0 - arr).astype(np.uint8)
    return Image.fromarray(out)


def vpm_logic_xor(vpm1: Image.Image, vpm2: Image.Image) -> Image.Image:
    """Absolute per-pixel difference (XOR-like) between two VPM images."""
    a = np.array(vpm1.convert("RGB")).astype(np.float32) / 255.0
    b = np.array(vpm2.convert("RGB")).astype(np.float32) / 255.0
    out = (np.abs(a - b) * 255.0).astype(np.uint8)
    return Image.fromarray(out)


# =============================
# Optional explicit exports
# =============================

__all__ = [
    "tensor_to_vpm",
    "vpm_to_tensor",
    "compare_vpm",
    "vpm_logic_and",
    "vpm_logic_or",
    "vpm_logic_not",
    "vpm_logic_xor",
]
