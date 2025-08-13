# zeromodel/provenance/core.py
"""
ZeroModel Provenance Core: Universal Tensor Snapshot + Deterministic Provenance

This module implements the provenance layer for ZeroModel. It extends the
Visual Policy Map (VPM) concept with:
  - A **universal tensor snapshot** (encode ANY Python/NumPy structure into a VPM image,
    then restore it bit-for-bit later).
  - **VPF** (Visual Policy Fingerprint), a compact, verifiable provenance payload that
    captures pipeline, parameters, determinism seeds, inputs, metrics, and lineage.
  - Multiple **embedding strategies** to attach VPF into artifacts (images/binaries):
      • "stripe" — right-edge metrics stripe + PNG footer container
      • "alpha"  — LSB stego in alpha channel
      • "stego"  — LSB stego across RGB channels (with optional stripe)
  - **Extraction + verification** routines to read back the VPF, re-hash the core image
    bytes, and confirm integrity (content hash and VPF hash).
  - **Visual debugging helpers** (compare/logic ops) that operate directly on VPM images.

Design notes
------------
* This file is the **Provenance** implementation. It intentionally duplicates several
  helper routines (e.g., logical ops) so it can remain self-contained and provenance-aware.
  Keeping these local avoids tight coupling with other modules and reduces the chance of
  accidental regressions when core VPM code changes elsewhere.

* Backward/forward-compatibility: where feasible, decoding functions include legacy fallbacks
  (e.g., handling raw zlib(JSON) in footers in addition to the canonical container).

* DO NOT modify the embeddings’ byte/bit layouts unless you also update the corresponding
  extract/verify paths and tests. The on-wire/container formats here are relied upon by
  tests and (potentially) external tooling.

Safety for future contributors / AIs
------------------------------------
* If you change how hashes are computed (e.g., what bytes are hashed), you MUST update
  the verification steps and tests. The verification path expects:
    - lineage.content_hash == SHA3(core PNG bytes without footer)
    - lineage.vpf_hash    == SHA3(JSON of VPF fields with vpf_hash temporarily removed)
* If you add new fields to the VPF JSON, ensure they are stable and serializable and that
  `_compute_vpf_hash` remains consistent across runs (JSON must be `sort_keys=True`).
"""

import json
import zlib
import struct
import hashlib
import base64
import pickle
import numpy as np
from typing import Any, Dict, Tuple, Union, Optional, List
from io import BytesIO
from PIL import Image

from zeromodel.provenance.utils import sha3_bytes

# =============================
# Constants and Configuration
# =============================

# VPF Constants
VPF_VERSION = "1.0"
VPF_MAGIC_HEADER = b"VPF1"  # Magic bytes to identify VPF data inside our container
VPF_FOOTER_MAGIC = b"ZMVF"  # Footer marker appended to PNG/binary artifacts
VPF_SCHEMA_HASH = "sha3-256:8d4a7c3e0b8f1a2d5c6e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
VPF_STRIPE_WIDTH_RATIO = 0.001  # 0.1% of image width for metrics stripe (minimally visible)
VPF_MIN_STRIPE_WIDTH = 1        # Minimum stripe width in pixels

# Tensor serialization constants
TENSOR_COMPRESSION_LEVEL = 9
TENSOR_DTYPE = np.float32
MAX_VPM_DIM = 4096  # Safety bound on generated VPM image dimensions

# Serialization format constants for tensor payloads
FORMAT_NUMERIC = b'F32'  # Legacy numeric-only format (float32)
FORMAT_PICKLE = b'PKL'   # Current canonical format: Python pickle payload

# =============================
# Universal Tensor Operations
# =============================

def tensor_to_vpm(
    tensor: Any,
    quality: int = 95,
    min_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """
    Encode ANY Python/NumPy structure into a VPM image.

    This is the universal "snapshot" operation:
      * Works for scalars, ndarrays (any dtype, shape), dicts/lists/tuples, nested structures.
      * Preserves full fidelity via a short header + pickle payload.
      * The resulting image is purely a carrier; it's not compressed with lossy codecs.

    Args:
        tensor: Arbitrary Python/NumPy object to snapshot.
        quality: Unused in current RGB packing (kept for API symmetry).
        min_size: Optional (width, height) lower bound for the output image to ensure
                  there is enough capacity (useful when you plan to also embed a metrics
                  stripe or stego data later).

    Returns:
        PIL.Image: RGB VPM image containing the serialized tensor (length-prefixed).
    """
    binary_data = _serialize_tensor_with_header(tensor)
    return _binary_to_vpm(binary_data, quality, min_size=min_size)

def vpm_to_tensor(
    vpm: Image.Image
) -> Any:
    """
    Decode a VPM image produced by `tensor_to_vpm` back to the original object.

    This is the universal "restore" operation:
      * Exact bit-for-bit restoration for arbitrary Python/NumPy structures.
      * Uses the on-image length prefix + pickle payload to reconstruct.

    Args:
        vpm: PIL.Image created by `tensor_to_vpm`.

    Returns:
        Any: The exact original object that was encoded.
    """
    binary_data = _vpm_to_binary(vpm)
    return _deserialize_tensor_with_header(binary_data)

def _serialize_tensor_with_header(tensor: Any) -> bytes:
    """
    Serialize an arbitrary object with a tiny format header.

    We use:
      - 3-byte format tag: FORMAT_PICKLE ('PKL') (legacy: 'F32' for raw numeric)
      - 4-byte big-endian payload length
      - payload bytes (pickle.dumps(...))

    This guarantees portable round-trips, including NaNs/inf, and arbitrary nested types.
    """
    data = pickle.dumps(tensor, protocol=pickle.HIGHEST_PROTOCOL)
    return FORMAT_PICKLE + struct.pack('>I', len(data)) + data

def _deserialize_tensor_with_header(binary_data: bytes) -> Any:
    """
    Restore an object serialized by `_serialize_tensor_with_header`.

    Primary path:
      - Read 3-byte format + 4-byte length
      - If 'PKL', unpickle the next `length` bytes
      - If 'F32', reconstruct legacy float32 numeric vector (shape info not stored)

    Falls back to direct pickle.loads(...) if structure doesn't match (for compatibility).
    """
    if len(binary_data) < 7:
        return binary_data

    header = binary_data[:3]
    length = struct.unpack('>I', binary_data[3:7])[0]

    if header == FORMAT_NUMERIC and len(binary_data) >= 7 + length * 4:
        return np.frombuffer(binary_data[7:7+length*4], dtype=TENSOR_DTYPE).reshape(-1)
    elif header == FORMAT_PICKLE and len(binary_data) >= 7 + length:
        return pickle.loads(binary_data[7:7+length])
    else:
        # Compatibility fallback
        try:
            return pickle.loads(binary_data)
        except Exception:
            return binary_data

def _normalize_tensor(tensor: Any) -> Any:
    """
    (Not used by the current serializer but kept for clarity)
    Normalize various inputs into stable forms. Helpful if a future serializer
    wants to ensure shapes/dtypes/keys before encoding.
    """
    if tensor is None:
        return None
    elif isinstance(tensor, (int, float, bool, str)):
        return tensor
    elif isinstance(tensor, np.ndarray):
        return tensor  # preserve dtype/shape
    elif isinstance(tensor, (list, tuple)):
        return type(tensor)(_normalize_tensor(x) for x in tensor)
    elif isinstance(tensor, dict):
        return {str(k): _normalize_tensor(v) for k, v in tensor.items()}
    else:
        return tensor

# Duplicate definition intentionally retained for API stability in existing tests.
def tensor_to_vpm(
    tensor: Any,
    quality: int = 95,
    min_size: Optional[Tuple[int, int]] = None,   # NEW
) -> Image.Image:
    """
    (Alias) Encode object into a VPM image.
    This duplicates the earlier definition on purpose to preserve existing imports/tests.
    """
    binary_data = _serialize_tensor_with_header(tensor)
    return _binary_to_vpm(binary_data, quality, min_size=min_size)

def _binary_to_vpm(binary_data: bytes, quality: int, *, min_size: Optional[Tuple[int,int]] = None) -> Image.Image:
    """
    Pack an arbitrary byte payload into an RGB image with a 4-byte length prefix.

    The payload is written across R,G,B channels in raster order. Zero padding fills
    the tail. If `min_size` is provided, the image is expanded (not shrunk) to satisfy it.
    """
    payload = struct.pack('>I', len(binary_data)) + binary_data
    data_len = len(payload)

    # Minimum square to hold payload across 3 channels
    side = int(np.ceil(np.sqrt(data_len / 3.0)))
    w = h = max(16, side)

    if min_size is not None:
        min_w, min_h = min_size
        w = max(w, int(min_w))
        h = max(h, int(min_h))

    img = Image.new('RGB', (w, h))
    pixels = img.load()

    idx = 0
    for y in range(h):
        for x in range(w):
            r = payload[idx] if idx < data_len else 0; idx += 1
            g = payload[idx] if idx < data_len else 0; idx += 1
            b = payload[idx] if idx < data_len else 0; idx += 1
            img.putpixel((x, y), (r, g, b))
    return img

def _vpm_to_binary(vpm: Image.Image) -> bytes:
    """
    Recover the raw payload from an RGB VPM image written by `_binary_to_vpm`.

    Reads pixels in raster order, then uses the 4-byte big-endian length prefix to
    return exactly the payload bytes (trims any tail padding). Falls back to a legacy
    heuristic (look for 000 triple) if no length prefix seems valid.
    """
    width, height = vpm.size
    pixels = vpm.load()

    # Gather raw bytes interleaving R, G, B
    binary_data = bytearray()
    for y in range(height):
        for x in range(width):
            r, g, b = vpm.getpixel((x, y))
            binary_data.extend((r, g, b))

    # Preferred path: 4-byte length prefix
    if len(binary_data) >= 4:
        try:
            payload_len = struct.unpack('>I', bytes(binary_data[:4]))[0]
            total = 4 + payload_len
            if 0 <= payload_len <= len(binary_data) - 4:
                return bytes(binary_data[4:total])
        except Exception:
            pass

    # Legacy fallback: truncate at first RGB zero-run
    null_idx = -1
    for i in range(len(binary_data) - 2):
        if binary_data[i] == 0 and binary_data[i+1] == 0 and binary_data[i+2] == 0:
            null_idx = i
            break
    return bytes(binary_data[:null_idx]) if null_idx > 0 else bytes(binary_data)

# =============================
# VPF (Visual Policy Fingerprint)
# =============================

def create_vpf(
    pipeline: Dict[str, Any],
    model: Dict[str, Any],
    determinism: Dict[str, Any],
    params: Dict[str, Any],
    inputs: Dict[str, Any],
    metrics: Dict[str, Any],
    lineage: Dict[str, Any],
    signature: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Assemble a VPF (Visual Policy Fingerprint) dictionary.

    The VPF captures everything needed to verify and deterministically replay a step:
      - pipeline: graph hash, step name, etc.
      - model: ids and asset hashes (exact versions)
      - determinism: seeds and RNG backends to recreate randomness
      - params: generation/render/transformation parameters
      - inputs: hashes and descriptors of inputs/prompts/docs
      - metrics: any selection/ranking/quality/safety metrics
      - lineage: parent links, content hash (set during embed), and self vpf_hash
      - signature (optional): for cryptographic signing

    Returns:
        dict: VPF dict with `lineage.vpf_hash` populated.
    """
    vpf = {
        "vpf_version": VPF_VERSION,
        "pipeline": pipeline,
        "model": model,
        "determinism": determinism,
        "params": params,
        "inputs": inputs,
        "metrics": metrics,
        "lineage": lineage,
        "signature": signature
    }

    # Hash the VPF itself (excluding its own hash field)
    vpf_hash = _compute_vpf_hash(vpf)
    vpf["lineage"]["vpf_hash"] = vpf_hash
    return vpf

def _compute_content_hash(data: bytes) -> str:
    """
    SHA3-256 of bytes, returned as 'sha3:<hex>'.
    Used for `lineage.content_hash` over core PNG bytes (no footer)."""
    return f"sha3:{hashlib.sha3_256(data).hexdigest()}"

def _compute_vpf_hash(vpf_dict: Dict[str, Any]) -> str:
    """
    Stable hash of the VPF JSON (with `lineage.vpf_hash` temporarily removed).

    JSON is serialized with `sort_keys=True` to guarantee identical byte stream
    for equal structures across runs/environments.
    """
    clean_dict = vpf_dict.copy()
    if "lineage" in clean_dict:
        lineage = clean_dict["lineage"].copy()
        lineage.pop("vpf_hash", None)
        clean_dict["lineage"] = lineage

    payload = json.dumps(clean_dict, sort_keys=True).encode('utf-8')
    return _compute_content_hash(payload)

def embed_vpf(
    artifact: Union[Image.Image, bytes],
    vpf: Dict[str, Any],
    tensor_state: Optional[Any] = None,
    mode: str = "stego",
    stripe_metrics_matrix: Optional[np.ndarray] = None,
    stripe_metric_names: Optional[List[str]] = None,
    stripe_channels: Tuple[str, ...] = ("R",),
) -> Union[Image.Image, bytes]:
    """
    Embed a VPF into an artifact (image or opaque bytes).

    For images, three modes are supported:
      - "stripe": append a right-edge metrics stripe and a PNG footer with VPF.
      - "alpha" : write VPF bits into alpha LSBs (requires an alpha channel).
      - "stego" : write VPF bits into RGB LSBs; if payload is too large, we
                  automatically fall back to a minimal "stripe" with footer.

    For non-image binary artifacts:
      - We append a footer (ZMVF | len | payload) where payload is the
        canonical container: VPF1 | len | zlib(JSON) [+ optional TNSR blob].

    Args:
        artifact: PIL.Image or bytes.
        vpf: The provenance dictionary created by `create_vpf`.
        tensor_state: Optional model/tensor state to snapshot and attach.
        mode: "stego" (default), "stripe", or "alpha".
        stripe_metrics_matrix: Optional (H-4, M) float32 matrix for quick-scan stripe.
        stripe_metric_names: Metric names aligned with columns of the matrix.
        stripe_channels: Which RGB channels to use per metric column.

    Returns:
        Image (if image input + stego/alpha) or bytes (for "stripe" PNG or binary artifacts).
    """
    tensor_vpm = tensor_to_vpm(tensor_state) if tensor_state is not None else None

    # If stripe args are provided, prefer stripe mode
    if mode == "stego" and (stripe_metrics_matrix is not None or stripe_metric_names is not None):
        mode = "stripe"

    if isinstance(artifact, Image.Image):
        return _embed_vpf_image(
            artifact, vpf, tensor_vpm, mode,
            stripe_metrics_matrix=stripe_metrics_matrix,
            stripe_metric_names=stripe_metric_names,
            stripe_channels=stripe_channels,
        )
    else:
        return _embed_vpf_binary(artifact, vpf, tensor_vpm)

def _embed_vpf_image(
    image: Image.Image,
    vpf: Dict[str, Any],
    tensor_vpm: Optional[Image.Image],
    mode: str = "stego",
    *,
    stripe_metrics_matrix: Optional[np.ndarray] = None,
    stripe_metric_names: Optional[List[str]] = None,
    stripe_channels: Tuple[str, ...] = ("R",),
) -> Union[Image.Image, bytes]:
    """
    Image-specific embedding dispatcher; see `embed_vpf` for details.

    Chooses the requested mode and applies stripe/alpha/stego logic. For "stego",
    if the VPF payload is too large to fit the image, we fall back to a minimal
    "stripe" + footer to guarantee success.
    """
    result = image.copy()

    if mode == "stripe":
        return _embed_vpf_stripe(
            result, vpf, tensor_vpm,
            stripe_metrics_matrix=stripe_metrics_matrix,
            stripe_metric_names=stripe_metric_names,
            stripe_channels=stripe_channels,
        )

    if mode == "alpha" and result.mode in ("RGBA", "LA"):
        return _embed_vpf_alpha(result, vpf, tensor_vpm)

    # default: stego with safe fallback to stripe
    try:
        return _embed_vpf_stego(result, vpf, tensor_vpm)
    except ValueError as e:
        msg = str(e)
        if "VPF too large for steganographic embedding" in msg:
            # Fallback: create a tiny stripe with a single metric if none provided.
            H = result.size[1]
            Hvals = max(1, H - 4)
            if stripe_metrics_matrix is None or stripe_metric_names is None:
                size_kb = len(json.dumps(vpf, sort_keys=True).encode("utf-8")) / 1024.0
                stripe_metrics_matrix = np.full((Hvals, 1), size_kb, dtype=np.float32)
                stripe_metric_names = ["vpf_kb"]
            return _embed_vpf_stripe(
                result, vpf, tensor_vpm,
                stripe_metrics_matrix=stripe_metrics_matrix,
                stripe_metric_names=stripe_metric_names,
                stripe_channels=stripe_channels,
            )
        raise

def _embed_vpf_binary(
    binary_data: bytes,
    vpf: Dict[str, Any],
    tensor_vpm: Optional[Image.Image]
) -> bytes:
    """
    Append a VPF footer to any opaque binary artifact.

    Footer layout:
      ZMVF | uint32(total_len) | payload
    where payload is the canonical container:
      VPF1 | uint32(zlib_json_len) | zlib(JSON)
      [+ optional "TNSR" | uint32(len) | raw_tensor_vpm_bytes]
    """
    vpf_bytes = _serialize_vpf(vpf)

    if tensor_vpm is not None:
        tensor_bytes = _vpm_to_binary(tensor_vpm)
        vpf_bytes += b"TNSR" + struct.pack('>I', len(tensor_bytes)) + tensor_bytes

    footer = VPF_FOOTER_MAGIC + struct.pack('>I', len(vpf_bytes)) + vpf_bytes
    return binary_data + footer

def _serialize_vpf(vpf: Dict[str, Any]) -> bytes:
    """
    Serialize a VPF to a compact binary container:
      VPF1 | uint32(zlib_len) | zlib(JSON)
    """
    json_data = json.dumps(vpf, sort_keys=True).encode('utf-8')
    compressed = zlib.compress(json_data, level=TENSOR_COMPRESSION_LEVEL)
    header = VPF_MAGIC_HEADER
    length = struct.pack('>I', len(compressed))
    return header + length + compressed

def _deserialize_vpf(binary_data: bytes) -> Dict[str, Any]:
    """
    Inverse of `_serialize_vpf`. Validates magic, reads length, inflates zlib(JSON).
    Raises ValueError if the magic header is missing or corrupted.
    """
    if binary_data[:4] != VPF_MAGIC_HEADER:
        raise ValueError("Invalid VPF data - missing magic header")

    length = struct.unpack('>I', binary_data[4:8])[0]
    compressed = binary_data[8:8+length]
    json_data = zlib.decompress(compressed)
    return json.loads(json_data)

def extract_vpf(
    artifact: Union[Image.Image, bytes]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract a VPF and metadata from an artifact.

    For images: try "stripe" → "alpha" → "stego".
    For opaque bytes: parse the ZMVF footer.

    Returns:
        (vpf_dict, metadata_dict)
    """
    if isinstance(artifact, Image.Image):
        return _extract_vpf_image(artifact)
    else:
        return _extract_vpf_binary(artifact)

def _extract_vpf_image(image: Image.Image) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Image extraction pipeline. Prefer the 'stripe' method because it also
    yields a quickscan metrics matrix and verifies CRC/hashes.
    """
    try:
        vpf, metadata = _extract_vpf_stripe(image)
        return vpf, metadata
    except Exception:
        try:
            vpf, metadata = _extract_vpf_alpha(image)
            return vpf, metadata
        except Exception:
            try:
                vpf, metadata = _extract_vpf_stego(image)
                return vpf, metadata
            except Exception as e:
                raise ValueError("No VPF found in the image") from e

def _extract_vpf_binary(binary_data: bytes) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Read a ZMVF footer from opaque bytes and return the decoded VPF.

    Supports both the canonical container (VPF1|len|zlib(JSON)) and a legacy
    "zlib(JSON)-only" mode. Also returns an optional tensor_vpm if present.
    """
    idx = binary_data.rfind(VPF_FOOTER_MAGIC)
    if idx == -1:
        raise ValueError("No VPF footer found in binary data")

    length = struct.unpack('>I', binary_data[idx+4:idx+8])[0]
    vpf_bytes = binary_data[idx+8:idx+8+length]

    if vpf_bytes[:4] == VPF_MAGIC_HEADER:
        vpf = _deserialize_vpf(vpf_bytes)
    else:
        try:
            vpf = json.loads(zlib.decompress(vpf_bytes))
        except Exception as e:
            raise ValueError("Invalid VPF footer payload") from e

    tensor_idx = vpf_bytes.find(b"TNSR")
    if tensor_idx != -1:
        tensor_length = struct.unpack('>I', vpf_bytes[tensor_idx+4:tensor_idx+8])[0]
        tensor_bytes = vpf_bytes[tensor_idx+8:tensor_idx+8+tensor_length]
        tensor_vpm = _binary_to_vpm(tensor_bytes, 95)
        return vpf, {"embedding_mode": "binary_footer", "tensor_vpm": tensor_vpm}

    return vpf, {"embedding_mode": "binary_footer"}

def verify_vpf(
    vpf: Dict[str, Any],
    artifact_data: bytes
) -> bool:
    """
    Verify that a VPF matches the artifact it claims to describe.

    Checks:
      1) Recompute SHA3 over the **core** bytes (artifact without ZMVF footer)
         and compare with `lineage.content_hash`.
      2) Recompute VPF hash (with vpf_hash temporarily removed) and ensure it equals
         `lineage.vpf_hash`.

    Returns:
        True if both checks pass, False otherwise.
    """
    expected_hash = vpf["lineage"]["content_hash"]
    core_bytes = artifact_data
    idx = artifact_data.rfind(VPF_FOOTER_MAGIC)
    if idx != -1:
        core_bytes = artifact_data[:idx]
    actual_hash = _compute_content_hash(core_bytes)

    if expected_hash != actual_hash:
        return False

    vpf_dict = vpf.copy()
    if "lineage" in vpf_dict:
        lineage = vpf_dict["lineage"].copy()
        lineage.pop("vpf_hash", None)
        vpf_dict["lineage"] = lineage

    expected_vpf_hash = _compute_vpf_hash(vpf_dict)
    actual_vpf_hash = vpf["lineage"].get("vpf_hash", "")
    return expected_vpf_hash == actual_vpf_hash

def replay_from_vpf(
    vpf: Dict[str, Any],
    tensor_vpm: Optional[Image.Image] = None,
    resolver: Optional[callable] = None
) -> Any:
    """
    Deterministic replay scaffold.

    If a tensor VPM is provided, this returns the restored tensor state
    directly (i.e., exact internal snapshot). Otherwise, returns the VPF
    dict so the caller can resolve assets by hash, seed RNGs, and re-run
    the pipeline step outside this module.
    """
    if tensor_vpm is not None:
        return vpm_to_tensor(tensor_vpm)
    return vpf

# =============================
# Visual Debugging Tools
# =============================

def compare_vpm(
    vpm1: Image.Image,
    vpm2: Image.Image
) -> Image.Image:
    """
    Visualize the absolute per-pixel channel differences between two VPM images.

    Returns:
        A new RGB image with the red channel carrying the difference magnitude.
        (This is a simple visual cue suitable for quick inspection.)
    """
    arr1 = np.array(vpm1)
    arr2 = np.array(vpm2)

    h1, w1, _ = arr1.shape
    h2, w2, _ = arr2.shape
    h = min(h1, h2)
    w = min(w1, w2)

    diff = np.abs(arr1[:h, :w].astype(np.float32) - arr2[:h, :w].astype(np.float32))
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:, :, 0] = diff[:, :, 0]  # Red channel shows differences
    return Image.fromarray(vis)

def vpm_logic_and(vpm1: Image.Image, vpm2: Image.Image) -> Image.Image:
    """
    Pixel-wise minimum (AND-like) composition of two VPM images.
    Kept locally so provenance module remains self-contained.
    """
    arr1 = np.array(vpm1).astype(np.float32) / 255.0
    arr2 = np.array(vpm2).astype(np.float32) / 255.0
    result = np.minimum(arr1, arr2) * 255.0
    return Image.fromarray(result.astype(np.uint8))

def vpm_logic_or(vpm1: Image.Image, vpm2: Image.Image) -> Image.Image:
    """
    Pixel-wise maximum (OR-like) composition of two VPM images.
    Kept locally so provenance module remains self-contained.
    """
    arr1 = np.array(vpm1).astype(np.float32) / 255.0
    arr2 = np.array(vpm2).astype(np.float32) / 255.0
    result = np.maximum(arr1, arr2) * 255.0
    return Image.fromarray(result.astype(np.uint8))

def vpm_logic_not(vpm: Image.Image) -> Image.Image:
    """
    Per-pixel inversion (NOT-like) of a VPM image.
    """
    arr = np.array(vpm).astype(np.float32)
    result = 255.0 - arr
    return Image.fromarray(result.astype(np.uint8))

def vpm_logic_xor(vpm1: Image.Image, vpm2: Image.Image) -> Image.Image:
    """
    Absolute per-pixel difference (XOR-like) between two VPM images.
    """
    arr1 = np.array(vpm1).astype(np.float32) / 255.0
    arr2 = np.array(vpm2).astype(np.float32) / 255.0
    result = np.abs(arr1 - arr2) * 255.0
    return Image.fromarray(result.astype(np.uint8))

def _extract_vpf_stripe(image: Image.Image) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract a VPF from a PNG with a right-edge metrics stripe and a ZMVF footer.

    Stripe format (rightmost columns):
      - Header column (R channel):
          [0..3]: ASCII 'ZM' 'V' '2' signature (0x5A,0x4D,0x56,0x32)
          [4..5]: uint16 big-endian: number of metric columns (M)
          [6..9]: CRC32 of stripe payload (all metric columns across used channels)
      - For each metric column (per selected channels):
          G channel rows 0..3 store float16 vmin/vmax (2 rows each).
          From row 4..(H-1), used color channel holds the quantized metric values (uint8).

    Footer:
      ZMVF | uint32(total_len) | (VPF1 | uint32(zlib_len) | zlib(JSON) [+ optional TNSR...])

    Returns:
        (vpf_dict, metadata) including quickscan metric means and stripe properties.
    """
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size

    # Locate stripe by scanning for the ZM V2 signature in the right edge
    found = False
    stripe_cols = 0
    arr = np.array(rgb_image)
    for test_cols in range(1, min(256, width) + 1):
        x0 = width - test_cols
        col_r = arr[:, x0, 0]
        if height >= 6 and (int(col_r[0]), int(col_r[1]), int(col_r[2]), int(col_r[3])) == (0x5A, 0x4D, 0x56, 0x32):
            found = True
            stripe_cols = test_cols
            break

    if not found:
        raise ValueError("No metrics stripe header found (ZMVP signature)")

    # Validate CRC over the stripe payload
    x0 = width - stripe_cols
    region = arr[:, x0:width, :]
    m_read = (region[4, 0, 0] << 8) | region[5, 0, 0]

    crc_read = 0
    for i in range(4):
        crc_read = (crc_read << 8) | region[6 + i, 0, 0]

    payload = region[:, 1:1 + m_read, :].tobytes()
    crc_calc = zlib.crc32(payload) & 0xFFFFFFFF
    if crc_calc != crc_read:
        raise ValueError("CRC mismatch in metrics stripe")

    # Reconstruct dequantized metrics for quickscan
    h_vals = height - 4
    mat_q = np.zeros((h_vals, m_read), dtype=np.uint8)
    vmins, vmaxs = [], []

    for j in range(m_read):
        vmin16 = bytes([region[0, 1 + j, 1], region[1, 1 + j, 1]])
        vmax16 = bytes([region[2, 1 + j, 1], region[3, 1 + j, 1]])
        vmin = float(np.frombuffer(vmin16, dtype=np.float16)[0])
        vmax = float(np.frombuffer(vmax16, dtype=np.float16)[0])
        vmins.append(vmin)
        vmaxs.append(vmax)

        q = region[4:4 + h_vals, 1 + j, 0].astype(np.uint8)  # using R channel in this format
        mat_q[:, j] = q

    metrics_matrix = np.zeros_like(mat_q, dtype=np.float32)
    for j in range(m_read):
        metrics_matrix[:, j] = dequantize_column(mat_q[:, j], vmins[j], vmaxs[j])

    # Extract VPF JSON from footer
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    try:
        vpf = read_json_footer(img_bytes.getvalue())
    except Exception as e:
        raise ValueError("Failed to extract VPF from PNG footer") from e

    # Verify content hash matches core PNG (i.e., image without footer)
    idx = img_bytes.getvalue().rfind(VPF_FOOTER_MAGIC)
    if idx == -1:
        raise ValueError("No VPF footer found in image")

    core_image_bytes = img_bytes.getvalue()[:idx]
    core_hash = sha3_bytes(core_image_bytes)

    if core_hash != vpf["lineage"]["content_hash"]:
        raise ValueError("Content hash mismatch between image and VPF")

    metric_names = list(vpf.get("metrics", {}).keys())
    quickscan = {
        "embedding_mode": "stripe",
        "stripe_present": True,
        "stripe_width": stripe_cols,
        "stripe_width_ratio": stripe_cols / width,
        "metrics": {metric_names[i] if i < len(metric_names) else f"m{i}": float(np.mean(metrics_matrix[:, i]))
                    for i in range(m_read)},
        "stripe_pixels": mat_q.tobytes()
    }

    return vpf, quickscan

def _extract_vpf_alpha(image: Image.Image) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract a VPF stored in the LSBs of the alpha channel.

    Requirements:
      - Image mode must have an alpha channel (RGBA/LA/PA).
      - Payload is located by searching for the VPF magic in the bit stream.

    Returns:
        (vpf_dict, metadata)
    """
    if image.mode not in ('RGBA', 'LA', 'PA'):
        raise ValueError("Alpha channel extraction requires RGBA, LA, or PA image mode")

    rgba_image = image if image.mode == 'RGBA' else image.convert('RGBA')
    width, height = rgba_image.size

    # Recover LSB stream
    binary_data = bytearray()
    for y in range(height):
        for x in range(width):
            _, _, _, a = rgba_image.getpixel((x, y))
            binary_data.append(a & 1)

    magic_bits = ''.join(format(b, '08b') for b in VPF_MAGIC_HEADER)
    bit_string = ''.join(str(b) for b in binary_data)

    start_idx = bit_string.find(magic_bits)
    if start_idx == -1:
        raise ValueError("No VPF magic header found in alpha channel")

    byte_start = start_idx // 8

    if len(binary_data) < byte_start + 8:
        raise ValueError("Incomplete VPF data in alpha channel")

    length_bytes = binary_data[byte_start + 4:byte_start + 8]
    payload_length = int.from_bytes(length_bytes, byteorder='big')

    payload_start = byte_start + 8
    payload_end = payload_start + payload_length
    if len(binary_data) < payload_end:
        raise ValueError("Incomplete VPF payload in alpha channel")

    payload_bits = bit_string[8 * payload_start:8 * payload_end]
    payload_bytes = bytearray()
    for i in range(0, len(payload_bits), 8):
        if i + 8 > len(payload_bits):
            break
        payload_bytes.append(int(payload_bits[i:i+8], 2))

    try:
        vpf = _deserialize_vpf(bytes(payload_bytes))
    except Exception as e:
        raise ValueError("Failed to deserialize VPF from alpha channel") from e

    # Verify content hash (prefer RGB-only, else try RGBA)
    rgb_image = rgba_image.convert('RGB')
    img_bytes = BytesIO()
    rgb_image.save(img_bytes, format='PNG')
    core_hash = sha3_bytes(img_bytes.getvalue())

    if core_hash != vpf["lineage"]["content_hash"]:
        img_bytes = BytesIO()
        rgba_image.save(img_bytes, format='PNG')
        core_hash = sha3_bytes(img_bytes.getvalue())
        if core_hash != vpf["lineage"]["content_hash"]:
            raise ValueError("Content hash mismatch between image and VPF")

    metadata = {
        "embedding_mode": "alpha",
        "payload_bits": len(payload_bits),
        "payload_bytes": payload_length,
        "embedding_density": payload_length / (width * height),
        "content_hash_verified": True
    }
    return vpf, metadata

def _extract_vpf_stego(image: Image.Image) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract a VPF stored via RGB LSB steganography.

    Notes:
      - This is a simplified spatial-domain LSB extractor (not DCT-domain).
      - For robustness in production, prefer the "stripe" + footer approach.
    """
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    pixels = np.array(rgb_image)

    # Optional hint: look for stripe to know how many rightmost cols to skip
    has_stripe = False
    stripe_width = 0
    for test_cols in range(1, min(16, width) + 1):
        x0 = width - test_cols
        col = pixels[:, x0, 0]
        if height >= 6 and np.array_equal(col[:4], [0x5A, 0x4D, 0x56, 0x32]):
            has_stripe = True
            stripe_width = test_cols
            break

    # Rebuild LSB bitstream from RGB
    binary_data = bytearray()
    scan_width = width - (stripe_width if has_stripe else 0)
    for y in range(height):
        for x in range(scan_width):
            r, g, b = pixels[y, x]
            binary_data.append(r & 1)
            binary_data.append(g & 1)
            binary_data.append(b & 1)

    # Search for the magic header
    magic_bits = ''.join(format(b, '08b') for b in VPF_MAGIC_HEADER)
    bit_string = ''.join(str(b) for b in binary_data)

    start_idx = bit_string.find(magic_bits)
    if start_idx == -1:
        reverse_magic = ''.join(format(b, '08b')[::-1] for b in VPF_MAGIC_HEADER)
        start_idx = bit_string.find(reverse_magic)
        if start_idx == -1:
            raise ValueError("No VPF magic header found in steganographic data")

    byte_start = start_idx // 8
    if len(binary_data) < byte_start + 8:
        raise ValueError("Incomplete VPF data in steganographic embedding")

    length_bytes = binary_data[byte_start + 4:byte_start + 8]
    payload_length = int.from_bytes(length_bytes, byteorder='big')

    payload_start = byte_start + 8
    payload_end = payload_start + payload_length
    if len(binary_data) < payload_end:
        raise ValueError("Incomplete VPF payload in steganographic embedding")

    payload_bits = bit_string[8 * payload_start:8 * payload_end]
    payload_bytes = bytearray()
    for i in range(0, len(payload_bits), 8):
        if i + 8 > len(payload_bits):
            break
        payload_bytes.append(int(payload_bits[i:i+8], 2))

    try:
        vpf = _deserialize_vpf(bytes(payload_bytes))
    except Exception as e:
        raise ValueError("Failed to deserialize VPF from steganographic embedding") from e

    # Content hash verification: best-effort under stego (pixels altered)
    img_bytes = BytesIO()
    rgb_image.save(img_bytes, format='PNG')
    core_hash = sha3_bytes(img_bytes.getvalue())

    if core_hash != vpf["lineage"]["content_hash"]:
        img_bytes = BytesIO()
        image.save(img_bytes, format=image.format or 'PNG')
        core_hash = sha3_bytes(img_bytes.getvalue())

        if core_hash != vpf["lineage"]["content_hash"] and has_stripe:
            # If stripe is present and metrics match, allow a warning mode
            try:
                metrics_matrix = extract_metrics_stripe(
                    rgb_image, M=len(vpf["metrics"]), channels_per_metric=1, use_channels=("R",)
                )
                for i, (name, value) in enumerate(vpf["metrics"].items()):
                    stripe_mean = float(np.mean(metrics_matrix[:, i]))
                    if abs(stripe_mean - value) > 0.05:
                        raise ValueError(f"Metric mismatch for {name}")
                print("Warning: Content hash mismatch, but metrics match. Proceeding with caution.")
            except Exception as e:
                raise ValueError("Content hash mismatch and metrics verification failed") from e

    metadata = {
        "embedding_mode": "stego",
        "payload_bits": len(payload_bits),
        "payload_bytes": payload_length,
        "embedding_density": payload_length / (width * height * 3),
        "has_metrics_stripe": has_stripe,
        "stripe_width": stripe_width if has_stripe else 0,
        "content_hash_verified": core_hash == vpf["lineage"]["content_hash"]
    }
    return vpf, metadata

def dequantize_column(q: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    Inverse of `quantize_column`. Map uint8 quantized values back to float range [vmin, vmax].

    Handles the degenerate case (vmin ~== vmax) by returning a constant column.
    """
    q = np.asarray(q, dtype=np.float32)
    if np.isclose(vmin, vmax):
        return np.full_like(q, vmin, dtype=np.float32)
    return (q / 255.0) * (vmax - vmin) + vmin

def read_json_footer(png_with_footer: bytes) -> Dict[str, Any]:
    """
    Read the ZMVF footer from PNG bytes and return the VPF dict.

    Canonical container:
      ZMVF | uint32(total_len) | (VPF1 | uint32(zlib_len) | zlib(JSON) [...])

    Legacy mode:
      ZMVF | uint32(total_len) | zlib(JSON)

    Raises:
        ValueError if the footer is missing or corrupt.
    """
    idx = png_with_footer.rfind(VPF_FOOTER_MAGIC)
    if idx == -1:
        raise ValueError("No VPF footer found in the image data")

    total_len = struct.unpack(">I", png_with_footer[idx+4:idx+8])[0]
    buf = memoryview(png_with_footer)[idx+8:idx+8+total_len]

    # Canonical container first
    if len(buf) >= 8 and bytes(buf[:4]) == VPF_MAGIC_HEADER:
        comp_len = struct.unpack(">I", bytes(buf[4:8]))[0]
        comp_end = 8 + comp_len
        return json.loads(zlib.decompress(bytes(buf[8:comp_end])))

    # Legacy: payload is just zlib(JSON)
    return json.loads(zlib.decompress(bytes(buf)))

def _embed_vpf_stripe(
    image: Image.Image,
    vpf: Dict[str, Any],
    tensor_vpm: Optional[Image.Image] = None,
    *,
    stripe_metrics_matrix: Optional[np.ndarray] = None,
    stripe_metric_names: Optional[List[str]] = None,
    stripe_channels: Tuple[str, ...] = ("R",),
) -> bytes:
    """
    Embed VPF into a PNG via:
      1) Right-edge metrics stripe (visible but narrow and CRC-protected).
      2) Footer container (ZMVF → VPF1 → zlib(JSON) [+ TNSR blob if present]).

    Also computes `lineage.content_hash` over the **core** PNG bytes after the stripe
    is added (but before the footer). Then computes and injects `lineage.vpf_hash`.
    """
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size

    # Compute stripe width (must at least fit header + metrics)
    stripe_width = max(VPF_MIN_STRIPE_WIDTH, int(width * VPF_STRIPE_WIDTH_RATIO))

    # Prepare metrics matrix (H-4 rows) & names
    if stripe_metrics_matrix is not None and stripe_metric_names is not None:
        metrics_matrix = stripe_metrics_matrix
        metric_names = list(stripe_metric_names)
    else:
        metric_names = list(vpf.get("metrics", {}).keys())
        metrics_matrix = np.zeros((height - 4, len(metric_names)), dtype=np.float32)
        for i, metric_name in enumerate(metric_names):
            metrics_matrix[:, i] = float(vpf["metrics"][metric_name]) if metric_name in vpf.get("metrics", {}) else 0.0

    channels_per_metric = max(1, len(stripe_channels))
    needed_cols = len(metric_names) * channels_per_metric + 1  # +1 for header column
    if stripe_width < needed_cols:
        stripe_width = needed_cols

    # Paint stripe into right edge
    img_with_stripe, _ = embed_metrics_stripe(
        rgb_image,
        metrics_matrix,
        metric_names,
        stripe_cols=stripe_width,
        use_channels=stripe_channels,
    )

    # Serialize core PNG (with stripe, without footer) and hash it
    buf = BytesIO()
    img_with_stripe.save(buf, format="PNG")
    png_core_bytes = buf.getvalue()

    content_hash_hex = sha3_bytes(png_core_bytes)
    lineage = vpf.get("lineage", {})
    lineage["content_hash"] = f"sha3:{content_hash_hex}"
    vpf["lineage"] = lineage

    # Compute VPF hash after injecting content_hash
    vpf_hash = _compute_vpf_hash(vpf)
    vpf["lineage"]["vpf_hash"] = vpf_hash

    # Build canonical container and append as footer
    json_payload = json.dumps(vpf, sort_keys=True).encode("utf-8")
    comp = zlib.compress(json_payload, level=TENSOR_COMPRESSION_LEVEL)
    container = VPF_MAGIC_HEADER + struct.pack(">I", len(comp)) + comp

    if tensor_vpm is not None:
        tensor_bytes = _vpm_to_binary(tensor_vpm)
        container += b"TNSR" + struct.pack(">I", len(tensor_bytes)) + tensor_bytes

    footer = VPF_FOOTER_MAGIC + struct.pack(">I", len(container)) + container
    return png_core_bytes + footer

def _embed_vpf_alpha(
    image: Image.Image,
    vpf: Dict[str, Any],
    tensor_vpm: Optional[Image.Image] = None
) -> Image.Image:
    """
    Embed VPF bits in alpha channel LSBs (requires RGBA/LA/PA).

    Capacity:
      available_bits = width * height (one bit per pixel in alpha LSB)
    Raises ValueError if payload doesn’t fit.
    """
    if image.mode not in ('RGBA', 'LA', 'PA'):
        raise ValueError("Alpha channel embedding requires RGBA, LA, or PA image mode")

    rgba_image = image if image.mode == 'RGBA' else image.convert('RGBA')
    width, height = rgba_image.size

    vpf_bytes = _serialize_vpf(vpf)
    if tensor_vpm is not None:
        tensor_bytes = _vpm_to_binary(tensor_vpm)
        vpf_bytes += b"TNSR" + struct.pack('>I', len(tensor_bytes)) + tensor_bytes

    binary_data = ''.join(format(byte, '08b') for byte in vpf_bytes)

    required_bits = len(binary_data)
    available_bits = width * height
    if required_bits > available_bits:
        raise ValueError(
            f"VPF too large for alpha channel embedding ({required_bits} bits > {available_bits} bits)"
        )

    idx = 0
    for y in range(height):
        for x in range(width):
            if idx >= required_bits:
                break
            r, g, b, a = rgba_image.getpixel((x, y))
            a = (a & 0xFE) | int(binary_data[idx])
            rgba_image.putpixel((x, y), (r, g, b, a))
            idx += 1

    return rgba_image

def _embed_vpf_stego(
    image: Image.Image,
    vpf: Dict[str, Any],
    tensor_vpm: Optional[Image.Image] = None
) -> Image.Image:
    """
    Embed VPF bits in RGB channel LSBs (spatial-domain stego).

    Capacity:
      available_bits = width * height * 3 (one bit per RGB channel)
    If payload exceeds capacity, a ValueError is raised and the caller
    may fall back to the "stripe" mode (see `_embed_vpf_image`).
    """
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    pixels = rgb_image.load()

    vpf_bytes = _serialize_vpf(vpf)
    if tensor_vpm is not None:
        tensor_bytes = _vpm_to_binary(tensor_vpm)
        vpf_bytes += b"TNSR" + struct.pack('>I', len(tensor_bytes)) + tensor_bytes

    binary_data = ''.join(format(byte, '08b') for byte in vpf_bytes)

    required_bits = len(binary_data)
    available_bits = width * height * 3
    if required_bits > available_bits:
        raise ValueError(
            f"VPF too large for steganographic embedding ({required_bits} bits > {available_bits} bits)"
        )

    idx = 0
    for y in range(height):
        for x in range(width):
            if idx >= required_bits:
                break
            r, g, b = pixels[x, y]
            if idx < required_bits:
                r = (r & 0xFE) | int(binary_data[idx]); idx += 1
            if idx < required_bits:
                g = (g & 0xFE) | int(binary_data[idx]); idx += 1
            if idx < required_bits:
                b = (b & 0xFE) | int(binary_data[idx]); idx += 1
            pixels[x, y] = (r, g, b)

    # Optionally add a tiny stripe with metric echoes (if space)
    metrics_matrix = np.zeros((height - 4, len(vpf["metrics"])), dtype=np.float32)
    metric_names = list(vpf["metrics"].keys())
    for i, metric_name in enumerate(metric_names):
        metrics_matrix[:, i] = vpf["metrics"][metric_name]

    stripe_width = max(1, int(width * VPF_STRIPE_WIDTH_RATIO))
    if stripe_width > 0 and width > stripe_width * 2:
        result = rgb_image.copy()
        return embed_metrics_stripe(
            result, metrics_matrix, metric_names, stripe_cols=stripe_width, use_channels=("R",)
        )[0]

    return rgb_image

def embed_metrics_stripe(
    img: Image.Image,
    metrics_matrix: np.ndarray,
    metric_names: List[str],
    stripe_cols: int = None,
    use_channels=("R",),
):
    """
    Write a CRC-protected metrics stripe into the rightmost columns of an RGB image.

    Layout:
      - Header column (R channel):
          [0..3]: 'Z' 'M' 'V' '2'
          [4..5]: uint16 M (number of metric columns)
          [6..9]: CRC32 of payload across metric columns
      - For each metric j (and each selected channel):
          G channel rows 0..3: vmin/vmax (float16)
          Rows 4..H-1 in the chosen channel: quantized values (uint8)

    Args:
        img: RGB image (PIL.Image).
        metrics_matrix: float32 array, shape (H-4, M).
        metric_names: list of names for the M columns.
        stripe_cols: number of columns to reserve on the right (auto if None).
        use_channels: tuple of channels per metric (e.g., ("R",) or ("R","G")).

    Returns:
        (out_img, info) where out_img is a new image with stripe painted.
    """
    arr = np.array(img.convert("RGB"))
    H, W, _ = arr.shape
    Hm, M = metrics_matrix.shape
    assert Hm <= H - 4, "Not enough height for metrics (need 4 header rows)."
    assert len(metric_names) == M, "Metric names length must match matrix columns."

    ch_idx_map = {"R":0, "G":1, "B":2}
    channel_indices = [ch_idx_map[c] for c in use_channels]
    channels_per_metric = len(channel_indices)

    needed_cols = M * channels_per_metric + 1
    if stripe_cols is None:
        stripe_cols = needed_cols
    assert stripe_cols >= needed_cols, "Stripe too narrow for requested metrics/channels."

    x0 = W - stripe_cols
    region = arr[:, x0:W, :].copy()
    region[:] = 0

    # Signature + metric count in header column (R)
    header_col = 0
    region[0, header_col, 0] = 0x5A  # Z
    region[1, header_col, 0] = 0x4D  # M
    region[2, header_col, 0] = 0x56  # V
    region[3, header_col, 0] = 0x32  # 2
    region[4, header_col, 0] = (M >> 8) & 0xFF
    region[5, header_col, 0] = M & 0xFF

    vmin_list, vmax_list = [], []
    col_offset = 1
    for j in range(M):
        q, vmin, vmax = quantize_column(metrics_matrix[:, j])
        vmin_list.append(vmin); vmax_list.append(vmax)
        for c_i, ch in enumerate(channel_indices):
            col_idx = col_offset + j * channels_per_metric + c_i
            vmin16 = np.frombuffer(np.float16(vmin).tobytes(), dtype=np.uint8)
            vmax16 = np.frombuffer(np.float16(vmax).tobytes(), dtype=np.uint8)
            region[0, col_idx, 1] = int(vmin16[0]); region[1, col_idx, 1] = int(vmin16[1])
            region[2, col_idx, 1] = int(vmax16[0]); region[3, col_idx, 1] = int(vmax16[1])
            region[4:4+len(q), col_idx, ch] = q

    stripe_payload = region[:, 1:1 + M*channels_per_metric, :].tobytes()
    crc = zlib.crc32(stripe_payload) & 0xFFFFFFFF
    for i in range(4):
        region[6+i, header_col, 0] = (crc >> (8*(3-i))) & 0xFF

    arr[:, x0:W, :] = region
    out_img = Image.fromarray(arr, mode="RGB")
    return out_img, {"coverage_percent_width": 100.0 * (stripe_cols / W)}

def extract_metrics_stripe(img: Image.Image, M: int, channels_per_metric: int = 1, use_channels=("R",)):
    """
    Read and validate a right-edge metrics stripe, returning the dequantized matrix.

    Args:
        img: RGB image with a metrics stripe.
        M: expected number of metric columns.
        channels_per_metric: how many channels per metric column were used.
        use_channels: same channel order that was used during embedding.

    Returns:
        np.ndarray float32 of shape (H-4, M).
    """
    arr = np.array(img.convert("RGB"))
    H, W, _ = arr.shape
    ch_idx_map = {"R":0, "G":1, "B":2}
    channel_indices = [ch_idx_map[c] for c in use_channels]

    found = False
    for stripe_cols in range(1, min(256, W)+1):
        x0 = W - stripe_cols
        col = arr[:, x0, 0]
        if H >= 6 and (col[0], col[1], col[2], col[3]) == (0x5A,0x4D,0x56,0x32):
            found = True
            break
    if not found:
        raise ValueError("No metrics stripe header found.")

    region = arr[:, x0:W, :]
    m_read = (int(region[4, 0, 0]) << 8) | int(region[5, 0, 0])
    if M != m_read:
        raise ValueError("Metric count mismatch.")

    crc_read = 0
    for i in range(4):
        crc_read = (crc_read << 8) | int(region[6+i, 0, 0])
    payload = region[:, 1:1 + M * channels_per_metric, :].tobytes()
    crc_calc = zlib.crc32(payload) & 0xFFFFFFFF
    if crc_calc != crc_read:
        raise ValueError("CRC mismatch in metrics stripe.")

    Hvals = H - 4
    mat_q = np.zeros((Hvals, M), dtype=np.uint8)
    vmins, vmaxs = [], []
    col_offset = 1
    for j in range(M):
        vmin16 = bytes([int(region[0, col_offset + j*channels_per_metric, 1]),
                        int(region[1, col_offset + j*channels_per_metric, 1])])
        vmax16 = bytes([int(region[2, col_offset + j*channels_per_metric, 1]),
                        int(region[3, col_offset + j*channels_per_metric, 1])])
        vmin = float(np.frombuffer(vmin16, dtype=np.float16)[0])
        vmax = float(np.frombuffer(vmax16, dtype=np.float16)[0])
        vmins.append(vmin); vmaxs.append(vmax)
        ch = channel_indices[0]
        q = region[4:4+Hvals, col_offset + j*channels_per_metric, ch].astype(np.uint8)
        mat_q[:, j] = q

    out = np.zeros_like(mat_q, dtype=np.float32)
    for j in range(M):
        out[:, j] = dequantize_column(mat_q[:, j], vmins[j], vmaxs[j])
    return out

def quantize_column(vals: np.ndarray):
    """
    Quantize a float32 column to uint8 and return (q, vmin, vmax) for reversible decoding.

    If the column is constant or non-finite, falls back to zeros with default bounds.
    """
    vals = np.asarray(vals, dtype=np.float32)
    vmin = float(np.nanmin(vals)) if np.isfinite(vals).any() else 0.0
    vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
    if np.isclose(vmin, vmax):
        q = np.zeros_like(vals, dtype=np.uint8)
    else:
        q = np.clip(np.round(255.0 * (vals - vmin) / (vmax - vmin)), 0, 255).astype(np.uint8)
    return q, vmin, vmax
