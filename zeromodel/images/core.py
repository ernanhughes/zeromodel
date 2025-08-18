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
