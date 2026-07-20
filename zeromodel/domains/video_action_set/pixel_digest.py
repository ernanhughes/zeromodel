"""Historical SHA-256 identity over contiguous array bytes.

This helper preserves the supplied array dtype. Callers whose contracts declare
uint8 pixels must normalize to uint8 before invoking it.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ...artifact import VPMValidationError


def pixel_digest_from_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def pixel_digest(pixels: object | None) -> str | None:
    if pixels is None:
        return None
    return pixel_digest_from_bytes(np.ascontiguousarray(pixels).tobytes(order="C"))


def array_digest(pixels: object) -> str:
    digest = pixel_digest(pixels)
    if digest is None:
        raise VPMValidationError("array digest requires pixels")
    return digest


__all__ = ["array_digest", "pixel_digest", "pixel_digest_from_bytes"]
