"""Dependency-light rendering helpers for ZeroModel v2.

The PNG writer is intentionally tiny and standard-library only. It writes an
8-bit grayscale PNG from a two-dimensional normalized field.
"""
from __future__ import annotations

import binascii
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def as_field(value: Any) -> np.ndarray:
    if hasattr(value, "field"):
        value = value.field
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError("VPM field operations require a 2D matrix")
    return arr


def to_uint8(field: Any) -> np.ndarray:
    arr = as_field(field)
    return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _chunk(kind: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", crc)


def png_bytes(field: Any) -> bytes:
    """Return an 8-bit grayscale PNG byte string for a field or artifact."""
    image = to_uint8(field)
    height, width = image.shape
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 0, 0, 0, 0)
    scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    return PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(scanlines)) + _chunk(b"IEND", b"")


def write_png(field: Any, path: str | Path) -> Path:
    """Write an 8-bit grayscale PNG and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(png_bytes(field))
    return target


def svg_text(field: Any, *, cell_size: int = 16) -> str:
    """Return a simple inline SVG heatmap for a field or artifact."""
    image = to_uint8(field)
    height, width = image.shape
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">'
        % (width * cell_size, height * cell_size, width * cell_size, height * cell_size)
    ]
    for row in range(height):
        for col in range(width):
            value = int(image[row, col])
            parts.append(
                '<rect x="%d" y="%d" width="%d" height="%d" fill="rgb(%d,%d,%d)" />'
                % (col * cell_size, row * cell_size, cell_size, cell_size, value, value, value)
            )
    parts.append("</svg>")
    return "".join(parts)


def write_svg(field: Any, path: str | Path, *, cell_size: int = 16) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(svg_text(field, cell_size=cell_size), encoding="utf-8")
    return target
