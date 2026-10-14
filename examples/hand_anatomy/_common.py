"""Shared constants and identity helpers (delegates to zeromodel.core)."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _bootstrap  # noqa: F401
from zeromodel.core.content_identity import (
    array_content_digest,
    canonical_json_bytes,
    sha256_digest,
)

EXPERIMENT = "ZM-HAND-01"
GEOMETRY_VERSION = "hand-skeleton/1"
RENDER_RECIPE_VERSION = "hand-capsule-render/1"
TEMPLATE_CANVAS = 160
WORK_CANVAS = 192
NOMINAL_AREA = 2600.0
MAX_IMAGE_SIDE = 2048
MIN_MASK_PIXELS = 400


def digest(value: Any) -> str:
    """Canonical-JSON sha256 digest ("sha256:<hex>")."""
    return sha256_digest(value)


def mask_digest(mask: np.ndarray) -> str:
    return array_content_digest(np.ascontiguousarray(np.asarray(mask, dtype=np.uint8)))


def canonical_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value)


def short(identity: str, n: int = 12) -> str:
    return identity.split(":")[-1][:n]


class HandAnatomyError(ValueError):
    """Base error for invalid experiment inputs."""


class UnsupportedInput(HandAnatomyError):
    """The input cannot be assessed (invalid, too large, empty, ...)."""
