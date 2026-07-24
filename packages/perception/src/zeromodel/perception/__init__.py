"""ZeroModel perception public API.

Phase P0 intentionally exposes package identity only. Scientific behavior is
introduced through later tested vertical slices.
"""

from __future__ import annotations

PERCEPTION_PACKAGE_VERSION = "1.0.13"
PERCEPTION_STAGE = "P0"

__all__ = ["PERCEPTION_PACKAGE_VERSION", "PERCEPTION_STAGE"]
