"""Re-exports of zeromodel.core's canonicalization primitives.

Centralized here so downstream packages (zeromodel-trust,
zeromodel-navigation, ...) depend on `zeromodel.artifacts` for canonical
bytes/digest computation instead of reaching into `zeromodel.core` directly
or reimplementing canonicalization themselves.
"""

from __future__ import annotations

from zeromodel.core.content_identity import canonical_json_bytes, sha256_digest

__all__ = ["canonical_json_bytes", "sha256_digest"]
