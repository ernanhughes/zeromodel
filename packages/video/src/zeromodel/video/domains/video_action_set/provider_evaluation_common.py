"""Shared primitives for the provider-evaluation DTO family."""

from __future__ import annotations

from collections.abc import Mapping
import math

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.dto import CanonicalJsonDTO


def nonempty_str(value: object, message: str) -> str:
    if not isinstance(value, str) or not value:
        raise VPMValidationError(message)
    return value


def optional_nonempty_str(value: object, message: str) -> str | None:
    if value is None:
        return None
    return nonempty_str(value, message)


def nonneg_int(value: object, message: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise VPMValidationError(message)
    return value


def optional_nonneg_int(value: object, message: str) -> int | None:
    if value is None:
        return None
    return nonneg_int(value, message)


def finite_unit_float(value: object, message: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise VPMValidationError(message)
    result = float(value)
    if not (0.0 <= result <= 1.0):
        raise VPMValidationError(message)
    return result


def optional_confidence(value: object, message: str) -> float | None:
    if value is None:
        return None
    return finite_unit_float(value, message)


def optional_canonical(value: object) -> CanonicalJsonDTO | None:
    if value is None:
        return None
    return CanonicalJsonDTO.from_value(value)


def decision_payload(decision: object) -> Mapping[str, object]:
    """Normalize a `PolicyLookupDecision`-shaped object (or a plain mapping
    with the same shape) into a mapping, without importing
    `zeromodel.core.policy_lookup` - this stays decoupled from that specific
    type."""
    to_dict = getattr(decision, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(decision, Mapping):
        return decision
    raise VPMValidationError("policy decision trace mismatch")


__all__ = [
    "decision_payload",
    "finite_unit_float",
    "nonempty_str",
    "nonneg_int",
    "optional_canonical",
    "optional_confidence",
    "optional_nonempty_str",
    "optional_nonneg_int",
]
