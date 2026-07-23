"""Shared primitives for the provider-evaluation DTO family."""

from __future__ import annotations

from collections.abc import Mapping

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


def basis_points(value: object, message: str) -> int:
    """A deterministic scaled-integer representation of a `[0, 1]` fraction,
    in ten-thousandths (`0..10000`). Used instead of an identity-bearing
    float so provider confidence has a canonical, exactly-comparable
    identity: `0.95` as a Python `float` is not guaranteed to compare or hash
    identically across serialization round trips the way an integer is.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise VPMValidationError(message)
    if not 0 <= value <= 10_000:
        raise VPMValidationError(message)
    return value


def optional_basis_points(value: object, message: str) -> int | None:
    if value is None:
        return None
    return basis_points(value, message)


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
    "basis_points",
    "decision_payload",
    "nonempty_str",
    "nonneg_int",
    "optional_basis_points",
    "optional_canonical",
    "optional_nonempty_str",
    "optional_nonneg_int",
]
