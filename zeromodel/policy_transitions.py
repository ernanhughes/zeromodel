"""Versioned transition contracts for bounded policy-row sequences."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from .artifact import VPMValidationError


POLICY_TRANSITION_SPEC_VERSION = "zeromodel-policy-transition-spec/v1"
POLICY_TRANSITION_EVIDENCE_VERSION = "zeromodel-policy-transition-evidence/v1"
_TRANSITION_STATUSES = {
    "initial",
    "possible",
    "possible_with_gap",
    "impossible",
    "unknown_due_to_gap",
    "unknown_due_to_rejection",
}


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _thaw(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError("transition values must be JSON-serializable") from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class PolicyTransitionSpec:
    """Declared row-transition graph for one bounded policy domain."""

    allowed_row_transitions: Mapping[str, Tuple[str, ...]]
    maximum_frame_gap: int = 1
    maximum_position_delta: Optional[int] = None
    action_conditioned: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = POLICY_TRANSITION_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.version != POLICY_TRANSITION_SPEC_VERSION:
            raise VPMValidationError("unsupported policy transition spec version")
        if int(self.maximum_frame_gap) < 1:
            raise VPMValidationError("maximum_frame_gap must be at least one")
        if self.maximum_position_delta is not None and int(self.maximum_position_delta) < 0:
            raise VPMValidationError("maximum_position_delta must be non-negative")
        frozen: Dict[str, Tuple[str, ...]] = {}
        for source, destinations in self.allowed_row_transitions.items():
            source_id = str(source)
            if not source_id:
                raise VPMValidationError("transition source row cannot be empty")
            values = tuple(str(value) for value in destinations)
            if any(not value for value in values):
                raise VPMValidationError("transition destination row cannot be empty")
            if len(set(values)) != len(values):
                raise VPMValidationError("transition destinations must be unique")
            frozen[source_id] = tuple(sorted(values))
        if not frozen:
            raise VPMValidationError("transition graph cannot be empty")
        known = set(frozen)
        unknown = sorted({item for values in frozen.values() for item in values if item not in known})
        if unknown:
            raise VPMValidationError("transition graph references unknown rows: %s" % unknown)
        metadata = _freeze(self.metadata)
        _json_bytes(metadata)
        object.__setattr__(self, "allowed_row_transitions", MappingProxyType(frozen))
        object.__setattr__(self, "maximum_frame_gap", int(self.maximum_frame_gap))
        object.__setattr__(
            self,
            "maximum_position_delta",
            None if self.maximum_position_delta is None else int(self.maximum_position_delta),
        )
        object.__setattr__(self, "action_conditioned", bool(self.action_conditioned))
        object.__setattr__(self, "metadata", metadata)

    @property
    def spec_id(self) -> str:
        return _digest(self.to_dict())

    @property
    def row_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self.allowed_row_transitions))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "allowed_row_transitions": {
                key: list(value) for key, value in sorted(self.allowed_row_transitions.items())
            },
            "maximum_frame_gap": self.maximum_frame_gap,
            "maximum_position_delta": self.maximum_position_delta,
            "action_conditioned": self.action_conditioned,
            "metadata": _thaw(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyTransitionSpec":
        return cls(
            allowed_row_transitions={
                str(key): tuple(str(item) for item in value)
                for key, value in dict(data["allowed_row_transitions"]).items()
            },
            maximum_frame_gap=int(data.get("maximum_frame_gap", 1)),
            maximum_position_delta=(
                None
                if data.get("maximum_position_delta") is None
                else int(data["maximum_position_delta"])
            ),
            action_conditioned=bool(data.get("action_conditioned", False)),
            metadata=data.get("metadata") or {},
            version=str(data.get("version", POLICY_TRANSITION_SPEC_VERSION)),
        )

    def _reachable_within(self, source_row_id: str, target_row_id: str, steps: int) -> bool:
        queue = deque([(source_row_id, 0)])
        visited = {(source_row_id, 0)}
        while queue:
            row_id, depth = queue.popleft()
            if depth >= steps:
                continue
            for destination in self.allowed_row_transitions.get(row_id, ()):
                if destination == target_row_id:
                    return True
                state = (destination, depth + 1)
                if state not in visited:
                    visited.add(state)
                    queue.append(state)
        return False

    def classify(
        self,
        previous_row_id: Optional[str],
        current_row_id: Optional[str],
        *,
        frame_gap: int,
        previous_observation_rejected: bool = False,
    ) -> str:
        if previous_row_id is None:
            return "initial"
        if current_row_id is None:
            return "unknown_due_to_rejection"
        if previous_row_id not in self.allowed_row_transitions:
            raise VPMValidationError("unknown previous policy row")
        if current_row_id not in self.allowed_row_transitions:
            raise VPMValidationError("unknown current policy row")
        gap = int(frame_gap)
        if gap < 1:
            raise VPMValidationError("frame_gap must be positive")
        if gap == 1:
            return (
                "possible"
                if current_row_id in self.allowed_row_transitions[previous_row_id]
                else "impossible"
            )
        if gap > self.maximum_frame_gap:
            return "unknown_due_to_gap"
        if previous_observation_rejected and self._reachable_within(previous_row_id, current_row_id, gap):
            return "possible_with_gap"
        if self._reachable_within(previous_row_id, current_row_id, gap):
            return "possible_with_gap"
        return "unknown_due_to_gap"


@dataclass(frozen=True)
class PolicyTransitionEvidence:
    previous_row_id: Optional[str]
    current_row_id: Optional[str]
    frame_gap: int
    timestamp_delta_seconds: Optional[float]
    status: str
    transition_spec_id: str
    version: str = POLICY_TRANSITION_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != POLICY_TRANSITION_EVIDENCE_VERSION:
            raise VPMValidationError("unsupported transition evidence version")
        if self.status not in _TRANSITION_STATUSES:
            raise VPMValidationError("unsupported transition status")
        if int(self.frame_gap) < 0:
            raise VPMValidationError("frame_gap must be non-negative")
        if not str(self.transition_spec_id):
            raise VPMValidationError("transition_spec_id cannot be empty")
        object.__setattr__(self, "frame_gap", int(self.frame_gap))
        object.__setattr__(
            self,
            "timestamp_delta_seconds",
            None if self.timestamp_delta_seconds is None else float(self.timestamp_delta_seconds),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "previous_row_id": self.previous_row_id,
            "current_row_id": self.current_row_id,
            "frame_gap": self.frame_gap,
            "timestamp_delta_seconds": self.timestamp_delta_seconds,
            "status": self.status,
            "transition_spec_id": self.transition_spec_id,
        }
