"""Temporal governance between frame-local visual evidence and VPM policy lookup."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from .artifact import VPMValidationError
from .policy_lookup import PolicyLookupDecision, VPMPolicyLookup
from .policy_transitions import PolicyTransitionEvidence, PolicyTransitionSpec
from .video import VideoClipManifest, VideoFrame, VideoFrameSource
from .visual_address import ImageObservation, VisualAddressContract, VisualAddressDecision, VisualAddressProvider

VIDEO_TEMPORAL_EVIDENCE_VERSION = "zeromodel-video-temporal-evidence/v1"
VIDEO_POLICY_DECISION_VERSION = "zeromodel-video-policy-decision/v1"
VIDEO_POLICY_TRACE_VERSION = "zeromodel-video-policy-trace/v1"


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
            _thaw(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError("video policy values must be JSON-serializable") from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def _contract_tuple(contract: VisualAddressContract) -> Tuple[str, ...]:
    return (
        contract.provider_kind, contract.provider_version,
        contract.score_semantics, contract.address_artifact_id,
        contract.calibration_artifact_id, contract.policy_artifact_id,
    )


def _decision_tuple(decision: VisualAddressDecision) -> Tuple[str, ...]:
    return (
        decision.provider_kind, decision.provider_version,
        decision.score_semantics, decision.address_artifact_id,
        decision.calibration_artifact_id, decision.policy_artifact_id,
    )


@dataclass(frozen=True)
class TemporalEvidence:
    frame_id: str
    row_id: Optional[str]
    previous_accepted_frame_id: Optional[str]
    previous_accepted_row_id: Optional[str]
    transition: PolicyTransitionEvidence
    temporal_persistence: int
    unmatched_duration_frames: int
    identical_frame_run: int
    candidate_path_count: int
    evidence_window: Tuple[str, ...]
    current_frame_independently_supported: bool
    version: str = VIDEO_TEMPORAL_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_TEMPORAL_EVIDENCE_VERSION:
            raise VPMValidationError("unsupported temporal evidence version")
        if not str(self.frame_id):
            raise VPMValidationError("frame_id cannot be empty")
        for name in (
            "temporal_persistence", "unmatched_duration_frames",
            "identical_frame_run", "candidate_path_count",
        ):
            if int(getattr(self, name)) < 0:
                raise VPMValidationError("%s must be non-negative" % name)
        if len(set(self.evidence_window)) != len(self.evidence_window):
            raise VPMValidationError("evidence_window frame IDs must be unique")
        object.__setattr__(self, "temporal_persistence", int(self.temporal_persistence))
        object.__setattr__(self, "unmatched_duration_frames", int(self.unmatched_duration_frames))
        object.__setattr__(self, "identical_frame_run", int(self.identical_frame_run))
        object.__setattr__(self, "candidate_path_count", int(self.candidate_path_count))
        object.__setattr__(self, "evidence_window", tuple(str(item) for item in self.evidence_window))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "frame_id": self.frame_id,
            "row_id": self.row_id,
            "previous_accepted_frame_id": self.previous_accepted_frame_id,
            "previous_accepted_row_id": self.previous_accepted_row_id,
            "transition": self.transition.to_dict(),
            "temporal_persistence": self.temporal_persistence,
            "unmatched_duration_frames": self.unmatched_duration_frames,
            "identical_frame_run": self.identical_frame_run,
            "candidate_path_count": self.candidate_path_count,
            "evidence_window": list(self.evidence_window),
            "current_frame_independently_supported": self.current_frame_independently_supported,
        }


@dataclass(frozen=True)
class VideoPolicyDecision:
    accepted: bool
    reason: str
    frame: VideoFrame
    address: VisualAddressDecision
    temporal: TemporalEvidence
    raw_policy: Optional[PolicyLookupDecision] = None
    policy: Optional[PolicyLookupDecision] = None
    rejection_reasons: Tuple[str, ...] = ()
    version: str = VIDEO_POLICY_DECISION_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_POLICY_DECISION_VERSION:
            raise VPMValidationError("unsupported video policy decision version")
        if not str(self.reason):
            raise VPMValidationError("reason cannot be empty")
        reasons = tuple(str(item) for item in self.rejection_reasons)
        if len(set(reasons)) != len(reasons):
            raise VPMValidationError("rejection reasons must be unique")
        object.__setattr__(self, "rejection_reasons", reasons)
        if self.accepted:
            if self.policy is None or not self.address.accepted:
                raise VPMValidationError("accepted video decision needs address and policy evidence")
            if self.policy.row_id != self.address.matched_row_id:
                raise VPMValidationError("accepted address and policy rows must match")
            if reasons:
                raise VPMValidationError("accepted video decision cannot have rejection reasons")
        elif self.policy is not None or not reasons:
            raise VPMValidationError("rejected video decision needs reasons and no executed policy")

    @property
    def raw_row_id(self) -> Optional[str]:
        return self.address.nearest_row_id

    @property
    def raw_action_id(self) -> Optional[str]:
        return None if self.raw_policy is None else self.raw_policy.action

    @property
    def accepted_row_id(self) -> Optional[str]:
        return None if self.policy is None else self.policy.row_id

    @property
    def accepted_action_id(self) -> Optional[str]:
        return None if self.policy is None else self.policy.action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "accepted": self.accepted,
            "reason": self.reason,
            "rejection_reasons": list(self.rejection_reasons),
            "frame": self.frame.to_descriptor(),
            "raw_row_id": self.raw_row_id,
            "raw_action_id": self.raw_action_id,
            "accepted_row_id": self.accepted_row_id,
            "accepted_action_id": self.accepted_action_id,
            "address": self.address.to_dict(),
            "temporal": self.temporal.to_dict(),
            "raw_policy": None if self.raw_policy is None else self.raw_policy.to_dict(),
            "policy": None if self.policy is None else self.policy.to_dict(),
        }


@dataclass(frozen=True)
class VideoPolicyTrace:
    manifest: VideoClipManifest
    policy_artifact_id: str
    provider_contract_digest: str
    transition_spec_id: str
    decisions: Tuple[VideoPolicyDecision, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = VIDEO_POLICY_TRACE_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_POLICY_TRACE_VERSION:
            raise VPMValidationError("unsupported video policy trace version")
        if len(self.decisions) != self.manifest.frame_count:
            raise VPMValidationError("trace decision count must match manifest")
        if any(not str(getattr(self, name)) for name in (
            "policy_artifact_id", "provider_contract_digest", "transition_spec_id"
        )):
            raise VPMValidationError("trace identities cannot be empty")
        metadata = _freeze(self.metadata)
        _json_bytes(metadata)
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "metadata", metadata)

    @property
    def trace_id(self) -> str:
        return _digest(self.to_dict(include_trace_id=False))

    @property
    def accepted_count(self) -> int:
        return sum(int(item.accepted) for item in self.decisions)

    def to_dict(self, *, include_trace_id: bool = True) -> Dict[str, Any]:
        result = {
            "version": self.version,
            "manifest": self.manifest.to_dict(),
            "policy_artifact_id": self.policy_artifact_id,
            "provider_contract_digest": self.provider_contract_digest,
            "transition_spec_id": self.transition_spec_id,
            "accepted_count": self.accepted_count,
            "rejected_count": len(self.decisions) - self.accepted_count,
            "decisions": [item.to_dict() for item in self.decisions],
            "metadata": _thaw(self.metadata),
        }
        if include_trace_id:
            result["trace_id"] = self.trace_id
        return result


class VideoPolicyReader:
    def __init__(
        self,
        provider: VisualAddressProvider,
        policy_lookup: VPMPolicyLookup,
        transition_spec: PolicyTransitionSpec,
        *,
        evidence_window_size: int = 4,
        maximum_identical_frame_run: Optional[int] = None,
    ) -> None:
        contract = provider.contract()
        contract.validate()
        if contract.policy_artifact_id != policy_lookup.artifact.artifact_id:
            raise VPMValidationError("video provider targets a different policy")
        if set(transition_spec.row_ids) != set(str(item) for item in policy_lookup.artifact.source.row_ids):
            raise VPMValidationError("transition spec must cover the exact policy row set")
        if int(evidence_window_size) < 1:
            raise VPMValidationError("evidence_window_size must be positive")
        if maximum_identical_frame_run is not None and int(maximum_identical_frame_run) < 1:
            raise VPMValidationError("maximum_identical_frame_run must be positive")
        self.provider = provider
        self.policy_lookup = policy_lookup
        self.transition_spec = transition_spec
        self.evidence_window_size = int(evidence_window_size)
        self.maximum_identical_frame_run = (
            None if maximum_identical_frame_run is None
            else int(maximum_identical_frame_run)
        )
        self._contract = contract

    def contract(self) -> VisualAddressContract:
        return self._contract

    @staticmethod
    def _validate_frame(manifest: VideoClipManifest, frame: VideoFrame, index: int) -> None:
        if frame.clip_id != manifest.clip_id:
            raise VPMValidationError("frame clip_id does not match manifest")
        if frame.source_digest != manifest.source_digest:
            raise VPMValidationError("frame source digest does not match manifest")
        if frame.frame_index != index or frame.decoding_order != index:
            raise VPMValidationError("frame order does not match manifest")
        if frame.frame_id != manifest.frame_ids[index]:
            raise VPMValidationError("frame_id does not match manifest")
        if frame.frame_digest != manifest.frame_digests[index]:
            raise VPMValidationError("frame digest does not match manifest")
        if frame.timestamp_seconds != manifest.timestamps_seconds[index]:
            raise VPMValidationError("frame timestamp does not match manifest")
        expected = (manifest.height, manifest.width)
        if manifest.channels != 1:
            expected = (manifest.height, manifest.width, manifest.channels)
        if frame.pixels.shape != expected:
            raise VPMValidationError("frame shape does not match manifest")

    def read(self, source: VideoFrameSource) -> VideoPolicyTrace:
        manifest = source.manifest()
        frames = tuple(source.frames())
        if len(frames) != manifest.frame_count:
            raise VPMValidationError("source frame count does not match manifest")
        decisions = []
        previous_row = previous_frame_id = None
        previous_index = previous_timestamp = None
        previous_rejected = False
        persistence = unmatched = identical = 0
        last_pixel_digest = None
        window = []

        for index, frame in enumerate(frames):
            self._validate_frame(manifest, frame, index)
            identical = identical + 1 if frame.pixel_digest == last_pixel_digest else 1
            last_pixel_digest = frame.pixel_digest
            window = (window + [frame.frame_id])[-self.evidence_window_size:]
            observation = ImageObservation(
                frame.pixels,
                timestamp="%.17g" % frame.timestamp_seconds,
                source_id=frame.frame_id,
                metadata={
                    "clip_id": frame.clip_id,
                    "frame_index": frame.frame_index,
                    "decoding_order": frame.decoding_order,
                    "frame_digest": frame.frame_digest,
                    "pixel_digest": frame.pixel_digest,
                    "source_digest": frame.source_digest,
                },
            )
            address = self.provider.read(observation)
            if _decision_tuple(address) != _contract_tuple(self._contract):
                raise VPMValidationError("visual address decision violates provider contract")
            raw_policy = (
                None if address.nearest_row_id is None
                else self.policy_lookup.read(str(address.nearest_row_id))
            )
            current_row = address.matched_row_id if address.accepted else None
            frame_gap = 0 if previous_index is None else frame.frame_index - previous_index
            timestamp_delta = (
                None if previous_timestamp is None
                else frame.timestamp_seconds - float(previous_timestamp)
            )
            status = self.transition_spec.classify(
                previous_row, current_row, frame_gap=max(1, frame_gap),
                previous_observation_rejected=previous_rejected,
            )
            transition = PolicyTransitionEvidence(
                previous_row_id=previous_row, current_row_id=current_row,
                frame_gap=frame_gap, timestamp_delta_seconds=timestamp_delta,
                status=status, transition_spec_id=self.transition_spec.spec_id,
            )
            reasons = []
            if not address.accepted:
                reasons.append(address.reason)
            if address.accepted and status == "impossible":
                reasons.append("transition_impossible")
            elif address.accepted and status == "unknown_due_to_gap":
                reasons.append("transition_unknown_due_to_gap")
            if self.maximum_identical_frame_run is not None and identical > self.maximum_identical_frame_run:
                reasons.append("stale_repeated_frame")
            accepted = not reasons
            if accepted:
                assert current_row is not None
                policy = self.policy_lookup.read(str(current_row))
                persistence = persistence + 1 if current_row == previous_row else 1
                unmatched = 0
            else:
                policy = None
                persistence = 0
                unmatched += 1
            temporal = TemporalEvidence(
                frame_id=frame.frame_id, row_id=current_row,
                previous_accepted_frame_id=previous_frame_id,
                previous_accepted_row_id=previous_row,
                transition=transition,
                temporal_persistence=persistence if accepted else 0,
                unmatched_duration_frames=unmatched,
                identical_frame_run=identical,
                candidate_path_count=1 if address.nearest_row_id is not None else 0,
                evidence_window=tuple(window),
                current_frame_independently_supported=address.accepted,
            )
            decisions.append(VideoPolicyDecision(
                accepted=accepted,
                reason="accepted" if accepted else reasons[0],
                rejection_reasons=tuple(reasons),
                frame=frame, address=address, temporal=temporal,
                raw_policy=raw_policy, policy=policy,
            ))
            if accepted:
                previous_row, previous_frame_id = current_row, frame.frame_id
                previous_index, previous_timestamp = frame.frame_index, frame.timestamp_seconds
                previous_rejected = False
            else:
                previous_rejected = True

        return VideoPolicyTrace(
            manifest=manifest,
            policy_artifact_id=self.policy_lookup.artifact.artifact_id,
            provider_contract_digest=self._contract.digest,
            transition_spec_id=self.transition_spec.spec_id,
            decisions=tuple(decisions),
            metadata={
                "reader": "VideoPolicyReader",
                "no_silent_carry_forward": True,
                "evidence_window_size": self.evidence_window_size,
                "maximum_identical_frame_run": self.maximum_identical_frame_run,
            },
        )


__all__ = [
    "TemporalEvidence",
    "VIDEO_POLICY_DECISION_VERSION",
    "VIDEO_POLICY_TRACE_VERSION",
    "VIDEO_TEMPORAL_EVIDENCE_VERSION",
    "VideoPolicyDecision",
    "VideoPolicyReader",
    "VideoPolicyTrace",
]
