"""Deterministic temporal source VPMs and incomplete-state diagnosis for Stage P8.

P8 materializes fixed-length, sequence-owned frame windows as addressable montage PNGs.
It then compares exact current-frame identity with exact temporal-context identity to report
when a current image is insufficient to determine the recorded action. This is a dataset
property under the declared encoding and window contract, not a claim about hidden-state
semantics or causality.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np

from .dataset import PerceptionDatasetManifestDTO, RecordedInteractionDTO
from .representation import SourceImageEncoderSpecDTO, SourceVPMDTO, encode_source_array

TEMPORAL_WINDOW_SPEC_VERSION: Final = "perception-temporal-window-spec/1"
TEMPORAL_SOURCE_VERSION: Final = "perception-temporal-source-vpm/1"
TEMPORAL_DIAGNOSIS_VERSION: Final = "perception-temporal-state-diagnosis/1"
TEMPORAL_LAYOUT_SEMANTICS: Final = "oldest_to_current_horizontal_frame_montage"
TEMPORAL_DIAGNOSIS_SEMANTICS: Final = (
    "exact_current_pixel_identity_conflict_resolved_by_exact_prior_context_identity"
)
TEMPORAL_DIAGNOSIS_STATUSES: Final = {
    "no_single_frame_conflict",
    "incomplete_state_confirmed",
    "unresolved_temporal_ambiguity",
    "insufficient_temporal_support",
}


class PerceptionTemporalError(ValueError):
    """Raised when temporal encoding or diagnosis violates the P8 contract."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return f"sha256:{hasher.hexdigest()}"


@dataclass(frozen=True)
class TemporalWindowSpecDTO:
    """Explicit fixed-window and montage layout contract."""

    frame_count: int
    require_contiguous_steps: bool = True
    layout_semantics: str = TEMPORAL_LAYOUT_SEMANTICS
    version: str = TEMPORAL_WINDOW_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.frame_count < 2:
            raise PerceptionTemporalError("temporal frame_count must be at least two")
        if self.layout_semantics != TEMPORAL_LAYOUT_SEMANTICS:
            raise PerceptionTemporalError("unsupported temporal layout semantics")

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "frame_count": self.frame_count,
            "layout_semantics": self.layout_semantics,
            "require_contiguous_steps": self.require_contiguous_steps,
            "version": self.version,
        }

    @property
    def temporal_window_spec_id(self) -> str:
        return _digest(_canonical_json(self.canonical_payload()))


@dataclass(frozen=True)
class TemporalSourceVPMDTO:
    """One fixed temporal window materialized as a canonical montage Source VPM."""

    temporal_source_id: str
    temporal_window_spec_id: str
    sequence_id: str
    target_interaction_id: str
    target_step_index: int
    action_label: str
    frame_source_vpm_ids: tuple[str, ...]
    frame_pixel_digests: tuple[str, ...]
    current_source_vpm_id: str
    current_pixel_digest: str
    montage_source_vpm: SourceVPMDTO
    layout_semantics: str = TEMPORAL_LAYOUT_SEMANTICS
    version: str = TEMPORAL_SOURCE_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.temporal_source_id,
                self.temporal_window_spec_id,
                self.sequence_id,
                self.target_interaction_id,
                self.action_label,
                self.current_source_vpm_id,
                self.current_pixel_digest,
            )
        ):
            raise PerceptionTemporalError("temporal source identities must be non-empty")
        if self.target_step_index < 0:
            raise PerceptionTemporalError("target_step_index must be non-negative")
        if len(self.frame_source_vpm_ids) < 2:
            raise PerceptionTemporalError("temporal source requires at least two frames")
        if len(self.frame_source_vpm_ids) != len(self.frame_pixel_digests):
            raise PerceptionTemporalError("frame ids and pixel digests must align")
        if self.frame_source_vpm_ids[-1] != self.current_source_vpm_id:
            raise PerceptionTemporalError("last temporal frame must be the current source")
        if self.frame_pixel_digests[-1] != self.current_pixel_digest:
            raise PerceptionTemporalError("last temporal digest must be the current digest")
        if self.layout_semantics != TEMPORAL_LAYOUT_SEMANTICS:
            raise PerceptionTemporalError("unsupported temporal layout semantics")

    @property
    def temporal_context_signature(self) -> tuple[str, ...]:
        return self.frame_pixel_digests


@dataclass(frozen=True)
class TemporalConflictGroupDTO:
    """One byte-identical current-frame group and its temporal disambiguation result."""

    group_id: str
    current_pixel_digest: str
    interaction_ids: tuple[str, ...]
    action_labels: tuple[str, ...]
    temporal_source_ids: tuple[str, ...]
    temporal_context_count: int
    temporally_conflicting_context_count: int
    status: str

    def __post_init__(self) -> None:
        if self.status not in TEMPORAL_DIAGNOSIS_STATUSES:
            raise PerceptionTemporalError("unsupported temporal conflict status")
        if not self.group_id or not self.current_pixel_digest or not self.interaction_ids:
            raise PerceptionTemporalError("temporal conflict identities must be non-empty")
        if self.interaction_ids != tuple(sorted(set(self.interaction_ids))):
            raise PerceptionTemporalError("interaction_ids must be unique and sorted")
        if self.action_labels != tuple(sorted(set(self.action_labels))):
            raise PerceptionTemporalError("action_labels must be unique and sorted")
        if self.temporal_source_ids != tuple(sorted(set(self.temporal_source_ids))):
            raise PerceptionTemporalError("temporal_source_ids must be unique and sorted")
        if self.temporal_context_count < 0 or self.temporally_conflicting_context_count < 0:
            raise PerceptionTemporalError("temporal counts must be non-negative")


@dataclass(frozen=True)
class TemporalStateDiagnosisReportDTO:
    report_id: str
    dataset_id: str
    temporal_window_spec_id: str
    split: str
    groups: tuple[TemporalConflictGroupDTO, ...]
    single_frame_conflict_group_count: int
    resolved_by_temporal_context_count: int
    unresolved_temporal_group_count: int
    insufficient_support_group_count: int
    diagnosis_semantics: str = TEMPORAL_DIAGNOSIS_SEMANTICS
    version: str = TEMPORAL_DIAGNOSIS_VERSION

    def __post_init__(self) -> None:
        if not self.report_id or not self.dataset_id or not self.temporal_window_spec_id:
            raise PerceptionTemporalError("temporal diagnosis identities must be non-empty")
        if self.split not in {"train", "validation", "test", "all"}:
            raise PerceptionTemporalError("unsupported temporal diagnosis split")
        if self.groups != tuple(sorted(self.groups, key=lambda item: item.current_pixel_digest)):
            raise PerceptionTemporalError("temporal groups must be sorted by current digest")
        if self.diagnosis_semantics != TEMPORAL_DIAGNOSIS_SEMANTICS:
            raise PerceptionTemporalError("unsupported diagnosis semantics")
        counts = (
            self.single_frame_conflict_group_count,
            self.resolved_by_temporal_context_count,
            self.unresolved_temporal_group_count,
            self.insufficient_support_group_count,
        )
        if any(value < 0 for value in counts):
            raise PerceptionTemporalError("diagnosis counts must be non-negative")


def _selected_interactions(
    manifest: PerceptionDatasetManifestDTO,
    split: str,
) -> tuple[RecordedInteractionDTO, ...]:
    if split not in {"train", "validation", "test", "all"}:
        raise PerceptionTemporalError("split must be train, validation, test, or all")
    selected = tuple(
        item
        for item in manifest.interactions
        if split == "all" or manifest.split_for(item.interaction_id) == split
    )
    if not selected:
        raise PerceptionTemporalError(f"dataset contains no {split!r} interactions")
    return selected


def _montage_array(frames: tuple[SourceVPMDTO, ...]) -> np.ndarray:
    first = frames[0]
    for frame in frames[1:]:
        if (
            frame.encoder_spec_id != first.encoder_spec_id
            or frame.width != first.width
            or frame.height != first.height
            or frame.channels != first.channels
            or frame.color_space != first.color_space
            or frame.dtype != first.dtype
        ):
            raise PerceptionTemporalError(
                "all temporal frames must share encoder, dimensions, channels, color space, and dtype"
            )
    arrays = tuple(frame.to_array() for frame in frames)
    return np.concatenate(arrays, axis=1)


def build_temporal_source_vpms(
    manifest: PerceptionDatasetManifestDTO,
    source_vpms: Mapping[str, SourceVPMDTO],
    spec: TemporalWindowSpecDTO,
    *,
    split: str = "all",
) -> tuple[TemporalSourceVPMDTO, ...]:
    """Build deterministic oldest-to-current montage VPMs from authoritative sequences."""

    selected = _selected_interactions(manifest, split)
    by_sequence: dict[str, list[RecordedInteractionDTO]] = {}
    for interaction in selected:
        by_sequence.setdefault(interaction.sequence_id, []).append(interaction)

    results: list[TemporalSourceVPMDTO] = []
    for sequence_id, items in sorted(by_sequence.items()):
        ordered = sorted(items, key=lambda item: (item.step_index, item.interaction_id))
        for end_index in range(spec.frame_count - 1, len(ordered)):
            window = tuple(ordered[end_index - spec.frame_count + 1 : end_index + 1])
            if spec.require_contiguous_steps:
                steps = tuple(item.step_index for item in window)
                expected = tuple(range(steps[0], steps[0] + spec.frame_count))
                if steps != expected:
                    continue
            try:
                frames = tuple(source_vpms[item.source_vpm_id] for item in window)
            except KeyError as exc:
                raise PerceptionTemporalError(f"missing SourceVPMDTO for {exc.args[0]}") from exc
            for interaction, frame in zip(window, frames):
                if interaction.source_pixel_digest != frame.pixel_digest:
                    raise PerceptionTemporalError(
                        "source pixel identity disagrees with temporal interaction"
                    )
            montage = _montage_array(frames)
            first = frames[0]
            montage_spec = SourceImageEncoderSpecDTO(
                color_space=first.color_space,
                max_width=max(first.width * spec.frame_count, 1),
                max_height=max(first.height, 1),
                max_pixels=max(first.width * spec.frame_count * first.height, 1),
                max_input_bytes=64 * 1024 * 1024,
                version=(
                    f"{TEMPORAL_SOURCE_VERSION};base={first.encoder_spec_id};"
                    f"window={spec.temporal_window_spec_id}"
                ),
            )
            montage_source = encode_source_array(montage, montage_spec)
            current_interaction = window[-1]
            frame_ids = tuple(frame.source_vpm_id for frame in frames)
            frame_digests = tuple(frame.pixel_digest for frame in frames)
            payload: Mapping[str, object] = {
                "action_label": current_interaction.action_label,
                "current_pixel_digest": frames[-1].pixel_digest,
                "current_source_vpm_id": frames[-1].source_vpm_id,
                "frame_pixel_digests": list(frame_digests),
                "frame_source_vpm_ids": list(frame_ids),
                "layout_semantics": TEMPORAL_LAYOUT_SEMANTICS,
                "montage_source_vpm_id": montage_source.source_vpm_id,
                "sequence_id": sequence_id,
                "target_interaction_id": current_interaction.interaction_id,
                "target_step_index": current_interaction.step_index,
                "temporal_window_spec_id": spec.temporal_window_spec_id,
                "version": TEMPORAL_SOURCE_VERSION,
            }
            results.append(
                TemporalSourceVPMDTO(
                    temporal_source_id=_digest(_canonical_json(payload)),
                    temporal_window_spec_id=spec.temporal_window_spec_id,
                    sequence_id=sequence_id,
                    target_interaction_id=current_interaction.interaction_id,
                    target_step_index=current_interaction.step_index,
                    action_label=current_interaction.action_label,
                    frame_source_vpm_ids=frame_ids,
                    frame_pixel_digests=frame_digests,
                    current_source_vpm_id=frames[-1].source_vpm_id,
                    current_pixel_digest=frames[-1].pixel_digest,
                    montage_source_vpm=montage_source,
                )
            )
    return tuple(sorted(results, key=lambda item: item.temporal_source_id))


def diagnose_temporal_state_completeness(
    manifest: PerceptionDatasetManifestDTO,
    temporal_sources: tuple[TemporalSourceVPMDTO, ...],
    spec: TemporalWindowSpecDTO,
    *,
    split: str = "all",
) -> TemporalStateDiagnosisReportDTO:
    """Report whether exact prior context resolves exact current-frame action conflicts."""

    selected = _selected_interactions(manifest, split)
    temporal_by_interaction = {
        item.target_interaction_id: item
        for item in temporal_sources
        if split == "all" or manifest.split_for(item.target_interaction_id) == split
    }
    by_current: dict[str, list[RecordedInteractionDTO]] = {}
    for interaction in selected:
        by_current.setdefault(interaction.source_pixel_digest, []).append(interaction)

    groups: list[TemporalConflictGroupDTO] = []
    for current_digest, interactions in sorted(by_current.items()):
        action_labels = tuple(sorted({item.action_label for item in interactions}))
        available = tuple(
            temporal_by_interaction[item.interaction_id]
            for item in interactions
            if item.interaction_id in temporal_by_interaction
        )
        context_actions: dict[tuple[str, ...], set[str]] = {}
        for temporal in available:
            context_actions.setdefault(temporal.temporal_context_signature, set()).add(
                temporal.action_label
            )
        conflicting_contexts = sum(1 for labels in context_actions.values() if len(labels) > 1)
        if len(action_labels) <= 1:
            status = "no_single_frame_conflict"
        elif len(available) < len(interactions) or len(context_actions) < 2:
            status = "insufficient_temporal_support"
        elif conflicting_contexts > 0:
            status = "unresolved_temporal_ambiguity"
        else:
            status = "incomplete_state_confirmed"
        payload: Mapping[str, object] = {
            "action_labels": list(action_labels),
            "current_pixel_digest": current_digest,
            "interaction_ids": sorted(item.interaction_id for item in interactions),
            "status": status,
            "temporal_context_count": len(context_actions),
            "temporal_source_ids": sorted(item.temporal_source_id for item in available),
            "temporally_conflicting_context_count": conflicting_contexts,
        }
        groups.append(
            TemporalConflictGroupDTO(
                group_id=_digest(_canonical_json(payload)),
                current_pixel_digest=current_digest,
                interaction_ids=tuple(payload["interaction_ids"]),  # type: ignore[arg-type]
                action_labels=action_labels,
                temporal_source_ids=tuple(payload["temporal_source_ids"]),  # type: ignore[arg-type]
                temporal_context_count=len(context_actions),
                temporally_conflicting_context_count=conflicting_contexts,
                status=status,
            )
        )

    ordered_groups = tuple(sorted(groups, key=lambda item: item.current_pixel_digest))
    conflict_groups = tuple(
        item for item in ordered_groups if len(item.action_labels) > 1
    )
    payload = {
        "dataset_id": manifest.dataset_id,
        "diagnosis_semantics": TEMPORAL_DIAGNOSIS_SEMANTICS,
        "groups": [
            {
                "group_id": item.group_id,
                "status": item.status,
            }
            for item in ordered_groups
        ],
        "split": split,
        "temporal_window_spec_id": spec.temporal_window_spec_id,
        "version": TEMPORAL_DIAGNOSIS_VERSION,
    }
    return TemporalStateDiagnosisReportDTO(
        report_id=_digest(_canonical_json(payload)),
        dataset_id=manifest.dataset_id,
        temporal_window_spec_id=spec.temporal_window_spec_id,
        split=split,
        groups=ordered_groups,
        single_frame_conflict_group_count=len(conflict_groups),
        resolved_by_temporal_context_count=sum(
            item.status == "incomplete_state_confirmed" for item in conflict_groups
        ),
        unresolved_temporal_group_count=sum(
            item.status == "unresolved_temporal_ambiguity" for item in conflict_groups
        ),
        insufficient_support_group_count=sum(
            item.status == "insufficient_temporal_support" for item in conflict_groups
        ),
    )
