"""Shared decode helpers for reconstructing report DTOs from canonical JSON.

Both `report_loading.py` (compiled report) and `adapted_report_persistence.py`
(adapted report) need to turn a decoded canonical JSON payload back into the
same nested DTOs (`AdaptedSubjectDTO`, `AdaptedDimensionDTO`,
`AdaptedValueDTO`, `SourceBindingDTO`, `ReportFindingRefDTO`). Centralizing
that decode logic here means the two loaders cannot silently drift apart on
how a field is reconstructed.
"""

from __future__ import annotations

from typing import Tuple

from zeromodel.artifacts.report_dto import (
    AdaptedDimensionDTO,
    AdaptedSubjectDTO,
    AdaptedValueDTO,
    ReportFindingRefDTO,
    SourceBindingDTO,
)
from zeromodel.artifacts.score_semantics import ScoreSemantics


def decode_attributes(payload: dict) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(payload.items()))


def decode_subject(payload: dict) -> AdaptedSubjectDTO:
    return AdaptedSubjectDTO(
        subject_id=payload["subject_id"],
        label=payload["label"],
        ordinal=payload["ordinal"],
        source_ref=payload["source_ref"],
        attributes=decode_attributes(payload["attributes"]),
    )


def decode_dimension(payload: dict) -> AdaptedDimensionDTO:
    return AdaptedDimensionDTO(
        dimension_id=payload["dimension_id"],
        label=payload["label"],
        score_semantics=ScoreSemantics(payload["score_semantics"]),
        family=payload["family"],
        value_min=payload["value_min"],
        value_max=payload["value_max"],
        target_min=payload["target_min"],
        target_max=payload["target_max"],
        default_importance=payload["default_importance"],
        attributes=decode_attributes(payload["attributes"]),
    )


def decode_finding_ref(payload: dict) -> ReportFindingRefDTO:
    return ReportFindingRefDTO(
        report_id=payload["report_id"],
        finding_id=payload["finding_id"],
        finding_kind=payload["finding_kind"],
    )


def decode_source_binding(payload: dict) -> SourceBindingDTO:
    return SourceBindingDTO(
        subject_id=payload["subject_id"],
        dimension_id=payload["dimension_id"],
        finding_ref=decode_finding_ref(payload["finding_ref"]),
        source_uri=payload["source_uri"],
        source_start=payload["source_start"],
        source_end=payload["source_end"],
        attributes=decode_attributes(payload["attributes"]),
    )


def decode_value(payload: dict) -> AdaptedValueDTO:
    return AdaptedValueDTO(
        subject_id=payload["subject_id"],
        dimension_id=payload["dimension_id"],
        raw_value=payload["raw_value"],
        source_binding=decode_source_binding(payload["source_binding"]),
        confidence=payload["confidence"],
        importance=payload["importance"],
    )
