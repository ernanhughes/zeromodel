from __future__ import annotations

import math

import pytest

from zeromodel.artifact import VPMValidationError
from zeromodel.video_complete_row_evidence import (
    QUANTIZATION_SCALE,
    build_complete_row_evidence,
    quantize_similarity,
)


def _rows(count: int = 112) -> list[tuple[str, float]]:
    return [(f"row-{index:03d}", max(0.0, 1.0 - index / 200.0)) for index in range(count)]


def test_quantize_similarity_frozen_values() -> None:
    assert quantize_similarity(0.0) == 0
    assert quantize_similarity(1.0) == QUANTIZATION_SCALE
    assert quantize_similarity(-1.0) == 0
    assert quantize_similarity(2.0) == QUANTIZATION_SCALE
    assert quantize_similarity(0.5000004) == 500000


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_quantize_similarity_rejects_non_finite(value: float) -> None:
    with pytest.raises(VPMValidationError):
        quantize_similarity(value)


def test_complete_row_evidence_requires_exactly_112_rows() -> None:
    with pytest.raises(VPMValidationError):
        build_complete_row_evidence(
            row_scores=_rows(111),
            policy_artifact_id="policy",
            provider_id="P1",
            provider_version="v1",
        )


def test_complete_row_evidence_rejects_duplicate_rows() -> None:
    rows = _rows()
    rows[-1] = rows[0]
    with pytest.raises(VPMValidationError):
        build_complete_row_evidence(
            row_scores=rows,
            policy_artifact_id="policy",
            provider_id="P1",
            provider_version="v1",
        )


def test_complete_row_evidence_preserves_ties_without_semantic_uniqueness() -> None:
    rows = [(f"row-{index:03d}", 0.25) for index in range(112)]
    evidence = build_complete_row_evidence(
        row_scores=rows,
        policy_artifact_id="policy",
        provider_id="P1",
        provider_version="v1",
    )
    assert len(evidence.ranking.ranked_row_ids) == 112
    assert len(evidence.ranking.tie_groups) == 1
    assert len(evidence.ranking.tie_groups[0].row_ids) == 112
    assert evidence.ranking.ranked_row_ids == tuple(sorted(row_id for row_id, _score in rows))

