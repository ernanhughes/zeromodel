from __future__ import annotations

import math

import pytest

from zeromodel.artifact import VPMValidationError
from zeromodel.video_complete_row_evidence import (
    QUANTIZATION_SCALE,
    CompleteRowEvidence,
    VIDEO_RAW_SCORE_DIAGNOSTIC_VERSION,
    VIDEO_QUANTIZED_SCORE_VECTOR_VERSION,
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


def test_v2_digest_separates_quantized_identity_from_raw_diagnostics() -> None:
    rows = _rows()
    base = build_complete_row_evidence(row_scores=rows, policy_artifact_id="policy", provider_id="P1", provider_version="v1")
    tweaked = list(rows)
    tweaked[0] = (tweaked[0][0], tweaked[0][1] + 1e-12)
    same_bin = build_complete_row_evidence(row_scores=tweaked, policy_artifact_id="policy", provider_id="P1", provider_version="v1")
    changed = list(rows)
    changed[0] = (changed[0][0], changed[0][1] - 0.001)
    different_bin = build_complete_row_evidence(row_scores=changed, policy_artifact_id="policy", provider_id="P1", provider_version="v1")
    assert base.version != "zeromodel-video-complete-row-evidence/v1"
    assert base.quantized_score_vector_digest == same_bin.quantized_score_vector_digest
    assert base.raw_score_diagnostic_digest != same_bin.raw_score_diagnostic_digest
    assert base.quantized_score_vector_digest != different_bin.quantized_score_vector_digest
    assert base.score_vector_digest == base.quantized_score_vector_digest
    payload = base.to_dict()
    assert payload["quantized_score_vector_digest"].startswith("sha256:")
    assert payload["raw_score_diagnostic_digest"].startswith("sha256:")


def test_complete_row_evidence_rejects_foreign_digest() -> None:
    evidence = build_complete_row_evidence(row_scores=_rows(), policy_artifact_id="policy", provider_id="P1", provider_version="v1")
    data = evidence.__dict__.copy()
    data["policy_row_universe_digest"] = "sha256:" + "0" * 64
    with pytest.raises(VPMValidationError):
        CompleteRowEvidence(**data)

