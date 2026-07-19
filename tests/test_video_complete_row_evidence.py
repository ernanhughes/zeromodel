from __future__ import annotations

import math

import pytest

from zeromodel.artifact import VPMValidationError
from zeromodel.video_complete_row_evidence import (
    QUANTIZATION_SCALE,
    CompleteRowEvidence,
    SemanticTopSetOutcome,
    VIDEO_RAW_SCORE_DIAGNOSTIC_VERSION,
    VIDEO_QUANTIZED_SCORE_VECTOR_VERSION,
    build_complete_row_evidence,
    build_semantic_top_set_outcome,
    quantize_similarity,
    semantic_top_set_outcome_from_dict,
)


def _rows(count: int = 112) -> list[tuple[str, float]]:
    return [(f"row-{index:03d}", max(0.0, 1.0 - index / 200.0)) for index in range(count)]


def _row_actions(prefix: str = "row") -> dict[str, str]:
    return {f"{prefix}-{index:03d}": ("LEFT" if index % 2 == 0 else "RIGHT") for index in range(112)}


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


def test_semantic_outcome_unique_top_row_resolves_row_and_action() -> None:
    evidence = build_complete_row_evidence(row_scores=_rows(), policy_artifact_id="policy", provider_id="P1", provider_version="v1")
    outcome = build_semantic_top_set_outcome(evidence=evidence, row_action=_row_actions())
    assert outcome.status == "unique_row"
    assert outcome.resolved_row_id == "row-000"
    assert outcome.resolved_action_id == "LEFT"
    assert outcome.semantic_outcome_digest.startswith("sha256:")


def test_semantic_outcome_same_action_tie_resolves_only_action() -> None:
    rows = _rows()
    rows[1] = ("row-001", rows[0][1])
    actions = _row_actions()
    actions["row-001"] = "LEFT"
    outcome = build_semantic_top_set_outcome(
        evidence=build_complete_row_evidence(row_scores=rows, policy_artifact_id="policy", provider_id="P1", provider_version="v1"),
        row_action=actions,
    )
    assert outcome.status == "action_unanimous_tie"
    assert outcome.resolved_row_id is None
    assert outcome.resolved_action_id == "LEFT"


def test_semantic_outcome_conflicting_action_tie_resolves_neither() -> None:
    rows = _rows()
    rows[1] = ("row-001", rows[0][1])
    outcome = build_semantic_top_set_outcome(
        evidence=build_complete_row_evidence(row_scores=rows, policy_artifact_id="policy", provider_id="P1", provider_version="v1"),
        row_action=_row_actions(),
    )
    assert outcome.status == "conflicting_action_tie"
    assert outcome.resolved_row_id is None
    assert outcome.resolved_action_id is None
    assert outcome.rejection_reason


def test_semantic_outcome_lexical_score_order_does_not_change_semantics() -> None:
    rows = _rows()
    rows[1] = ("row-001", rows[0][1])
    actions = _row_actions()
    actions["row-001"] = "LEFT"
    forward = build_semantic_top_set_outcome(
        evidence=build_complete_row_evidence(row_scores=rows, policy_artifact_id="policy", provider_id="P1", provider_version="v1"),
        row_action=actions,
    )
    reversed_scores = build_semantic_top_set_outcome(
        evidence=build_complete_row_evidence(row_scores=list(reversed(rows)), policy_artifact_id="policy", provider_id="P1", provider_version="v1"),
        row_action=dict(reversed(list(actions.items()))),
    )
    assert forward.status == reversed_scores.status
    assert forward.resolved_action_id == reversed_scores.resolved_action_id
    assert forward.semantic_outcome_digest == reversed_scores.semantic_outcome_digest


def test_semantic_outcome_renaming_tied_rows_preserves_semantic_class() -> None:
    rows = [(f"alt-{index:03d}", score) for index, (_row_id, score) in enumerate(_rows())]
    rows[1] = ("alt-001", rows[0][1])
    actions = _row_actions(prefix="alt")
    actions["alt-001"] = "LEFT"
    outcome = build_semantic_top_set_outcome(
        evidence=build_complete_row_evidence(row_scores=rows, policy_artifact_id="policy", provider_id="P1", provider_version="v1"),
        row_action=actions,
    )
    assert outcome.status == "action_unanimous_tie"
    assert outcome.resolved_action_id == "LEFT"


def test_semantic_outcome_rejects_changed_top_row_action() -> None:
    rows = _rows()
    rows[1] = ("row-001", rows[0][1])
    actions = _row_actions()
    actions["row-001"] = "LEFT"
    evidence = build_complete_row_evidence(row_scores=rows, policy_artifact_id="policy", provider_id="P1", provider_version="v1")
    payload = build_semantic_top_set_outcome(evidence=evidence, row_action=actions).to_dict()
    payload["top_row_actions"][1]["action_id"] = "RIGHT"
    with pytest.raises(VPMValidationError):
        semantic_top_set_outcome_from_dict(payload, evidence=evidence, row_action=actions)


def test_semantic_outcome_digest_changes_when_top_quantized_score_changes() -> None:
    actions = _row_actions()
    baseline = build_semantic_top_set_outcome(
        evidence=build_complete_row_evidence(row_scores=_rows(), policy_artifact_id="policy", provider_id="P1", provider_version="v1"),
        row_action=actions,
    )
    changed_rows = _rows()
    changed_rows[0] = ("row-000", 0.999)
    changed = build_semantic_top_set_outcome(
        evidence=build_complete_row_evidence(row_scores=changed_rows, policy_artifact_id="policy", provider_id="P1", provider_version="v1"),
        row_action=actions,
    )
    assert baseline.top_row_ids == changed.top_row_ids
    assert baseline.top_quantized_score != changed.top_quantized_score
    assert baseline.semantic_outcome_digest != changed.semantic_outcome_digest


def test_semantic_outcome_digest_ignores_raw_score_changes_inside_quantized_bin() -> None:
    actions = _row_actions()
    baseline_evidence = build_complete_row_evidence(row_scores=_rows(), policy_artifact_id="policy", provider_id="P1", provider_version="v1")
    tweaked = _rows()
    tweaked[0] = ("row-000", tweaked[0][1] - 1e-12)
    tweaked_evidence = build_complete_row_evidence(row_scores=tweaked, policy_artifact_id="policy", provider_id="P1", provider_version="v1")
    baseline = build_semantic_top_set_outcome(evidence=baseline_evidence, row_action=actions)
    same_bin = build_semantic_top_set_outcome(evidence=tweaked_evidence, row_action=actions)
    assert baseline_evidence.raw_score_diagnostic_digest != tweaked_evidence.raw_score_diagnostic_digest
    assert baseline_evidence.quantized_score_vector_digest == tweaked_evidence.quantized_score_vector_digest
    assert baseline.semantic_outcome_digest == same_bin.semantic_outcome_digest


def test_semantic_outcome_constructor_rejects_invalid_compatibility_fields() -> None:
    rows = _rows()
    rows[1] = ("row-001", rows[0][1])
    actions = _row_actions()
    actions["row-001"] = "LEFT"
    payload = build_semantic_top_set_outcome(
        evidence=build_complete_row_evidence(row_scores=rows, policy_artifact_id="policy", provider_id="P1", provider_version="v1"),
        row_action=actions,
    ).to_dict()
    payload["resolved_row_id"] = "row-000"
    payload["semantic_outcome_digest"] = ""
    with pytest.raises(VPMValidationError):
        SemanticTopSetOutcome(**payload)


def test_semantic_outcome_constructor_rejects_duplicate_top_rows_and_foreign_digest() -> None:
    outcome = build_semantic_top_set_outcome(
        evidence=build_complete_row_evidence(row_scores=_rows(), policy_artifact_id="policy", provider_id="P1", provider_version="v1"),
        row_action=_row_actions(),
    )
    duplicate = outcome.to_dict()
    duplicate["top_row_ids"] = ["row-000", "row-000"]
    duplicate["semantic_outcome_digest"] = ""
    with pytest.raises(VPMValidationError):
        SemanticTopSetOutcome(**duplicate)
    foreign = outcome.to_dict()
    foreign["semantic_outcome_digest"] = "sha256:" + "0" * 64
    with pytest.raises(VPMValidationError):
        SemanticTopSetOutcome(**foreign)

