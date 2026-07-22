from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

import research.benchmarks.video_action_set_benchmark as benchmark
from zeromodel.video.domains.video_action_set import reference_verification as verification
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from research.evidence.video_complete_row_evidence import (
    CompleteRanking,
    CompleteRowEvidence,
    SemanticTopSetOutcome,
    TieGroup,
    build_complete_row_evidence,
    build_semantic_top_set_outcome,
)


class _TieGroup:
    def __init__(self, rows: tuple[str, ...]) -> None:
        self._rows = rows

    def to_dict(self) -> dict[str, object]:
        return {"row_ids": list(self._rows)}


def _result(*, quantized: tuple[int, ...], ranking: tuple[str, ...]):
    complete_ranking = SimpleNamespace(
        ranked_row_ids=ranking,
        tie_groups=(_TieGroup(ranking),),
        ranking_digest="sha256:ranking",
    )
    evidence = SimpleNamespace(
        ranking=complete_ranking,
        score_vector_digest="sha256:scores",
    )
    return SimpleNamespace(quantized_scores=quantized, evidence=evidence)


def _row_ids() -> list[str]:
    return [f"row-{index:03d}" for index in range(112)]


def _row_actions() -> dict[str, str]:
    return {
        row_id: ("LEFT" if index % 2 == 0 else "RIGHT")
        for index, row_id in enumerate(_row_ids())
    }


def _complete_evidence_row(
    *,
    tied_top: bool = False,
    unanimous_action: bool = False,
) -> tuple[dict[str, object], CompleteRowEvidence, SemanticTopSetOutcome]:
    row_ids = _row_ids()
    scores = [max(0.0, 1.0 - index / 200.0) for index in range(112)]
    actions = _row_actions()
    if tied_top:
        scores[1] = scores[0]
    if unanimous_action:
        actions[row_ids[1]] = actions[row_ids[0]]
    evidence = build_complete_row_evidence(
        row_scores=list(zip(row_ids, scores)),
        policy_artifact_id="policy:frozen",
        provider_id="P1",
        provider_version="provider:v1",
        policy_row_ids=row_ids,
    )
    outcome = build_semantic_top_set_outcome(evidence=evidence, row_action=actions)
    score_rows = evidence.to_dict()["row_scores"]
    row: dict[str, object] = {
        "frame_id": "frame:frozen",
        "policy_artifact_id": evidence.policy_artifact_id,
        "provider_id": evidence.provider_id,
        "provider_version": evidence.provider_version,
        "all_112_row_ids": [item["row_id"] for item in score_rows],
        "all_112_raw_scores": [item["raw_score"] for item in score_rows],
        "all_112_quantized_scores": [item["quantized_score"] for item in score_rows],
        "complete_ordered_ranking": list(evidence.ranking.ranked_row_ids),
        "tie_groups": [group.to_dict() for group in evidence.ranking.tie_groups],
        "policy_row_universe_digest": evidence.policy_row_universe_digest,
        "quantized_score_vector_digest": evidence.quantized_score_vector_digest,
        "raw_score_diagnostic_digest": evidence.raw_score_diagnostic_digest,
        "score_vector_digest": evidence.score_vector_digest,
        "ranking_digest": evidence.ranking.ranking_digest,
        "semantic_top_set_outcome": outcome.to_dict(),
    }
    return row, evidence, outcome


@pytest.fixture(scope="module")
def complete_row_fixture() -> tuple[
    dict[str, object], CompleteRowEvidence, SemanticTopSetOutcome
]:
    return _complete_evidence_row()


def test_reference_legacy_helpers_are_direct_aliases() -> None:
    assert benchmark._finding is verification._finding
    assert benchmark._gate is verification._gate
    assert benchmark._primary_failure is verification._primary_failure
    assert (
        benchmark._stored_quantized_evidence is verification._stored_quantized_evidence
    )
    assert (
        benchmark._expected_semantic_for_row is verification._expected_semantic_for_row
    )


def test_reference_failure_precedence_and_key_order_are_frozen() -> None:
    gates = [
        verification._gate(
            "semantic_outcome",
            [verification._finding("ranking_reconstruction_mismatch", "ranking")],
        ),
        verification._gate(
            "structural_identity",
            [verification._finding("provider_contract_mismatch", "provider")],
        ),
    ]
    report = {"gates": gates}

    assert list(gates[0]) == ["gate", "status", "finding_count", "findings", "counts"]
    assert verification._primary_failure(report) == {
        "code": "provider_contract_mismatch",
        "gate": "structural_identity",
        "finding": {
            "code": "provider_contract_mismatch",
            "message": "provider",
        },
    }
    assert verification._report_failure_codes(report) == [
        {"gate": "structural_identity", "code": "provider_contract_mismatch"},
        {"gate": "semantic_outcome", "code": "ranking_reconstruction_mismatch"},
    ]


def test_provider_equivalence_kernel_preserves_provider_order() -> None:
    same = _result(quantized=(1, 2), ranking=("r1", "r2"))
    different = _result(quantized=(2, 1), ranking=("r2", "r1"))
    comparisons = [
        verification.compare_provider_results(
            provider_id="P2",
            observation_id="frame:2",
            reference=same,
            optimized=different,
        ),
        verification.compare_provider_results(
            provider_id="P1",
            observation_id="frame:1",
            reference=same,
            optimized=same,
        ),
    ]

    payload = verification.build_provider_equivalence_payload(comparisons)

    assert list(payload["summary"]) == ["P1", "P2", "P3"]
    assert payload["mismatching_providers"] == ["P2"]
    assert payload["summary"]["P2"] == {
        "quantized_mismatch_count": 1,
        "ranking_mismatch_count": 1,
        "tie_group_mismatch_count": 1,
        "digest_mismatch_count": 0,
    }


def test_valid_complete_row_reconstruction_has_frozen_identities(
    complete_row_fixture: tuple[
        dict[str, object], CompleteRowEvidence, SemanticTopSetOutcome
    ],
) -> None:
    row, evidence, outcome = complete_row_fixture

    rebuilt, findings = verification._stored_quantized_evidence(row, _row_ids())
    semantic, semantic_findings = verification._expected_semantic_for_row(
        row,
        _row_actions(),
        _row_ids(),
    )

    assert isinstance(evidence, CompleteRowEvidence)
    assert isinstance(evidence.ranking, CompleteRanking)
    assert all(isinstance(group, TieGroup) for group in evidence.ranking.tie_groups)
    assert isinstance(outcome, SemanticTopSetOutcome)
    assert findings == []
    assert semantic_findings == []
    assert rebuilt == evidence
    assert semantic == outcome
    assert evidence.quantized_score_vector_digest == (
        "sha256:4afd0132267b3d093bf9f5a3082303963d78fbf0cb18fab3e0af0cccb81e7398"
    )
    assert evidence.raw_score_diagnostic_digest == (
        "sha256:103120c74d6042c463993bc2f787b3f3d02cb528f585ab6632636661c3423153"
    )
    assert evidence.ranking.ranking_digest == (
        "sha256:b28cc4bbfeda234d128e7110c3395c19c99bb318e7df52f5ef7440ceb185be8b"
    )
    assert canonical_sha256([group.to_dict() for group in evidence.ranking.tie_groups]) == (
        "sha256:4b8c500d6ea5469e5f46749580871b1ffe9151c6fbae45adc46bc706a173471d"
    )
    assert outcome.semantic_outcome_digest == (
        "sha256:8630af8ec49a3209472ee7178d5fc23a45550055d4a74e6eb01dae9ac9455229"
    )


def test_complete_row_mismatch_order_and_primary_precedence_are_frozen(
    complete_row_fixture: tuple[
        dict[str, object], CompleteRowEvidence, SemanticTopSetOutcome
    ],
) -> None:
    row = deepcopy(complete_row_fixture[0])
    row["all_112_quantized_scores"][0] -= 1  # type: ignore[index]
    row["quantized_score_vector_digest"] = "sha256:foreign-quantized"
    row["score_vector_digest"] = "sha256:foreign-score"
    row["raw_score_diagnostic_digest"] = "sha256:foreign-raw"
    row["complete_ordered_ranking"] = list(
        reversed(row["complete_ordered_ranking"])  # type: ignore[arg-type]
    )
    row["tie_groups"] = []
    row["ranking_digest"] = "sha256:foreign-ranking"

    _evidence, findings = verification._stored_quantized_evidence(row, _row_ids())
    gate = verification._gate("structural_identity", findings)
    report = {
        "gates": [
            verification._gate(
                "semantic_outcome",
                [verification._finding("semantic_status_mismatch", "semantic")],
            ),
            gate,
        ]
    }

    assert [finding["code"] for finding in findings] == [
        "quantized_score_vector_mismatch",
        "quantized_score_vector_mismatch",
        "raw_diagnostic_digest_mismatch",
        "ranking_reconstruction_mismatch",
        "tie_group_reconstruction_mismatch",
        "ranking_reconstruction_mismatch",
    ]
    assert [list(finding) for finding in findings] == [
        ["code", "message", "frame_id"]
    ] * 6
    assert verification._primary_failure(report) == {
        "code": "quantized_score_vector_mismatch",
        "gate": "structural_identity",
        "finding": findings[0],
    }


def test_malformed_evidence_boundaries_and_unrelated_exceptions_escape(
    complete_row_fixture: tuple[
        dict[str, object], CompleteRowEvidence, SemanticTopSetOutcome
    ],
) -> None:
    non_finite = deepcopy(complete_row_fixture[0])
    non_finite["all_112_raw_scores"][0] = float("nan")  # type: ignore[index]
    evidence, findings = verification._stored_quantized_evidence(
        non_finite, _row_ids()
    )
    assert evidence is None
    assert [list(finding) for finding in findings] == [
        ["code", "message", "frame_id"]
    ]
    assert findings[0]["code"] == "raw_diagnostic_digest_mismatch"

    missing_identity = deepcopy(complete_row_fixture[0])
    del missing_identity["policy_artifact_id"]
    evidence, findings = verification._stored_quantized_evidence(
        missing_identity, _row_ids()
    )
    assert evidence is None
    assert list(findings[0]) == ["code", "message", "frame_id", "error"]

    class _UnexpectedScore:
        def __float__(self) -> float:
            raise RuntimeError("unexpected score conversion")

    unexpected = deepcopy(complete_row_fixture[0])
    unexpected["all_112_raw_scores"][0] = _UnexpectedScore()  # type: ignore[index]
    with pytest.raises(RuntimeError, match="unexpected score conversion"):
        verification._stored_quantized_evidence(unexpected, _row_ids())


def test_action_unanimous_and_conflicting_ties_are_reconstructed() -> None:
    unanimous_row, unanimous_evidence, unanimous = _complete_evidence_row(
        tied_top=True,
        unanimous_action=True,
    )
    conflicting_row, conflicting_evidence, conflicting = _complete_evidence_row(
        tied_top=True,
    )
    unanimous_actions = _row_actions()
    unanimous_actions["row-001"] = "LEFT"

    rebuilt_unanimous, unanimous_findings = verification._expected_semantic_for_row(
        unanimous_row,
        unanimous_actions,
        _row_ids(),
    )
    rebuilt_conflicting, conflicting_findings = verification._expected_semantic_for_row(
        conflicting_row,
        _row_actions(),
        _row_ids(),
    )

    assert isinstance(unanimous_evidence.ranking, CompleteRanking)
    assert isinstance(conflicting_evidence.ranking.tie_groups[0], TieGroup)
    assert unanimous_findings == []
    assert conflicting_findings == []
    assert rebuilt_unanimous == unanimous
    assert rebuilt_conflicting == conflicting
    assert unanimous.status == "action_unanimous_tie"
    assert unanimous.semantic_outcome_digest == (
        "sha256:20e4869824a9e3d9d7fc86dcb46004a7eba551e8409f2de8da66ebd6a2d830e7"
    )
    assert conflicting.status == "conflicting_action_tie"
    assert conflicting.semantic_outcome_digest == (
        "sha256:8f2e5b7399f1e735e2cc6fade8888c63a5d3b0f76084750bea332fcc347e66dc"
    )


def test_malformed_reference_evidence_is_read_only() -> None:
    row = {
        "frame_id": "frame:1",
        "all_112_row_ids": ["row:1"],
        "all_112_raw_scores": [0.5],
        "all_112_quantized_scores": [500_000],
        "metadata": {"sentinel": {"nested": [1, 2, 3]}},
    }
    before = deepcopy(row)

    evidence, findings = verification._stored_quantized_evidence(
        row,
        [f"row:{index}" for index in range(112)],
    )

    assert evidence is None
    assert findings[0]["code"] == "score_row_universe_mismatch"
    assert row == before


def test_read_only_payload_detects_nested_snapshot_change() -> None:
    first = {"verification_digest": "sha256:one", "nested": {"value": 1}}
    second = deepcopy(first)
    stable = {"file.json": {"nested": {"value": 1}}}
    changed = {"file.json": {"nested": {"value": 2}}}

    passed = verification.build_read_only_verification_payload(
        before=stable,
        middle=deepcopy(stable),
        after=deepcopy(stable),
        first=first,
        second=second,
    )
    failed = verification.build_read_only_verification_payload(
        before=stable,
        middle=changed,
        after=stable,
        first=first,
        second=second,
    )

    assert passed["status"] == "passed"
    assert failed["status"] == "failed"

    nondeterministic = verification.build_read_only_verification_payload(
        before=stable,
        middle=deepcopy(stable),
        after=deepcopy(stable),
        first=first,
        second={"verification_digest": "sha256:two", "nested": {"value": 1}},
    )
    assert nondeterministic["read_only"] is True
    assert nondeterministic["deterministic"] is False
    assert nondeterministic["status"] == "failed"
