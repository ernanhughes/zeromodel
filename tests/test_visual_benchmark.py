from __future__ import annotations

import json

import pytest

from zeromodel.artifact import VPMValidationError
from zeromodel.visual_benchmark import (
    BenchmarkSystemResult,
    GovernanceAuditResult,
    VisualBenchmarkMetrics,
    VisualBenchmarkReport,
    wilson_score_interval,
)
from zeromodel.visual_dataset import (
    CorruptionFamilySpec,
    VisualDatasetManifest,
    VisualExampleRecord,
    VISUAL_BENCHMARK_SYSTEMS,
)


def _manifest() -> VisualDatasetManifest:
    families = [
        CorruptionFamilySpec(family_id="canonical", kind="clean"),
        CorruptionFamilySpec(family_id="brightness", kind="brightness"),
        CorruptionFamilySpec(family_id="translation", kind="translation"),
        CorruptionFamilySpec(family_id="foreign", kind="ood"),
    ]
    records = []
    for split, family in (
        ("prototype", "canonical"),
        ("calibration", "brightness"),
        ("test", "translation"),
    ):
        for row_id, action_id in (("left", "A"), ("right", "B")):
            records.append(
                VisualExampleRecord(
                    observation_id=f"{split}:{row_id}",
                    observation_digest=f"sha256:{split}:{row_id}",
                    split=split,
                    family_id=family,
                    row_id=row_id,
                    action_id=action_id,
                )
            )
    records.append(
        VisualExampleRecord(
            observation_id="ood:foreign",
            observation_digest="sha256:ood",
            split="ood",
            family_id="foreign",
        )
    )
    return VisualDatasetManifest(
        source_scope="fixture:arcade",
        policy_artifact_id="policy-1",
        families=families,
        records=records,
    )


def test_manifest_enforces_family_holdout_and_round_trips() -> None:
    manifest = _manifest()
    loaded = VisualDatasetManifest.from_dict(manifest.to_dict())

    assert loaded.digest == manifest.digest
    assert loaded.enforce_family_holdout
    assert json.loads(json.dumps(loaded.to_dict())) == loaded.to_dict()


def test_manifest_rejects_family_leakage() -> None:
    manifest = _manifest()
    records = list(manifest.records)
    records.append(
        VisualExampleRecord(
            observation_id="test:left:leak",
            observation_digest="sha256:leak",
            split="test",
            family_id="canonical",
            row_id="left",
            action_id="A",
        )
    )

    with pytest.raises(VPMValidationError, match="held out by split"):
        VisualDatasetManifest(
            source_scope=manifest.source_scope,
            policy_artifact_id=manifest.policy_artifact_id,
            families=manifest.families,
            records=records,
        )


def test_manifest_requires_equal_row_coverage_across_core_splits() -> None:
    manifest = _manifest()
    records = [
        record
        for record in manifest.records
        if not (record.split == "test" and record.row_id == "right")
    ]

    with pytest.raises(VPMValidationError, match="identical rows"):
        VisualDatasetManifest(
            source_scope=manifest.source_scope,
            policy_artifact_id=manifest.policy_artifact_id,
            families=manifest.families,
            records=records,
        )


def _metrics() -> VisualBenchmarkMetrics:
    return VisualBenchmarkMetrics(
        evaluation_count=100,
        accepted_count=77,
        rejected_count=23,
        correct_row_count=75,
        correct_action_count=76,
        conflicting_action_error_count=0,
        false_accept_count=1,
        false_accept_opportunities=20,
        false_reject_count=4,
        false_reject_opportunities=80,
        top1_correct_row_count=76,
        top1_correct_action_count=79,
    )


def test_benchmark_report_records_explicit_benign_and_top1_metrics() -> None:
    metrics = _metrics()
    system = BenchmarkSystemResult(
        system_id="C",
        system_name=VISUAL_BENCHMARK_SYSTEMS["C"],
        contract_digest="contract-c",
        metrics=metrics,
    )
    governance = GovernanceAuditResult(
        system_id="C",
        question_id="replay-decision",
        answered=True,
        fidelity_score=1.0,
        effort_minutes=3.5,
        evidence={"trace_id": "trace-1"},
    )
    report = VisualBenchmarkReport(
        dataset_manifest_digest=_manifest().digest,
        systems=[system],
        governance_audit=[governance],
        declared_false_acceptance_target=0.01,
        validation_status="research",
    )

    assert metrics.action_accuracy == pytest.approx(0.76)
    assert metrics.benign_action_accuracy == pytest.approx(76 / 80)
    assert metrics.top1_benign_row_accuracy == pytest.approx(76 / 80)
    assert metrics.accepted_benign_count == 76
    assert metrics.accepted_benign_action_correctness == pytest.approx(1.0)
    assert metrics.false_acceptance_rate == pytest.approx(0.05)
    assert not report.deployment_permitted
    assert report.digest
    payload = report.to_dict()
    assert payload["systems"][0]["metrics"]["confidence_intervals_95"]
    assert json.loads(json.dumps(payload)) == payload


def test_wilson_interval_is_bounded_and_handles_empty_denominator() -> None:
    assert wilson_score_interval(0, 0) == (0.0, 0.0)
    lower, upper = wilson_score_interval(5, 10)
    assert 0.0 <= lower < 0.5 < upper <= 1.0

    with pytest.raises(VPMValidationError, match="inconsistent"):
        wilson_score_interval(11, 10)


def test_benchmark_metrics_reject_inconsistent_counts() -> None:
    metrics = VisualBenchmarkMetrics(
        evaluation_count=10,
        accepted_count=8,
        rejected_count=3,
        correct_row_count=7,
        correct_action_count=7,
        conflicting_action_error_count=0,
        false_accept_count=0,
        false_accept_opportunities=2,
        false_reject_count=0,
        false_reject_opportunities=8,
        top1_correct_row_count=7,
        top1_correct_action_count=7,
    )

    with pytest.raises(VPMValidationError, match="must equal"):
        metrics.validate()


def test_benchmark_slots_include_mandatory_conventional_baselines() -> None:
    assert set(VISUAL_BENCHMARK_SYSTEMS) == {"A", "B", "C", "D", "G", "H"}
    assert VISUAL_BENCHMARK_SYSTEMS["H"] == "governance_parity_wrapper"
