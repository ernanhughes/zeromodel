from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "docs" / "results" / "video-action-set-reachability-benchmark-v1"


def test_claim_quarantine_status_files() -> None:
    status_path = RESULTS / "STATUS.md"
    readme_path = RESULTS / "README.md"
    invalidated_path = RESULTS / "invalidated-artifacts-v1.json"
    claim_path = RESULTS / "claim-status-v1.json"
    withdrawn_path = (
        REPO_ROOT
        / "docs"
        / "research"
        / "video-action-set-reachability-withdrawn-claims-v1.md"
    )

    assert status_path.exists()
    status_text = status_path.read_text(encoding="utf-8")
    assert "reference_instrument_invalid" in status_text
    assert "reference_instrument_correctness_unresolved" in status_text
    assert "prospective_materialization_prohibited" in status_text
    assert "package_identity_foundations_correct" in status_text
    assert "evidence_schema_v2_defined" in status_text

    withdrawn_text = withdrawn_path.read_text(encoding="utf-8")
    assert "quarantine base main SHA:" in withdrawn_text
    assert "db9c99041e3627aab0e1f0819245a17bd5702c55" in withdrawn_text
    assert "integration merge SHA:" in withdrawn_text
    assert withdrawn_text.count("db9c99041e3627aab0e1f0819245a17bd5702c55") == 2
    assert "runtime amendment blob SHA:" in withdrawn_text
    assert "1055beecdf324cd7fbeafde152c714712913ac15" in withdrawn_text
    assert "phase-access schema:" in withdrawn_text
    assert "zeromodel-video-prospective-phase-access/v1" in withdrawn_text

    invalidated = json.loads(invalidated_path.read_text(encoding="utf-8"))
    assert len(invalidated["artifacts"]) == 6
    invalidated_paths = {row["path"] for row in invalidated["artifacts"]}
    for artifact in (
        "docs/results/video-action-set-reachability-benchmark-v1/runtime-comparison.json",
        "docs/results/video-action-set-reachability-benchmark-v1/runtime-profile-optimized.json",
        "docs/results/video-action-set-reachability-benchmark-v1/runtime-profile-optimized.md",
        "docs/results/video-action-set-reachability-benchmark-v1/provider-runtime-equivalence.json",
        "docs/results/video-action-set-reachability-benchmark-v1/provider-runtime-equivalence.csv",
        "docs/results/video-action-set-reachability-benchmark-v1/phase-access-audits.json",
    ):
        assert artifact in invalidated_paths

    inspected_absent_paths = {
        row["path"] for row in invalidated["inspected_absent_artifacts"]
    }
    expected_absent_paths = {
        "docs/results/video-action-set-reachability-benchmark-v1/provider-equivalence-results.json",
        "docs/results/video-action-set-reachability-benchmark-v1/tie-safety-results.json",
        "docs/results/video-action-set-reachability-benchmark-v1/instrument-verification.json",
    }
    assert len(invalidated["inspected_absent_artifacts"]) == 3
    assert inspected_absent_paths == expected_absent_paths
    assert invalidated_paths.isdisjoint(inspected_absent_paths)
    claim_data = json.loads(claim_path.read_text(encoding="utf-8"))
    claims = {row["claim"]: row for row in claim_data["claims"]}
    assert claims["runtime equivalence verified"]["status"] == "withdrawn"
    assert claims["runtime speedup measured"]["status"] == "invalid_measurement"
    assert claims["prospective instrument complete"]["status"] == "unsupported"
    assert claims["B3 canonical self-retrieval 112/112"]["status"] == "supported_within_narrow_scope"
    assert claims["historical System B row/action gap"]["status"] == "supported_within_narrow_scope"
    assert claims["historical R1 row/action gap"]["status"] == "supported_within_narrow_scope"

    readme_text = readme_path.read_text(encoding="utf-8")
    assert readme_text.startswith("# INVALID PROSPECTIVE INSTRUMENT")
