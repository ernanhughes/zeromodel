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

    assert status_path.exists()
    status_text = status_path.read_text(encoding="utf-8")
    assert "reference_instrument_invalid" in status_text
    assert "prospective_materialization_prohibited" in status_text

    invalidated = json.loads(invalidated_path.read_text(encoding="utf-8"))
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
