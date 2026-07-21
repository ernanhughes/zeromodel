from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from video_final_test_support import (
    approved_protocol,
    authorization,
    evidence_bundle,
    final_rows,
)
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set.final_access_dto import (
    FinalEvaluationProtocolDTO,
    FinalEvidenceBundleDTO,
)
from zeromodel.domains.video_action_set.final_evaluation import (
    evaluate_final_protocol,
)
from zeromodel.domains.video_action_set.final_historical_authority import (
    verify_historical_authority,
)


pytestmark = pytest.mark.integration


def _historical(tmp_path: Path) -> dict[str, object]:
    auth = authorization(tmp_path, approved_protocol())
    contract = auth.authorization_payload.to_value()
    assert isinstance(contract, Mapping)
    historical = contract["historical_authority"]
    assert isinstance(historical, Mapping)
    return dict(historical)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("historical_authority_id", "other-authority", "manifest binding"),
        ("stage8_commit", "other-commit", "manifest binding"),
    ],
)
def test_historical_manifest_bindings_reject_declared_mismatch(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    historical = _historical(tmp_path)
    historical[field] = replacement
    with pytest.raises(VPMValidationError, match=message):
        verify_historical_authority(historical)


def test_historical_authority_rejects_relative_paths(tmp_path: Path) -> None:
    historical = _historical(tmp_path)
    historical["historical_database_path"] = "relative-stage8.sqlite3"
    with pytest.raises(VPMValidationError, match="absolute"):
        verify_historical_authority(historical)


def test_duplicate_evidence_identity_is_rejected() -> None:
    protocol = approved_protocol()
    rows = final_rows()
    payload = evidence_bundle(protocol).to_dict()
    payload["rows"] = [rows[0], rows[0]]
    payload["actual_counts"] = {
        "evidence_row_count": 2,
        "episode_count": 1,
        "frame_count": 1,
        "provider_count": 1,
    }
    with pytest.raises(VPMValidationError, match="duplicate final evidence identity"):
        FinalEvidenceBundleDTO.from_dict(payload)


@pytest.mark.parametrize("provider_order", [("P2",), ("P1", "P2")])
def test_missing_or_undeclared_evidence_provider_is_rejected(
    provider_order: tuple[str, ...],
) -> None:
    protocol = approved_protocol()
    payload = evidence_bundle(protocol).to_dict()
    payload["provider_order"] = list(provider_order)
    payload["provider_versions"] = {
        provider_id: "synthetic-v1" for provider_id in provider_order
    }
    payload.pop("actual_counts")
    payload.pop("evidence_digest")
    with pytest.raises(VPMValidationError, match="provider order"):
        FinalEvidenceBundleDTO.create(payload)


def test_unknown_protocol_keys_fail_before_evaluation() -> None:
    payload = approved_protocol().to_dict()
    payload["unexpected"] = True
    with pytest.raises(VPMValidationError, match="keys"):
        FinalEvaluationProtocolDTO.from_dict(payload)


def test_incomplete_required_evidence_is_indeterminate() -> None:
    protocol = approved_protocol(expected_row_count=3)
    evidence = evidence_bundle(
        protocol,
        expected_counts={
            "evidence_row_count": 3,
            "episode_count": 3,
            "frame_count": 3,
            "provider_count": 1,
        },
    )
    result = evaluate_final_protocol(protocol, evidence)
    assert result.decision == "indeterminate"
    assert result.indeterminate_reasons.to_value() == ["incomplete_final_evidence"]
