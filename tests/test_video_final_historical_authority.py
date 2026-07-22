from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from video_final_test_support import (
    SyntheticFinalExecutor,
    approved_protocol,
    authorization,
    digest,
    request,
)
from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.final_access_dto import (
    FinalExecutionAuthorizationDTO,
    FinalExecutionReceiptDTO,
    access_id_for_authorization,
)
from zeromodel.video.domains.video_action_set.final_access_service import FinalAccessService
from zeromodel.video.domains.video_action_set.final_historical_authority import (
    HistoricalEvidenceManifestDTO,
    VerifiedHistoricalAuthorityDTO,
    verify_historical_authority,
)
from zeromodel.video.domains.video_action_set.final_publication import FINAL_RECEIPT_NAME
from zeromodel.video.stores.video_action_set_memory import InMemoryVideoActionSetStore


def _historical(
    auth: FinalExecutionAuthorizationDTO,
) -> dict[str, object]:
    payload = auth.authorization_payload.to_value()
    assert isinstance(payload, Mapping)
    historical = payload["historical_authority"]
    assert isinstance(historical, Mapping)
    return dict(historical)


def _with_historical(
    auth: FinalExecutionAuthorizationDTO,
    historical: Mapping[str, object],
) -> FinalExecutionAuthorizationDTO:
    payload = auth.to_dict()
    contract = payload["authorization_payload"]
    assert isinstance(contract, dict)
    contract["historical_authority"] = dict(historical)
    payload.pop("authorization_digest")
    return FinalExecutionAuthorizationDTO.create(payload)


def test_historical_authority_recomputes_file_and_manifest_digests(
    tmp_path: Path,
) -> None:
    auth = authorization(tmp_path, approved_protocol())
    historical = _historical(auth)
    verified = verify_historical_authority(historical)

    assert (
        verified.historical_database_sha256 == historical["historical_database_sha256"]
    )
    assert verified.evidence_manifest_digest == historical["evidence_manifest_digest"]
    assert VerifiedHistoricalAuthorityDTO.from_dict(verified.to_dict()) == verified
    tampered = verified.to_dict()
    tampered["historical_database_sha256"] = digest("f")
    with pytest.raises(VPMValidationError, match="authority digest"):
        VerifiedHistoricalAuthorityDTO.from_dict(tampered)

    manifest_path = Path(str(historical["evidence_manifest_path"]))
    import json

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = HistoricalEvidenceManifestDTO.from_dict(manifest_payload)
    assert manifest.evidence_manifest_digest == verified.evidence_manifest_digest
    manifest_payload["extra"] = True
    with pytest.raises(VPMValidationError, match="keys"):
        HistoricalEvidenceManifestDTO.from_dict(manifest_payload)


def test_incorrect_declared_database_hash_fails_closed(tmp_path: Path) -> None:
    auth = authorization(tmp_path, approved_protocol())
    historical = _historical(auth)
    historical["historical_database_sha256"] = digest("f")
    with pytest.raises(VPMValidationError, match="database digest"):
        verify_historical_authority(historical)


@pytest.mark.parametrize("path_kind", ["missing", "directory"])
def test_historical_database_must_be_an_existing_regular_file(
    tmp_path: Path,
    path_kind: str,
) -> None:
    auth = authorization(tmp_path, approved_protocol())
    historical = _historical(auth)
    replacement = tmp_path / path_kind
    if path_kind == "directory":
        replacement.mkdir()
    historical["historical_database_path"] = str(replacement.resolve())
    with pytest.raises(VPMValidationError, match="database path"):
        verify_historical_authority(historical)


def test_symlinked_historical_file_fails_when_supported(tmp_path: Path) -> None:
    auth = authorization(tmp_path, approved_protocol())
    historical = _historical(auth)
    target = Path(str(historical["historical_database_path"]))
    link = tmp_path / "linked-stage8.sqlite3"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    historical["historical_database_path"] = str(link.absolute())
    with pytest.raises(VPMValidationError, match="database path"):
        verify_historical_authority(historical)


def test_symlinked_historical_parent_fails_when_supported(tmp_path: Path) -> None:
    target_root = tmp_path / "target"
    target_root.mkdir()
    auth = authorization(target_root, approved_protocol())
    historical = _historical(auth)
    link_root = tmp_path / "linked-parent"
    try:
        link_root.symlink_to(target_root, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    historical["historical_database_path"] = str(
        link_root / Path(str(historical["historical_database_path"])).name
    )
    historical["evidence_manifest_path"] = str(
        link_root / Path(str(historical["evidence_manifest_path"])).name
    )
    with pytest.raises(VPMValidationError, match="path"):
        verify_historical_authority(historical)


def test_authorization_manifest_digest_must_match_actual_manifest(
    tmp_path: Path,
) -> None:
    auth = authorization(tmp_path, approved_protocol())
    historical = _historical(auth)
    historical["evidence_manifest_digest"] = digest("f")
    with pytest.raises(VPMValidationError, match="manifest binding"):
        verify_historical_authority(historical)


def test_historical_file_change_after_authorization_fails_before_reservation(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
    )
    created = service.create_authorization(auth, protocol)
    historical = _historical(auth)
    Path(str(historical["historical_database_path"])).write_bytes(b"changed")

    with pytest.raises(VPMValidationError, match="database digest"):
        service.execute_final_once(request(tmp_path, auth))
    record = service.load_record(created.access_id)
    assert record is not None
    assert record.state == "authorized"
    assert len(service.list_events(created.access_id)) == 1


def test_historical_file_change_after_reservation_marks_execution_failed(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    historical = _historical(auth)
    database_path = Path(str(historical["historical_database_path"]))
    original = database_path.read_bytes()

    def mutate_at_reverification(boundary: str) -> None:
        if boundary == "before_historical_authority_reverification":
            database_path.write_bytes(b"changed after reservation")

    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
        failure_injector=mutate_at_reverification,
    )
    execution_request = request(tmp_path, auth)
    with pytest.raises(VPMValidationError, match="database digest"):
        service.execute_final_once(execution_request)

    access_id = access_id_for_authorization(auth)
    record = service.load_record(access_id)
    assert record is not None
    assert record.state == "failed"
    assert not (Path(auth.output_dir) / FINAL_RECEIPT_NAME).exists()

    database_path.write_bytes(original)
    with pytest.raises(VPMValidationError, match="terminal state|transition"):
        service.execute_final_once(execution_request)


def test_receipt_binds_only_computed_historical_values(tmp_path: Path) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    verified = verify_historical_authority(_historical(auth))
    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
    )
    receipt = FinalExecutionReceiptDTO.from_dict(
        service.execute_final_once(request(tmp_path, auth))
    )

    assert receipt.historical_authority_id == verified.historical_authority_id
    assert receipt.historical_database_sha256 == verified.historical_database_sha256
    assert (
        receipt.historical_evidence_manifest_digest == verified.evidence_manifest_digest
    )
    assert receipt.historical_authority_digest == verified.historical_authority_digest
    tampered_receipt = receipt.to_dict()
    tampered_receipt["historical_database_sha256"] = digest("f")
    with pytest.raises(VPMValidationError, match="receipt digest"):
        FinalExecutionReceiptDTO.from_dict(tampered_receipt)


def test_preflight_verifies_hash_without_database_or_reservation(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    service = FinalAccessService(store=InMemoryVideoActionSetStore())
    result = service.execute_final_once(request(tmp_path, auth, preflight_only=True))

    assert (
        result["historical_database_sha256"]
        == _historical(auth)["historical_database_sha256"]
    )
    assert result["reservation_created"] is False
    assert service.load_authorization(auth.authorization_id) is None
    assert not Path(auth.database_path).exists()
