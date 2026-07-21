from __future__ import annotations

from pathlib import Path

import pytest

from video_final_test_support import (
    SyntheticFinalExecutor,
    approved_protocol,
    authorization,
    request,
)
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.domains.video_action_set.final_access_dto import (
    access_id_for_authorization,
)
from zeromodel.domains.video_action_set.final_access_service import FinalAccessService
from zeromodel.domains.video_action_set.final_publication import (
    FINAL_RECEIPT_NAME,
    load_published_receipt,
    validate_final_artifact_filename,
)
from zeromodel.domains.video_action_set.final_reconstruction import (
    reconstruct_final_access_ledger,
)
from zeromodel.stores.video_action_set_memory import InMemoryVideoActionSetStore


class InjectedFailure(RuntimeError):
    pass


@pytest.mark.parametrize(
    "name",
    [
        "con",
        "CON",
        "Con.json",
        "nul",
        "PRN.txt",
        "aux",
        "com1",
        "COM9.json",
        "lpt1",
        "LPT9.log",
        "CONIN$",
        "CONOUT$",
    ],
)
def test_windows_reserved_artifact_names_are_rejected(name: str) -> None:
    with pytest.raises(VPMValidationError, match="filename"):
        validate_final_artifact_filename(name)


@pytest.mark.parametrize(
    "name",
    [
        "console.json",
        "connection.json",
        "com10.json",
        "lpt10.json",
        "auxiliary.json",
    ],
)
def test_similar_nonreserved_artifact_names_are_allowed(name: str) -> None:
    assert validate_final_artifact_filename(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "path/name.json",
        "path\\name.json",
        "C:\\name.json",
        "\\\\server\\share\\name.json",
        "name.json:stream",
        "name.json\n",
        "name.json.",
        "name.json ",
    ],
)
def test_invalid_windows_artifact_path_forms_remain_rejected(name: str) -> None:
    with pytest.raises(VPMValidationError, match="filename"):
        validate_final_artifact_filename(name)


@pytest.mark.parametrize(
    (
        "boundary",
        "expected_state",
        "canonical_present",
        "expected_publication_status",
    ),
    [
        ("before_promotion", "failed", False, "failed"),
        ("during_promotion", "failed", False, "failed"),
        ("after_promotion", "failed", True, "failed"),
        ("during_completed_transition", "failed", True, "failed"),
        ("before_receipt", "completed", True, "completed_receipt_missing"),
        (
            "during_receipt_write",
            "completed",
            True,
            "completed_receipt_missing",
        ),
    ],
)
def test_receipt_last_failure_boundaries(
    tmp_path: Path,
    boundary: str,
    expected_state: str,
    canonical_present: bool,
    expected_publication_status: str,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)

    def inject(name: str) -> None:
        if name == boundary:
            raise InjectedFailure(name)

    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
        failure_injector=inject,
    )
    execution_request = request(tmp_path, auth)

    with pytest.raises(InjectedFailure, match=boundary):
        service.execute_final_once(execution_request)

    access_id = access_id_for_authorization(auth)
    reconstruction = reconstruct_final_access_ledger(service.store, access_id)
    output_dir = Path(auth.output_dir)
    assert reconstruction["record"]["state"] == expected_state
    assert reconstruction["publication_status"] == expected_publication_status
    assert reconstruction["publishable_success"] is False
    assert output_dir.exists() is canonical_present
    assert load_published_receipt(output_dir) is None
    assert not (output_dir / FINAL_RECEIPT_NAME).exists()
    with pytest.raises(VPMValidationError, match="terminal state|transition"):
        service.execute_final_once(execution_request)


@pytest.mark.parametrize("terminal_state", ["running", "failed", "interrupted"])
def test_noncompleted_states_without_receipts_are_not_publishable(
    tmp_path: Path,
    terminal_state: str,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    service = FinalAccessService(store=InMemoryVideoActionSetStore())
    record = service.create_authorization(auth, protocol)
    record = service.reserve(record.access_id)
    record = service.mark_running(record.access_id)
    if terminal_state == "failed":
        record = service.fail(
            record.access_id,
            failure_kind="synthetic",
            error_code="synthetic",
            error_message="synthetic",
        )
    elif terminal_state == "interrupted":
        record = service.interrupt(
            record.access_id,
            failure_kind="synthetic",
            error_code="synthetic",
            error_message="synthetic",
        )
    reconstruction = reconstruct_final_access_ledger(service.store, record.access_id)
    assert reconstruction["publication_status"] == terminal_state
    assert reconstruction["publication"]["receipt_status"] == "absent"
    assert reconstruction["publishable_success"] is False


def test_completed_valid_and_invalid_receipts_have_distinct_statuses(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
    )
    service.execute_final_once(request(tmp_path, auth))
    access_id = access_id_for_authorization(auth)
    valid = reconstruct_final_access_ledger(service.store, access_id)
    assert valid["publication_status"] == "completed_receipt_valid"
    assert valid["publishable_success"] is True

    receipt_path = Path(auth.output_dir) / FINAL_RECEIPT_NAME
    receipt_path.write_bytes(b'{"tampered":true}')
    invalid = reconstruct_final_access_ledger(service.store, access_id)
    assert invalid["publication_status"] == "completed_receipt_invalid"
    assert invalid["publishable_success"] is False


def test_staging_tamper_between_validation_and_promotion_is_rejected(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)

    def tamper(name: str) -> None:
        if name != "during_promotion":
            return
        candidates = tuple(tmp_path.glob(".*.final-staging-*"))
        assert len(candidates) == 1
        (candidates[0] / "final-summary.json").write_bytes(b"tampered")

    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
        failure_injector=tamper,
    )
    with pytest.raises(VPMValidationError, match="manifest validation"):
        service.execute_final_once(request(tmp_path, auth))
    assert not Path(auth.output_dir).exists()
    assert load_published_receipt(Path(auth.output_dir)) is None


def test_authorized_episode_and_frame_counts_must_match_before_staging(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    duplicate_identity_rows = (
        {
            "family_id": "family-1",
            "episode_id": "episode-1",
            "frame_ordinal": 0,
            "frame_id": "frame-1",
            "provider_id": "P1",
            "split": "final",
            "metrics": {"score": "0.8"},
        },
        {
            "family_id": "family-1",
            "episode_id": "episode-1",
            "frame_ordinal": 0,
            "frame_id": "frame-1",
            "provider_id": "P1",
            "split": "final",
            "metrics": {"score": "0.9"},
        },
    )
    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(rows=duplicate_identity_rows),
    )
    with pytest.raises(VPMValidationError, match="authorized counts"):
        service.execute_final_once(request(tmp_path, auth))
    assert not Path(auth.output_dir).exists()


def test_authorized_episode_identities_must_match_before_staging(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    wrong_rows = (
        {
            "family_id": "family-1",
            "episode_id": "other-1",
            "frame_ordinal": 0,
            "frame_id": "frame-1",
            "provider_id": "P1",
            "split": "final",
            "metrics": {"score": "0.8"},
        },
        {
            "family_id": "family-2",
            "episode_id": "other-2",
            "frame_ordinal": 0,
            "frame_id": "frame-2",
            "provider_id": "P1",
            "split": "final",
            "metrics": {"score": "0.9"},
        },
    )
    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(rows=wrong_rows),
    )
    with pytest.raises(VPMValidationError, match="authorized identities"):
        service.execute_final_once(request(tmp_path, auth))
    assert not Path(auth.output_dir).exists()


def test_preexisting_staging_and_canonical_paths_fail_closed(tmp_path: Path) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    token = canonical_sha256(auth.authorization_id)[7:23]
    staging = tmp_path / f".{Path(auth.output_dir).name}.final-staging-{token}"
    staging.mkdir()
    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
    )
    with pytest.raises(VPMValidationError, match="staging directory already exists"):
        service.execute_final_once(request(tmp_path, auth))
    assert load_published_receipt(Path(auth.output_dir)) is None

    other_root = tmp_path / "other"
    other_root.mkdir()
    other_protocol = approved_protocol(protocol_id="other-protocol")
    other_auth = authorization(
        other_root,
        other_protocol,
        authorization_id="auth-2",
    )
    Path(other_auth.output_dir).mkdir()
    other_service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
    )
    with pytest.raises(VPMValidationError, match="canonical final artifacts"):
        other_service.execute_final_once(request(other_root, other_auth))


@pytest.mark.parametrize(
    "artifacts",
    [
        {},
        {"final-summary.json": b"ok", "unexpected.json": b"no"},
    ],
)
def test_missing_or_unexpected_artifact_files_are_rejected(
    tmp_path: Path,
    artifacts: dict[str, bytes],
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    executor = SyntheticFinalExecutor(artifacts=artifacts)
    executor.artifacts = artifacts
    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=executor,
    )
    with pytest.raises(VPMValidationError, match="file set"):
        service.execute_final_once(request(tmp_path, auth))
    assert load_published_receipt(Path(auth.output_dir)) is None


def test_duplicate_artifact_names_and_path_traversal_ids_are_rejected(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    duplicate_auth = authorization(
        tmp_path,
        protocol,
        expected_artifacts=("same.json", "same.json"),
    )
    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
    )
    with pytest.raises(VPMValidationError, match="artifact list"):
        service.execute_final_once(request(tmp_path, duplicate_auth))
    with pytest.raises(VPMValidationError, match="authorization id"):
        authorization(tmp_path, protocol, authorization_id="../escape")


def test_symlinked_canonical_output_is_rejected_when_supported(tmp_path: Path) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    target = tmp_path / "target"
    target.mkdir()
    try:
        Path(auth.output_dir).symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not available")
    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
    )
    with pytest.raises(VPMValidationError, match="symlinked final path"):
        service.execute_final_once(request(tmp_path, auth))
