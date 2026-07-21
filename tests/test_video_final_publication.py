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
)
from zeromodel.domains.video_action_set.final_reconstruction import (
    reconstruct_final_access_ledger,
)
from zeromodel.stores.video_action_set_memory import InMemoryVideoActionSetStore


class InjectedFailure(RuntimeError):
    pass


@pytest.mark.parametrize(
    ("boundary", "expected_state", "canonical_present"),
    [
        ("before_promotion", "failed", False),
        ("during_promotion", "failed", False),
        ("after_promotion", "failed", True),
        ("during_completed_transition", "failed", True),
        ("before_receipt", "completed", True),
        ("during_receipt_write", "completed", True),
    ],
)
def test_receipt_last_failure_boundaries(
    tmp_path: Path,
    boundary: str,
    expected_state: str,
    canonical_present: bool,
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
    assert output_dir.exists() is canonical_present
    assert load_published_receipt(output_dir) is None
    assert not (output_dir / FINAL_RECEIPT_NAME).exists()
    with pytest.raises(VPMValidationError, match="terminal state|transition"):
        service.execute_final_once(execution_request)


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
