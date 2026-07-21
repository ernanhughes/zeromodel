from __future__ import annotations

from pathlib import Path

import pytest

from test_video_episode_plan_rmdto import plan_dto, sample_identity
from test_video_observation_rmdto import (
    _pixels,
    assert_records_equivalent,
    sample_record,
)
from video_final_test_support import approved_protocol, authorization
from zeromodel.artifact import VPMValidationError
from zeromodel.db.runtime import build_finalization_sqlite_runtime
from zeromodel.domains.video_action_set.final_access_dto import (
    FinalEvaluationProtocolDTO,
    FinalExecutionAuthorizationDTO,
)
from zeromodel.domains.video_action_set.final_access_service import FinalAccessService
from zeromodel.domains.video_action_set.observation_dto import (
    MaterializedObservationDTO,
)


pytestmark = pytest.mark.integration


def _database_url(path: Path) -> str:
    return path.resolve().as_uri().replace("file:///", "sqlite:///")


def _protocol_for_identity(identity) -> FinalEvaluationProtocolDTO:
    payload = approved_protocol().to_dict()
    payload["benchmark_seed_digest"] = identity.seed_digest
    payload.pop("protocol_digest")
    return FinalEvaluationProtocolDTO.create(payload)


def _setup(
    tmp_path: Path,
    *,
    authorization_id: str = "auth-1",
) -> tuple[
    object,
    FinalAccessService,
    object,
    FinalEvaluationProtocolDTO,
    dict[str, object],
]:
    identity = sample_identity(f"synthetic-final-{authorization_id}")
    plan = plan_dto(identity=identity, split="final", frame_count=1)
    protocol = _protocol_for_identity(identity)
    auth = authorization(
        tmp_path,
        protocol,
        authorization_id=authorization_id,
    )
    runtime = build_finalization_sqlite_runtime(
        _database_url(Path(auth.database_path)),
        initialize_authority=True,
    )
    facade = runtime.video_action_set
    facade.save_identity(identity)
    facade.save_episode_plan(plan)
    record = sample_record(split="final", plan=plan, pixels=_pixels())
    return facade, facade.engine.final_access_service, auth, protocol, record


@pytest.mark.parametrize(
    ("state", "should_succeed"),
    [
        ("authorized", False),
        ("reserved", False),
        ("running", True),
        ("completed", False),
        ("failed", False),
        ("interrupted", False),
    ],
)
def test_final_observations_are_writable_only_while_access_is_running(
    tmp_path: Path,
    state: str,
    should_succeed: bool,
) -> None:
    _facade, service, auth, protocol, record = _setup(tmp_path)
    access = service.create_authorization(auth, protocol)
    if state in {"reserved", "running", "completed", "failed", "interrupted"}:
        access = service.reserve(access.access_id)
    if state in {"running", "completed", "failed", "interrupted"}:
        access = service.mark_running(access.access_id)
    if state == "completed":
        transition = service._next_event(
            access.access_id,
            "completed",
            process_identity="synthetic",
            utc="2026-07-21T02:00:00Z",
            event_payload={"kind": "completed"},
        )
        access = service.store.complete_final_access(*transition)
    elif state == "failed":
        access = service.fail(
            access.access_id,
            failure_kind="synthetic",
            error_code="synthetic",
            error_message="synthetic",
        )
    elif state == "interrupted":
        access = service.interrupt(
            access.access_id,
            failure_kind="synthetic",
            error_code="synthetic",
            error_message="synthetic",
        )

    if should_succeed:
        saved = service.save_final_observation_record(access.access_id, record)
        assert saved.final_access_id == access.access_id
    else:
        with pytest.raises(VPMValidationError, match="final access state"):
            service.save_final_observation_record(access.access_id, record)


def test_final_observation_without_access_uses_no_legacy_write_path(
    tmp_path: Path,
) -> None:
    facade, _service, _auth, _protocol, record = _setup(tmp_path)
    with pytest.raises(VPMValidationError, match="final split observation"):
        facade.save_observation_records((record,))


def test_final_observation_cannot_move_between_accesses(tmp_path: Path) -> None:
    facade, service, first_auth, protocol, record = _setup(tmp_path)
    first = service.create_authorization(first_auth, protocol)
    first = service.reserve(first.access_id)
    first = service.mark_running(first.access_id)

    second_identity = sample_identity("synthetic-final-second-owner")
    second_plan = plan_dto(identity=second_identity, split="final", frame_count=1)
    facade.save_identity(second_identity)
    facade.save_episode_plan(second_plan)
    second_protocol = _protocol_for_identity(second_identity)
    second_auth = authorization(
        tmp_path,
        second_protocol,
        authorization_id="auth-2",
    )
    second_payload = second_auth.to_dict()
    second_payload["database_path"] = first_auth.database_path
    second_payload.pop("authorization_digest")
    second_auth = FinalExecutionAuthorizationDTO.create(second_payload)
    second = service.create_authorization(second_auth, second_protocol)
    second = service.reserve(second.access_id)
    second = service.mark_running(second.access_id)

    saved = service.save_final_observation_record(first.access_id, record)
    assert saved.final_access_id == first.access_id
    with pytest.raises(VPMValidationError, match="final access|conflict"):
        service.save_final_observation_record(second.access_id, record)

    wrong_owner = MaterializedObservationDTO.from_authorized_final_record(
        record,
        final_access_id=first.access_id,
    )
    with pytest.raises(
        VPMValidationError, match="final (access|execution authorization)"
    ):
        service.store.save_authorized_final_observations(second, (wrong_owner,))
    persisted = service.store.get_observation(saved.frame_id)
    assert persisted is not None
    assert persisted.final_access_id == first.access_id


def test_nonfinal_observation_identity_and_nullable_access_remain_unchanged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nonfinal.sqlite3"
    runtime = build_finalization_sqlite_runtime(
        _database_url(path),
        initialize_authority=True,
    )
    identity = sample_identity("synthetic-development")
    plan = plan_dto(identity=identity, split="development", frame_count=1)
    record = sample_record(plan=plan, split="development", pixels=_pixels())
    expected = MaterializedObservationDTO.from_record(record)
    runtime.video_action_set.save_identity(identity)
    runtime.video_action_set.save_episode_plan(plan)
    saved = runtime.video_action_set.save_observation_record(record)
    loaded = runtime.video_action_set.get_observation_record(saved.frame_id)

    assert saved.final_access_id is None
    assert saved.frame_id == expected.observation.frame_id
    assert (
        saved.observation_pixel_digest == expected.observation.observation_pixel_digest
    )
    assert loaded is not None
    assert_records_equivalent(loaded, record)

