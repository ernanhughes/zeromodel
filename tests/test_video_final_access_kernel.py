from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_video_episode_plan_rmdto import plan_dto, sample_identity
from test_video_observation_rmdto import _pixels, sample_record
from zeromodel import build_runtime
from zeromodel.artifact import VPMValidationError
from zeromodel.db.runtime import build_sqlite_runtime
from zeromodel.domains.video_action_set.final_access_dto import (
    FINAL_EVALUATION_PROTOCOL_VERSION,
    FINAL_EXECUTION_AUTHORIZATION_VERSION,
    FINAL_EXECUTION_REQUEST_VERSION,
    FinalEvaluationProtocolDTO,
    FinalExecutionAuthorizationDTO,
    FinalExecutionReceiptDTO,
    FinalExecutionRequestDTO,
    event_chain_digest,
)
from zeromodel.domains.video_action_set.final_claims import build_final_claim_registry
from zeromodel.domains.video_action_set.final_evaluation import evaluate_final_protocol
from zeromodel.domains.video_action_set.final_reconstruction import (
    reconstruct_final_access_ledger,
)
from zeromodel.domains.video_action_set.final_reconciler import (
    interrupt_abandoned_running_access,
)
from zeromodel.domains.video_action_set.final_reporting import generate_final_report
from zeromodel.domains.video_action_set.observation_dto import (
    MaterializedObservationDTO,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def _authorization(
    *,
    authorization_id: str = "auth-1",
    seed_digest: str | None = None,
    sealed_plan_digest: str = _digest("b"),
    protocol_digest: str = _digest("c"),
    output_dir: str = "out",
    database_path: str = "final.sqlite3",
    unattended_permitted: bool = False,
) -> FinalExecutionAuthorizationDTO:
    return FinalExecutionAuthorizationDTO.create(
        {
            "version": FINAL_EXECUTION_AUTHORIZATION_VERSION,
            "authorization_id": authorization_id,
            "authorization_status": "authorized",
            "created_utc": "2026-07-21T00:00:00Z",
            "created_by": "reviewer",
            "protocol_digest": protocol_digest,
            "expected_benchmark_seed_digest": seed_digest or _digest("a"),
            "expected_sealed_plan_digest": sealed_plan_digest,
            "expected_policy_artifact_id": "policy",
            "output_dir": output_dir,
            "database_path": database_path,
            "unattended_permitted": unattended_permitted,
            "operator_confirmation_text": "CONFIRM FINAL ACCESS",
            "authorization_payload": {"scope": "synthetic-test"},
        }
    )


def _approved_protocol(
    *,
    status: str = "approved",
    seed_digest: str = _digest("a"),
    sealed_plan_digest: str = _digest("b"),
) -> FinalEvaluationProtocolDTO:
    return FinalEvaluationProtocolDTO.create(
        {
            "version": FINAL_EVALUATION_PROTOCOL_VERSION,
            "protocol_id": "synthetic-protocol",
            "protocol_status": status,
            "created_utc": "2026-07-21T00:00:00Z",
            "approved_utc": "2026-07-21T00:01:00Z"
            if status == "approved"
            else None,
            "approved_by": "reviewer" if status == "approved" else None,
            "benchmark_seed_digest": seed_digest,
            "sealed_plan_digest": sealed_plan_digest,
            "policy_artifact_id": "policy",
            "candidate_set_id": "candidate-set",
            "selected_provider_id": "P1",
            "decision_rule": {
                "kind": "fixed_metric_threshold",
                "aggregate": "mean",
                "metric_id": "score",
                "operator": "gte",
                "threshold": 0.75,
            },
            "required_evidence": {"provider_id": "P1", "expected_row_count": 2},
            "claim_rule_set": {"claims": [{"claim_id": "synthetic-claim"}]},
            "review_notes": {"scope": "synthetic-test"},
        }
    )


def _request(
    authorization: FinalExecutionAuthorizationDTO,
    authorization_file: Path,
) -> FinalExecutionRequestDTO:
    return FinalExecutionRequestDTO.create(
        {
            "version": FINAL_EXECUTION_REQUEST_VERSION,
            "output_dir": authorization.output_dir,
            "authorization_file": str(authorization_file),
            "expected_authorization_digest": authorization.authorization_digest,
            "expected_sealed_plan_digest": authorization.expected_sealed_plan_digest,
            "database_path": authorization.database_path,
            "preflight_only": True,
            "operator_identity": "tester",
            "unattended": False,
            "request_payload": {"scope": "synthetic-test"},
        }
    )


def test_final_authorization_state_machine_and_digest_chain() -> None:
    runtime = build_runtime()
    auth = _authorization()

    record = runtime.video_action_set.create_final_authorization(auth)
    assert record.state == "authorized"
    events = runtime.video_action_set.list_final_access_events(record.access_id)
    assert len(events) == 1
    assert events[0].previous_state is None
    assert events[0].new_state == "authorized"

    reserved = runtime.video_action_set.reserve_final_access(record.access_id)
    running = runtime.video_action_set.mark_final_access_running(record.access_id)
    events = runtime.video_action_set.list_final_access_events(record.access_id)

    assert reserved.state == "reserved"
    assert running.state == "running"
    assert [event.ordinal for event in events] == [0, 1, 2]
    assert events[2].previous_event_digest == events[1].event_digest
    assert event_chain_digest(events).startswith("sha256:")


def test_sqlite_final_authorization_enforces_unique_seed_sealed_pair() -> None:
    runtime = build_sqlite_runtime("sqlite:///:memory:", initialize_schema=True)
    auth = _authorization()
    runtime.video_action_set.create_final_authorization(auth)
    second = _authorization(authorization_id="auth-2")

    with pytest.raises(VPMValidationError, match="final access record conflict"):
        runtime.video_action_set.create_final_authorization(second)


def test_authorized_final_observation_uses_new_service_path_only() -> None:
    identity = sample_identity()
    plan = plan_dto(identity=identity, split="final", frame_count=1)
    runtime = build_runtime()
    runtime.video_action_set.save_identity(identity)
    runtime.video_action_set.save_episode_plan(plan)
    auth = _authorization(seed_digest=identity.seed_digest)
    access = runtime.video_action_set.create_final_authorization(auth)
    record = sample_record(split="final", plan=plan, pixels=_pixels())

    with pytest.raises(VPMValidationError, match="final split observation"):
        MaterializedObservationDTO.from_record(record)
    with pytest.raises(VPMValidationError, match="final split observation"):
        runtime.video_action_set.save_observation_records((record,))
    with pytest.raises(VPMValidationError, match="final access state"):
        runtime.video_action_set.save_final_observation_record(access.access_id, record)

    runtime.video_action_set.reserve_final_access(access.access_id)
    running = runtime.video_action_set.mark_final_access_running(access.access_id)
    saved = runtime.video_action_set.save_final_observation_record(
        running.access_id,
        record,
    )

    assert saved.split == "final"
    assert saved.final_access_id == running.access_id


def test_final_evaluator_requires_approved_protocol_and_complete_final_rows() -> None:
    protocol = _approved_protocol()
    rows = (
        {"split": "final", "provider_id": "P1", "metrics": {"score": 0.8}},
        {"split": "final", "provider_id": "P1", "metrics": {"score": 0.9}},
    )
    result = evaluate_final_protocol(
        protocol,
        rows,
        benchmark_seed_digest=protocol.benchmark_seed_digest,
        sealed_plan_digest=protocol.sealed_plan_digest,
    )

    assert result["decision"] == "passed"
    with pytest.raises(VPMValidationError, match="not approved"):
        evaluate_final_protocol(
            _approved_protocol(status="draft"),
            rows,
            benchmark_seed_digest=protocol.benchmark_seed_digest,
            sealed_plan_digest=protocol.sealed_plan_digest,
        )
    with pytest.raises(VPMValidationError, match="evidence split"):
        evaluate_final_protocol(
            protocol,
            ({"split": "selection", "provider_id": "P1", "metrics": {"score": 1}},),
            benchmark_seed_digest=protocol.benchmark_seed_digest,
            sealed_plan_digest=protocol.sealed_plan_digest,
        )
    incomplete = evaluate_final_protocol(
        protocol,
        rows[:1],
        benchmark_seed_digest=protocol.benchmark_seed_digest,
        sealed_plan_digest=protocol.sealed_plan_digest,
    )
    assert incomplete["decision"] == "indeterminate"


def test_final_cli_preflight_is_read_only(tmp_path: Path) -> None:
    auth = _authorization(
        output_dir=str(tmp_path / "out"),
        database_path=str(tmp_path / "final.sqlite3"),
        unattended_permitted=True,
    )
    auth_file = tmp_path / "authorization.json"
    auth_file.write_text(json.dumps(auth.to_dict()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "zeromodel.video_action_set_final_cli",
            "--output-dir",
            auth.output_dir,
            "--authorization-file",
            str(auth_file),
            "--expected-authorization-digest",
            auth.authorization_digest,
            "--expected-sealed-plan-digest",
            auth.expected_sealed_plan_digest,
            "--database-path",
            auth.database_path,
            "--preflight-only",
            "--non-interactive",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["preflight"] == "passed"
    assert payload["reservation_created"] is False
    assert not Path(auth.database_path).exists()


def test_final_cli_rejects_force_like_options(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "zeromodel.video_action_set_final_cli",
            "--output-dir",
            str(tmp_path),
            "--authorization-file",
            str(tmp_path / "missing.json"),
            "--expected-authorization-digest",
            _digest("a"),
            "--expected-sealed-plan-digest",
            _digest("b"),
            "--database-path",
            str(tmp_path / "final.sqlite3"),
            "--force",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unrecognized arguments: --force" in result.stderr


def test_reconstruction_claims_and_report_require_valid_receipt() -> None:
    runtime = build_runtime()
    auth = _authorization()
    record = runtime.video_action_set.create_final_authorization(auth)
    runtime.video_action_set.reserve_final_access(record.access_id)
    running = runtime.video_action_set.mark_final_access_running(record.access_id)
    service = runtime.video_action_set.engine.final_access_service
    completed = service.complete(
        running.access_id,
        evidence_digest=_digest("e"),
        measurements={"score": 1.0},
    )
    reconstruction = reconstruct_final_access_ledger(
        service.store,
        completed.access_id,
    )
    receipt = completed.record_payload.to_value()["receipt"]
    registry = build_final_claim_registry(
        receipt=FinalExecutionReceiptDTO.from_dict(receipt),
        evaluation_result={
            "decision": "passed",
            "evaluation_digest": _digest("f"),
        },
        claim_rules={"claims": [{"claim_id": "synthetic-claim"}]},
    )
    report = generate_final_report(
        receipt=FinalExecutionReceiptDTO.from_dict(receipt),
        evaluation_result={"decision": "passed"},
        claim_registry=registry,
    )

    assert reconstruction["counters"]["completion_count"] == 1
    assert registry["claims"][0]["status"] == "eligible"
    assert completed.access_id in report


def test_reconciler_interrupts_running_access_only_after_process_is_gone() -> None:
    runtime = build_runtime()
    auth = _authorization()
    record = runtime.video_action_set.create_final_authorization(auth)
    runtime.video_action_set.reserve_final_access(record.access_id)
    running = runtime.video_action_set.mark_final_access_running(record.access_id)
    store = runtime.video_action_set.engine.final_access_service.store

    assert (
        interrupt_abandoned_running_access(
            store,
            running.access_id,
            process_is_alive=lambda _identity: True,
            reconciler_identity="reconciler",
        )
        is None
    )
    interrupted = interrupt_abandoned_running_access(
        store,
        running.access_id,
        process_is_alive=lambda _identity: False,
        reconciler_identity="reconciler",
    )

    assert interrupted is not None
    assert interrupted.state == "interrupted"
