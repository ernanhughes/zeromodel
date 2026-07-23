from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_UP, getcontext, setcontext
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_video_episode_plan_rmdto import plan_dto, sample_identity
from test_video_observation_rmdto import _pixels, sample_record
from video_final_test_support import (
    SyntheticFinalExecutor,
    approved_protocol,
    authorization,
    evidence_bundle,
    final_rows,
    request,
)
from zeromodel.video.runtime import build_runtime
from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_json_bytes
from zeromodel.video.domains.video_action_set.final_access_dto import (
    FinalEvaluationResultDTO,
    FinalExecutionAuthorizationDTO,
    FinalExecutionReceiptDTO,
    event_chain_digest,
)
import zeromodel.video.domains.video_action_set.final_access_service as final_access_service_module
from zeromodel.video.domains.video_action_set.final_access_service import FinalAccessService
from zeromodel.video.domains.video_action_set.final_claims import build_final_claim_registry
from zeromodel.video.domains.video_action_set.final_evaluation import evaluate_final_protocol
from zeromodel.video.domains.video_action_set.final_publication import (
    FINAL_EVALUATION_NAME,
    FINAL_RECEIPT_NAME,
)
from zeromodel.video.domains.video_action_set.final_reconciler import (
    interrupt_abandoned_running_access,
)
from zeromodel.video.domains.video_action_set.final_reconstruction import (
    reconstruct_final_access_ledger,
)
from zeromodel.video.domains.video_action_set.final_reporting import generate_final_report
from zeromodel.video.domains.video_action_set.observation_dto import (
    MaterializedObservationDTO,
)
from zeromodel.video.stores.video_action_set_memory import InMemoryVideoActionSetStore


REPO_ROOT = Path(__file__).resolve().parents[1]


def _source_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        str(REPO_ROOT / path)
        for path in (
            "packages/core/src",
            "packages/observation/src",
            "packages/video/src",
            "packages/sqlalchemy/src",
        )
    )
    return env


def test_final_authorization_state_machine_and_digest_chain(tmp_path: Path) -> None:
    runtime = build_runtime()
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)

    record = runtime.video_action_set.create_final_authorization(auth, protocol)
    reserved = runtime.video_action_set.reserve_final_access(record.access_id)
    running = runtime.video_action_set.mark_final_access_running(record.access_id)
    events = runtime.video_action_set.list_final_access_events(record.access_id)

    assert record.state == "authorized"
    assert reserved.state == "reserved"
    assert running.state == "running"
    assert running.current_event_ordinal == 2
    assert [event.ordinal for event in events] == [0, 1, 2]
    assert events[2].previous_event_digest == events[1].event_digest
    assert event_chain_digest(events).startswith("sha256:")


def test_authorized_final_observation_uses_service_path_only(tmp_path: Path) -> None:
    identity = sample_identity()
    plan = plan_dto(identity=identity, split="final", frame_count=1)
    runtime = build_runtime()
    runtime.video_action_set.save_identity(identity)
    runtime.video_action_set.save_episode_plan(plan)
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    auth_payload = auth.to_dict()
    auth_payload["expected_benchmark_seed_digest"] = identity.seed_digest
    auth_payload.pop("authorization_digest")
    from zeromodel.video.domains.video_action_set.final_access_dto import (
        FinalExecutionAuthorizationDTO,
    )

    auth = FinalExecutionAuthorizationDTO.create(auth_payload)
    protocol_payload = protocol.to_dict()
    protocol_payload["benchmark_seed_digest"] = identity.seed_digest
    protocol_payload.pop("protocol_digest")
    from zeromodel.video.domains.video_action_set.final_access_dto import (
        FinalEvaluationProtocolDTO,
    )

    protocol = FinalEvaluationProtocolDTO.create(protocol_payload)
    auth_payload = auth.to_dict()
    auth_payload["protocol_digest"] = protocol.protocol_digest
    auth_payload.pop("authorization_digest")
    auth = FinalExecutionAuthorizationDTO.create(auth_payload)
    access = runtime.video_action_set.create_final_authorization(auth, protocol)
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


def test_evaluator_is_order_independent_and_digest_bearing() -> None:
    protocol = approved_protocol(threshold="0.85")
    rows = final_rows()
    first = evidence_bundle(protocol, rows=rows)
    second = evidence_bundle(protocol, rows=tuple(reversed(rows)))
    first_result = evaluate_final_protocol(protocol, first)
    second_result = evaluate_final_protocol(protocol, second)

    assert first.to_dict() == second.to_dict()
    assert first_result.to_dict() == second_result.to_dict()
    assert canonical_json_bytes(first_result.to_dict()) == canonical_json_bytes(
        second_result.to_dict()
    )
    assert first_result.decision == "passed"
    assert first_result.descriptive_measurements.to_value()["threshold_equality"] == (
        "inclusive"
    )
    tampered_evidence = first.to_dict()
    tampered_evidence["rows"][0]["metrics"]["score"] = "1.0"  # type: ignore[index]
    from zeromodel.video.domains.video_action_set.final_access_dto import (
        FinalEvidenceBundleDTO,
    )

    with pytest.raises(VPMValidationError, match="evidence digest"):
        FinalEvidenceBundleDTO.from_dict(tampered_evidence)
    assert FinalEvaluationResultDTO.from_dict(first_result.to_dict()) == first_result
    tampered = first_result.to_dict()
    tampered["decision"] = "failed"
    with pytest.raises(VPMValidationError, match="evaluation digest"):
        FinalEvaluationResultDTO.from_dict(tampered)


def test_evaluator_is_isolated_from_hostile_global_decimal_contexts() -> None:
    protocol = approved_protocol(
        threshold="1.6666666666666666666666666666666666", expected_row_count=3
    )
    expected_counts = {
        "evidence_row_count": 3,
        "episode_count": 3,
        "frame_count": 3,
        "provider_count": 1,
    }
    evidence = evidence_bundle(
        protocol,
        rows=final_rows(("1", "2", "2")),
        expected_counts=expected_counts,
    )
    original = getcontext().copy()
    results: list[FinalEvaluationResultDTO] = []
    try:
        for precision, rounding in ((2, ROUND_DOWN), (9, ROUND_UP), (77, ROUND_DOWN)):
            getcontext().prec = precision
            getcontext().rounding = rounding
            results.append(evaluate_final_protocol(protocol, evidence))
            assert getcontext().prec == precision
            assert getcontext().rounding == rounding
    finally:
        setcontext(original)

    assert all(result.decision == "passed" for result in results)
    assert all(result.to_dict() == results[0].to_dict() for result in results[1:])
    assert all(
        canonical_json_bytes(result.to_dict())
        == canonical_json_bytes(results[0].to_dict())
        for result in results[1:]
    )
    assert all(
        result.evaluation_digest == results[0].evaluation_digest
        for result in results[1:]
    )


@pytest.mark.parametrize(
    ("aggregate", "operator", "scores", "threshold"),
    [
        ("mean", "gte", ("1", "2"), "1.5"),
        ("mean", "lte", ("1", "2"), "1.5"),
        ("minimum", "gte", ("0.2", "0.9"), "0.2"),
        ("maximum", "lte", ("0.2", "0.9"), "0.9"),
    ],
)
def test_decimal_aggregate_threshold_equality_remains_inclusive(
    aggregate: str,
    operator: str,
    scores: tuple[str, ...],
    threshold: str,
) -> None:
    protocol = approved_protocol(
        decision_rule={
            "kind": "fixed_metric_threshold",
            "aggregate": aggregate,
            "metric_id": "score",
            "operator": operator,
            "threshold": threshold,
        }
    )
    result = evaluate_final_protocol(
        protocol, evidence_bundle(protocol, rows=final_rows(scores))
    )
    assert result.decision == "passed"


@pytest.mark.parametrize(
    ("scores", "threshold", "expected_count", "expected_decision"),
    [
        (("0.8", "0.9"), "0.75", 2, "passed"),
        (("0.5", "0.6"), "0.75", 2, "failed"),
        (("0.8", "0.9"), "0.75", 3, "indeterminate"),
    ],
)
def test_pass_fail_and_indeterminate_are_deterministic(
    scores: tuple[str, ...],
    threshold: str,
    expected_count: int,
    expected_decision: str,
) -> None:
    protocol = approved_protocol(
        threshold=threshold,
        expected_row_count=expected_count,
    )
    expected_counts = {
        "evidence_row_count": expected_count,
        "episode_count": expected_count,
        "frame_count": expected_count,
        "provider_count": 1,
    }
    evidence = evidence_bundle(
        protocol,
        rows=final_rows(scores),
        expected_counts=expected_counts,
    )
    first = evaluate_final_protocol(protocol, evidence)
    second = evaluate_final_protocol(protocol, evidence)
    assert first.decision == expected_decision
    assert first.to_dict() == second.to_dict()


@pytest.mark.parametrize(
    "rule",
    [
        {
            "aggregate": "mean",
            "metric_id": "score",
            "operator": "gte",
            "threshold": "1",
        },
        {
            "kind": "unknown",
            "aggregate": "mean",
            "metric_id": "score",
            "operator": "gte",
            "threshold": "1",
        },
        {
            "kind": "fixed_metric_threshold",
            "aggregate": "median",
            "metric_id": "score",
            "operator": "gte",
            "threshold": "1",
        },
        {
            "kind": "fixed_metric_threshold",
            "aggregate": "mean",
            "metric_id": "score",
            "operator": "eq",
            "threshold": "1",
        },
        {
            "kind": "fixed_metric_threshold",
            "aggregate": "mean",
            "metric_id": "score",
            "operator": "gte",
            "threshold": 0.75,
        },
        {
            "kind": "fixed_metric_threshold",
            "aggregate": "mean",
            "metric_id": "score",
            "operator": "gte",
            "threshold": "1",
            "extra": True,
        },
    ],
)
def test_evaluator_rejects_unsupported_decision_rule_shapes(
    rule: dict[str, object],
) -> None:
    protocol = approved_protocol(decision_rule=rule)
    evidence = evidence_bundle(protocol)
    with pytest.raises(VPMValidationError):
        evaluate_final_protocol(protocol, evidence)


@pytest.mark.parametrize("status", ["draft", "review", "retired"])
def test_nonapproved_protocol_cannot_enter_completion(
    tmp_path: Path,
    status: str,
) -> None:
    protocol = approved_protocol(status=status)
    auth = authorization(tmp_path, protocol)
    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
    )
    with pytest.raises(VPMValidationError, match="not approved"):
        service.execute_final_once(request(tmp_path, auth))
    assert not Path(auth.output_dir).exists()


def test_controlled_completion_binds_evaluation_claims_and_report(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    executor = SyntheticFinalExecutor(
        artifacts={"final-summary.json": b'{"evidence_digest":"fake"}'},
    )
    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=executor,
    )

    receipt = FinalExecutionReceiptDTO.from_dict(
        service.execute_final_once(request(tmp_path, auth))
    )
    output_dir = Path(auth.output_dir)
    evaluation = FinalEvaluationResultDTO.from_dict(
        json.loads((output_dir / FINAL_EVALUATION_NAME).read_text(encoding="utf-8"))
    )
    reconstruction = reconstruct_final_access_ledger(service.store, receipt.access_id)
    registry = build_final_claim_registry(
        receipt=receipt,
        evaluation_result=evaluation,
        claim_rules={"claims": [{"claim_id": "synthetic-claim"}]},
    )
    report = generate_final_report(
        receipt=receipt,
        evaluation_result=evaluation,
        claim_registry=registry,
    )

    assert not hasattr(service, "complete")
    assert executor.calls == [
        "materialize",
        "score_providers",
        "assess_reachability",
        "build_artifacts",
    ]
    assert receipt.evidence_digest == evaluation.evidence_digest
    assert receipt.evaluation_digest == evaluation.evaluation_digest
    assert receipt.protocol_digest == evaluation.protocol_digest
    assert receipt.artifact_manifest_digest.startswith("sha256:")
    assert receipt.event_chain_digest == reconstruction["event_chain_digest"]
    assert reconstruction["publication_status"] == "completed_receipt_valid"
    assert reconstruction["publishable_success"] is True
    assert (output_dir / FINAL_RECEIPT_NAME).is_file()
    assert registry["claims"][0]["status"] == "eligible"
    assert receipt.access_id in report
    for kind in (
        "final_materialization_started_count",
        "final_materialization_completed_count",
        "provider_scoring_started_count",
        "provider_scoring_completed_count",
        "reachability_started_count",
        "reachability_completed_count",
        "evaluation_started_count",
        "evaluation_completed_count",
        "artifact_validation_started_count",
        "artifact_validation_completed_count",
        "promotion_started_count",
        "promotion_completed_count",
        "receipt_publication_started_count",
        "receipt_publication_completed_count",
    ):
        assert reconstruction["counters"][kind] == 1


def test_fabricated_or_unbound_evaluation_cannot_create_claims(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
    )
    receipt = FinalExecutionReceiptDTO.from_dict(
        service.execute_final_once(request(tmp_path, auth))
    )
    bound_evaluation = FinalEvaluationResultDTO.from_dict(
        json.loads(
            (Path(auth.output_dir) / FINAL_EVALUATION_NAME).read_text(encoding="utf-8")
        )
    )
    registry = build_final_claim_registry(
        receipt=receipt,
        evaluation_result=bound_evaluation,
        claim_rules={"claims": []},
    )

    with pytest.raises(VPMValidationError, match="evaluation result"):
        build_final_claim_registry(
            receipt=receipt,
            evaluation_result={"decision": "passed"},  # type: ignore[arg-type]
            claim_rules={"claims": []},
        )
    other_protocol = approved_protocol(
        threshold="0.8",
        protocol_id="other-protocol",
    )
    other_evaluation = evaluate_final_protocol(
        other_protocol,
        evidence_bundle(other_protocol),
    )
    with pytest.raises(VPMValidationError, match="binding"):
        build_final_claim_registry(
            receipt=receipt,
            evaluation_result=other_evaluation,
            claim_rules={"claims": []},
        )
    with pytest.raises(VPMValidationError, match="evaluation result"):
        generate_final_report(
            receipt=receipt,
            evaluation_result={"decision": "passed"},  # type: ignore[arg-type]
            claim_registry=registry,
        )
    with pytest.raises(VPMValidationError, match="binding"):
        generate_final_report(
            receipt=receipt,
            evaluation_result=other_evaluation,
            claim_registry=registry,
        )
    other_evidence = evidence_bundle(
        protocol,
        rows=final_rows(("0.7", "0.8")),
    )
    other_evaluation = evaluate_final_protocol(protocol, other_evidence)
    with pytest.raises(VPMValidationError, match="binding"):
        build_final_claim_registry(
            receipt=receipt,
            evaluation_result=other_evaluation,
            claim_rules={"claims": []},
        )


@pytest.mark.parametrize("status", ["draft", "review", "retired"])
def test_protocol_reloaded_at_evaluation_must_still_be_approved(
    tmp_path: Path,
    status: str,
) -> None:
    approved = approved_protocol()
    auth = authorization(tmp_path, approved)
    store = InMemoryVideoActionSetStore()
    replacement = approved_protocol(
        status=status,
        protocol_id=f"replacement-{status}",
    )

    def replace_protocol(boundary: str) -> None:
        if boundary == "evaluation_started":
            store._final_protocols[approved.protocol_digest] = replacement

    service = FinalAccessService(
        store=store,
        final_executor=SyntheticFinalExecutor(),
        failure_injector=replace_protocol,
    )
    with pytest.raises(VPMValidationError, match="not approved"):
        service.execute_final_once(request(tmp_path, auth))
    assert not (Path(auth.output_dir) / FINAL_RECEIPT_NAME).exists()


def test_cli_preflight_is_read_only_in_memory_validation(tmp_path: Path) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    req = request(tmp_path, auth, preflight_only=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "zeromodel.persistence.sqlalchemy.video_action_set_final_cli",
            "--output-dir",
            auth.output_dir,
            "--authorization-file",
            req.authorization_file,
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
        env=_source_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["preflight"] == "passed"
    assert payload["validation_store"] == "in-memory-nonauthoritative"
    assert payload["reservation_created"] is False
    assert not Path(auth.database_path).exists()


def test_cli_rejects_force_like_options(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "zeromodel.persistence.sqlalchemy.video_action_set_final_cli",
            "--output-dir",
            str(tmp_path),
            "--authorization-file",
            str(tmp_path / "missing.json"),
            "--expected-authorization-digest",
            "sha256:" + "a" * 64,
            "--expected-sealed-plan-digest",
            "sha256:" + "b" * 64,
            "--database-path",
            str(tmp_path / "final.sqlite3"),
            "--force",
        ],
        cwd=REPO_ROOT,
        env=_source_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --force" in result.stderr


def test_reconciler_interrupts_only_after_process_is_gone(tmp_path: Path) -> None:
    runtime = build_runtime()
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    record = runtime.video_action_set.create_final_authorization(auth, protocol)
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


def test_execute_final_once_reads_authorization_and_protocol_files_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the final-authorization TOCTOU fix: before the fix,
    `execute_final_once()` called `load_final_authorization_file()` once
    inside `preflight_final_execution()` and a second time directly, with
    an open window between the two reads. Every authority-bearing file may
    now be read at most once per `execute_final_once()` call."""
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    req = request(tmp_path, auth)
    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
    )

    authorization_reads: list[Path] = []
    protocol_reads: list[Path] = []
    original_load_authorization = final_access_service_module.load_final_authorization_file
    original_load_protocol = final_access_service_module.load_final_protocol_file

    def counting_load_authorization(path: Path) -> FinalExecutionAuthorizationDTO:
        authorization_reads.append(path)
        return original_load_authorization(path)

    def counting_load_protocol(path: Path):
        protocol_reads.append(path)
        return original_load_protocol(path)

    monkeypatch.setattr(
        final_access_service_module,
        "load_final_authorization_file",
        counting_load_authorization,
    )
    monkeypatch.setattr(
        final_access_service_module, "load_final_protocol_file", counting_load_protocol
    )

    service.execute_final_once(req)

    assert len(authorization_reads) == 1
    assert len(protocol_reads) == 1


def test_authorization_file_replaced_mid_execution_cannot_become_execution_authority(
    tmp_path: Path,
) -> None:
    """Authorization A is resolved once at the start of `execute_final_once()`.
    Replacing the authorization file on disk after that resolution - here,
    at the `final_materialization_started` boundary, well inside actual
    execution - must not let the replacement (authorization B) become the
    execution authority: the resulting receipt must still reflect A's own
    `execution_commit`, never B's."""
    protocol = approved_protocol()
    auth_a = authorization(tmp_path, protocol)
    req = request(tmp_path, auth_a)
    authorization_file = Path(req.authorization_file)

    replacement_payload = auth_a.to_dict()
    replacement_payload["authorization_payload"] = dict(
        replacement_payload["authorization_payload"]
    )
    replacement_payload["authorization_payload"]["execution_commit"] = (
        "replacement-commit-must-never-execute"
    )
    replacement_payload.pop("authorization_digest")
    auth_b = FinalExecutionAuthorizationDTO.create(replacement_payload)
    assert auth_b.authorization_digest != auth_a.authorization_digest

    def replace_authorization_file(boundary: str) -> None:
        if boundary == "final_materialization_started":
            authorization_file.write_text(
                json.dumps(auth_b.to_dict(), ensure_ascii=False), encoding="utf-8"
            )

    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
        failure_injector=replace_authorization_file,
    )

    receipt = FinalExecutionReceiptDTO.from_dict(service.execute_final_once(req))

    # The replacement genuinely landed on disk...
    on_disk = json.loads(authorization_file.read_text(encoding="utf-8"))
    assert (
        on_disk["authorization_payload"]["execution_commit"]
        == "replacement-commit-must-never-execute"
    )
    # ...but execution still used the authorization resolved before that
    # boundary fired, never the replacement.
    assert receipt.execution_commit == "synthetic-execution-commit"
    assert receipt.authorization_digest == auth_a.authorization_digest


def test_preflight_only_creates_no_store_state(tmp_path: Path) -> None:
    """`preflight_only=True` must remain read-only at the service level
    (not merely at the CLI level): no authorization, access record, or
    reservation is created in the store."""
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    req = request(tmp_path, auth, preflight_only=True)
    store = InMemoryVideoActionSetStore()
    service = FinalAccessService(store=store, final_executor=SyntheticFinalExecutor())

    result = service.execute_final_once(req)

    assert result["preflight"] == "passed"
    assert result["reservation_created"] is False
    assert store.load_final_authorization(auth.authorization_id) is None
    assert not Path(auth.database_path).exists()


def test_mismatched_existing_authorization_creates_no_reservation(
    tmp_path: Path,
) -> None:
    """An authorization already recorded under the same id, but not equal
    to the one currently on disk, must be rejected before any reservation
    is attempted."""
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    store = InMemoryVideoActionSetStore()
    service = FinalAccessService(store=store, final_executor=SyntheticFinalExecutor())
    service.create_authorization(auth, protocol)

    mutated_payload = auth.to_dict()
    mutated_payload["authorization_payload"] = dict(
        mutated_payload["authorization_payload"]
    )
    mutated_payload["authorization_payload"]["execution_commit"] = "different-commit"
    mutated_payload.pop("authorization_digest")
    mutated_auth = FinalExecutionAuthorizationDTO.create(mutated_payload)
    # request() writes whatever authorization object it is given to
    # `<authorization_id>-authorization.json` - both `auth` and
    # `mutated_auth` share authorization_id "auth-1", so this overwrites
    # the file with the mutated content while producing a request whose
    # expected digest matches it.
    req = request(tmp_path, mutated_auth)

    with pytest.raises(VPMValidationError, match="authorization mismatch"):
        service.execute_final_once(req)

    record = store.load_final_access_record(
        final_access_service_module.access_id_for_authorization(auth)
    )
    assert record is not None
    assert record.state == "authorized"
