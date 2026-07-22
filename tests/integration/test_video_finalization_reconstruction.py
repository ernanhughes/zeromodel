from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from video_final_test_support import (
    SyntheticFinalExecutor,
    approved_protocol,
    authorization,
    request,
)
from zeromodel.persistence.sqlalchemy.db.runtime import build_finalization_sqlite_runtime
from zeromodel.persistence.sqlalchemy.db.session import sqlite_database_url
from zeromodel.video.domains.video_action_set.final_access_dto import (
    FinalExecutionReceiptDTO,
)
from zeromodel.video.domains.video_action_set.final_access_service import FinalAccessService
from zeromodel.video.domains.video_action_set.final_publication import FINAL_RECEIPT_NAME


pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]


def _database_url(path: Path) -> str:
    return sqlite_database_url(path)


def _service(database_path: Path) -> FinalAccessService:
    runtime = build_finalization_sqlite_runtime(
        _database_url(database_path),
        initialize_authority=True,
    )
    return FinalAccessService(
        store=runtime.video_action_set.engine.final_access_service.store,
        final_executor=SyntheticFinalExecutor(),
    )


def _run_admin(database: Path, access_id: str) -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "zeromodel.video_action_set_final_admin_cli",
            "reconstruct",
            "--database-path",
            str(database),
            "--access-id",
            access_id,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_completed_authority_reconstructs_all_bindings_in_fresh_process(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    service = _service(Path(auth.database_path))
    receipt = FinalExecutionReceiptDTO.from_dict(
        service.execute_final_once(request(tmp_path, auth))
    )
    script = """
import json
from pathlib import Path
import sys
from zeromodel.db.runtime import build_finalization_sqlite_runtime
from zeromodel.domains.video_action_set.final_access_dto import FinalEvaluationResultDTO
from zeromodel.domains.video_action_set.final_publication import (
    FINAL_EVALUATION_NAME,
    load_canonical_artifact_manifest,
    load_published_receipt,
)
from zeromodel.domains.video_action_set.final_reconstruction import reconstruct_final_access_ledger

database = Path(sys.argv[1]).resolve()
access_id = sys.argv[2]
url = database.as_uri().replace("file:///", "sqlite:///")
store = build_finalization_sqlite_runtime(url).video_action_set.engine.final_access_service.store
record = store.load_final_access_record(access_id)
authorization = store.load_final_authorization(record.authorization_id)
protocol = store.load_final_evaluation_protocol(record.protocol_digest)
output = Path(authorization.output_dir)
manifest = load_canonical_artifact_manifest(output)
receipt = load_published_receipt(output)
evaluation = FinalEvaluationResultDTO.from_dict(json.loads((output / FINAL_EVALUATION_NAME).read_text(encoding="utf-8")))
contract = authorization.authorization_payload.to_value()
print(json.dumps({
    "reconstruction": reconstruct_final_access_ledger(store, access_id),
    "authorization": authorization.to_dict(),
    "protocol": protocol.to_dict(),
    "historical_authority": contract["historical_authority"],
    "manifest": manifest.to_dict(),
    "receipt": receipt.to_dict(),
    "evaluation": evaluation.to_dict(),
}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, auth.database_path, receipt.access_id],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    reconstruction = payload["reconstruction"]
    completion = reconstruction["record"]["record_payload"]["completion"]

    assert reconstruction["publication_status"] == "completed_receipt_valid"
    assert reconstruction["publishable_success"] is True
    assert reconstruction["event_chain_digest"] == receipt.event_chain_digest
    assert reconstruction["counters"]["durable_event_count"] == len(
        reconstruction["events"]
    )
    assert payload["authorization"]["authorization_digest"] == (
        receipt.authorization_digest
    )
    assert payload["protocol"]["protocol_digest"] == receipt.protocol_digest
    assert payload["historical_authority"]["historical_authority_id"] == (
        receipt.historical_authority_id
    )
    assert payload["evaluation"]["evidence_digest"] == receipt.evidence_digest
    assert payload["evaluation"]["evaluation_digest"] == receipt.evaluation_digest
    assert payload["manifest"]["artifact_manifest_digest"] == (
        receipt.artifact_manifest_digest
    )
    assert payload["receipt"]["receipt_digest"] == receipt.receipt_digest
    assert completion["evidence_digest"] == receipt.evidence_digest
    assert completion["evaluation_digest"] == receipt.evaluation_digest


@pytest.mark.parametrize(
    "expected_status",
    [
        "failed",
        "interrupted",
        "completed_receipt_missing",
        "completed_receipt_invalid",
        "completed_receipt_valid",
    ],
)
def test_terminal_publication_status_reconstructs_in_fresh_process(
    tmp_path: Path,
    expected_status: str,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    service = _service(Path(auth.database_path))
    if expected_status in {"failed", "interrupted"}:
        record = service.create_authorization(auth, protocol)
        record = service.reserve(record.access_id)
        record = service.mark_running(record.access_id)
        transition = service.fail if expected_status == "failed" else service.interrupt
        record = transition(
            record.access_id,
            failure_kind="synthetic",
            error_code="synthetic",
            error_message="synthetic",
        )
        access_id = record.access_id
    else:
        receipt = FinalExecutionReceiptDTO.from_dict(
            service.execute_final_once(request(tmp_path, auth))
        )
        access_id = receipt.access_id
        receipt_path = Path(auth.output_dir) / FINAL_RECEIPT_NAME
        if expected_status == "completed_receipt_missing":
            receipt_path.unlink()
        elif expected_status == "completed_receipt_invalid":
            receipt_path.write_text('{"invalid":true}', encoding="utf-8")

    reconstruction = _run_admin(Path(auth.database_path), access_id)
    assert reconstruction["publication_status"] == expected_status
    assert reconstruction["publishable_success"] is (
        expected_status == "completed_receipt_valid"
    )


def test_claim_eligibility_is_derived_in_fresh_process(tmp_path: Path) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    service = _service(Path(auth.database_path))
    receipt = FinalExecutionReceiptDTO.from_dict(
        service.execute_final_once(request(tmp_path, auth))
    )
    script = """
import json
from pathlib import Path
import sys
from zeromodel.db.runtime import build_finalization_sqlite_runtime
from zeromodel.domains.video_action_set.final_access_dto import FinalEvaluationResultDTO
from zeromodel.domains.video_action_set.final_claims import build_final_claim_registry
from zeromodel.domains.video_action_set.final_publication import FINAL_EVALUATION_NAME, load_published_receipt

database = Path(sys.argv[1]).resolve()
access_id = sys.argv[2]
url = database.as_uri().replace("file:///", "sqlite:///")
store = build_finalization_sqlite_runtime(url).video_action_set.engine.final_access_service.store
record = store.load_final_access_record(access_id)
authorization = store.load_final_authorization(record.authorization_id)
protocol = store.load_final_evaluation_protocol(record.protocol_digest)
output = Path(authorization.output_dir)
receipt = load_published_receipt(output)
evaluation = FinalEvaluationResultDTO.from_dict(json.loads((output / FINAL_EVALUATION_NAME).read_text(encoding="utf-8")))
registry = build_final_claim_registry(
    receipt=receipt,
    evaluation_result=evaluation,
    claim_rules=protocol.claim_rule_set.to_value(),
)
print(json.dumps(registry, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, auth.database_path, receipt.access_id],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    registry = json.loads(result.stdout)
    assert registry["receipt_digest"] == receipt.receipt_digest
    assert registry["claims"][0]["status"] == "eligible"
