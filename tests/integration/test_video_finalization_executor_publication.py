from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import sys

import pytest

from video_final_test_support import (
    SyntheticFinalExecutor,
    approved_protocol,
    authorization,
    final_rows,
    request,
)
from zeromodel import build_runtime
from zeromodel.artifact import VPMValidationError
from zeromodel.db.runtime import build_finalization_sqlite_runtime
from zeromodel.db.session import sqlite_database_url
from zeromodel.domains.video_action_set.final_access_dto import (
    FinalEvaluationResultDTO,
    FinalExecutionReceiptDTO,
)
from zeromodel.domains.video_action_set.final_access_service import FinalAccessService
from zeromodel.domains.video_action_set.final_publication import (
    FINAL_EVALUATION_NAME,
    FINAL_RECEIPT_NAME,
)


pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]


def _database_url(path: Path) -> str:
    return sqlite_database_url(path)


def _service(
    database_path: Path,
    executor: SyntheticFinalExecutor,
    failure_injector: Callable[[str], None] | None = None,
) -> FinalAccessService:
    runtime = build_finalization_sqlite_runtime(
        _database_url(database_path),
        initialize_authority=True,
    )
    return FinalAccessService(
        store=runtime.video_action_set.engine.final_access_service.store,
        final_executor=executor,
        failure_injector=failure_injector,
    )


def _row(**changes: object) -> dict[str, object]:
    row = dict(final_rows()[0])
    row.update(changes)
    return row


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            (_row(episode_id="wrong-episode"), final_rows()[1]),
            "authorized identities",
        ),
        ((final_rows()[0],), "authorized counts"),
        ((final_rows()[0], final_rows()[0]), "duplicate final evidence identity"),
        (
            (_row(provider_id="P2"), final_rows()[1]),
            "provider order",
        ),
        (tuple(reversed(final_rows())), "evidence ordering"),
    ],
    ids=[
        "wrong-episode",
        "missing-frame-and-wrong-count",
        "duplicate-frame-provider",
        "wrong-provider-and-provider-order",
        "noncanonical-order",
    ],
)
def test_untrusted_executor_rows_are_rejected(
    tmp_path: Path,
    rows: tuple[dict[str, object], ...],
    message: str,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    executor = SyntheticFinalExecutor(rows=rows)
    service = _service(Path(auth.database_path), executor)

    with pytest.raises(VPMValidationError, match=message):
        service.execute_final_once(request(tmp_path, auth))

    assert not (Path(auth.output_dir) / FINAL_RECEIPT_NAME).exists()
    assert executor.calls.count("assess_reachability") == 1
    assert "build_artifacts" not in executor.calls


def test_executor_cannot_fabricate_protocol_evaluation_decision_or_digest(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol(threshold="0.75")
    auth = authorization(tmp_path, protocol)
    fabricated = {
        "decision": "passed",
        "protocol_digest": "sha256:" + "f" * 64,
        "evaluation_digest": "sha256:" + "e" * 64,
        "evidence_digest": "sha256:" + "d" * 64,
    }
    executor = SyntheticFinalExecutor(
        rows=final_rows(("0.1", "0.2")),
        artifacts={
            "final-summary.json": json.dumps(fabricated, sort_keys=True).encode(),
        },
    )
    service = _service(Path(auth.database_path), executor)

    receipt = FinalExecutionReceiptDTO.from_dict(
        service.execute_final_once(request(tmp_path, auth))
    )
    evaluation = FinalEvaluationResultDTO.from_dict(
        json.loads(
            (Path(auth.output_dir) / FINAL_EVALUATION_NAME).read_text(encoding="utf-8")
        )
    )

    assert receipt.decision == "failed"
    assert evaluation.decision == "failed"
    assert receipt.protocol_digest == protocol.protocol_digest
    assert receipt.evaluation_digest == evaluation.evaluation_digest
    assert (
        json.loads(
            (Path(auth.output_dir) / "final-summary.json").read_text(encoding="utf-8")
        )
        == fabricated
    )


def test_real_filesystem_promotion_is_atomic_and_authorization_specific(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    service = _service(Path(auth.database_path), SyntheticFinalExecutor())

    receipt = FinalExecutionReceiptDTO.from_dict(
        service.execute_final_once(request(tmp_path, auth))
    )
    output = Path(auth.output_dir)
    assert output.is_dir()
    assert sorted(path.name for path in output.iterdir()) == [
        "final-artifact-manifest.json",
        "final-evaluation.json",
        "final-evidence.json",
        "final-execution-receipt.json",
        "final-summary.json",
    ]
    assert not tuple(tmp_path.glob(".*.final-staging-*"))
    assert receipt.authorization_id == auth.authorization_id

    with pytest.raises(VPMValidationError, match="terminal state|transition"):
        service.execute_final_once(request(tmp_path, auth))
    assert len(tuple(output.iterdir())) == 5


@pytest.mark.parametrize(
    ("filename", "replacement"),
    [
        ("final-summary.json", b"x"),
        ("final-summary.json", b'{"synthetic":null}'),
        ("final-evaluation.json", b"[]"),
    ],
    ids=["changed-length", "changed-hash", "changed-json-structure"],
)
def test_staged_file_mutation_is_rejected_before_real_promotion(
    tmp_path: Path,
    filename: str,
    replacement: bytes,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)

    def mutate(boundary: str) -> None:
        if boundary == "during_promotion":
            staging = tuple(tmp_path.glob(".*.final-staging-*"))
            assert len(staging) == 1
            (staging[0] / filename).write_bytes(replacement)

    service = _service(
        Path(auth.database_path), SyntheticFinalExecutor(), failure_injector=mutate
    )
    with pytest.raises(VPMValidationError, match="manifest validation"):
        service.execute_final_once(request(tmp_path, auth))
    assert not Path(auth.output_dir).exists()


def test_symlinked_staged_artifact_is_rejected_when_supported(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    target = tmp_path / "symlink-target.json"
    target.write_bytes(b'{"synthetic":true}')

    def replace_with_symlink(boundary: str) -> None:
        if boundary != "during_promotion":
            return
        staging = tuple(tmp_path.glob(".*.final-staging-*"))
        assert len(staging) == 1
        artifact = staging[0] / "final-summary.json"
        artifact.unlink()
        try:
            artifact.symlink_to(target)
        except OSError:
            pytest.skip("symlink creation is unavailable")

    service = _service(
        Path(auth.database_path),
        SyntheticFinalExecutor(),
        failure_injector=replace_with_symlink,
    )
    with pytest.raises(VPMValidationError, match="symlinked final staging"):
        service.execute_final_once(request(tmp_path, auth))
    assert not Path(auth.output_dir).exists()


def test_symlinked_output_parent_is_rejected_when_supported(tmp_path: Path) -> None:
    target = tmp_path / "target-parent"
    target.mkdir()
    linked = tmp_path / "linked-parent"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    protocol = approved_protocol()
    auth = authorization(linked, protocol)
    service = _service(Path(auth.database_path), SyntheticFinalExecutor())
    with pytest.raises(VPMValidationError, match="symlinked final path"):
        service.execute_final_once(request(linked, auth))
    assert not Path(auth.output_dir).exists()


def test_no_executor_registration_leaks_across_runtime_or_process() -> None:
    assert (
        build_runtime().video_action_set.engine.final_access_service.final_executor
        is None
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from zeromodel import build_runtime; "
                "s=build_runtime().video_action_set.engine.final_access_service; "
                "raise SystemExit(0 if s.final_executor is None else 1)"
            ),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
