from __future__ import annotations

from pathlib import Path

import pytest

from video_final_test_support import (
    SyntheticFinalExecutor,
    approved_protocol,
    authorization,
    request,
)
from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.final_access_dto import (
    FinalEvaluationResultDTO,
    access_id_for_authorization,
)
from zeromodel.video.domains.video_action_set.final_access_service import FinalAccessService
from zeromodel.video.domains.video_action_set.final_claims import build_final_claim_registry
from zeromodel.video.domains.video_action_set.final_publication import (
    FINAL_EVALUATION_NAME,
    FINAL_RECEIPT_NAME,
    load_published_receipt,
)
from zeromodel.video.domains.video_action_set.final_reconstruction import (
    reconstruct_final_access_ledger,
)
from zeromodel.video.domains.video_action_set.final_reporting import generate_final_report
from zeromodel.video.stores.video_action_set_memory import InMemoryVideoActionSetStore


pytestmark = pytest.mark.integration


class InjectedFailure(RuntimeError):
    pass


@pytest.mark.parametrize(
    (
        "boundary",
        "state",
        "staging",
        "canonical",
        "receipt",
        "publication_status",
        "last_kind",
    ),
    [
        (
            "before_staging_validation",
            "failed",
            False,
            False,
            False,
            "failed",
            "failed",
        ),
        (
            "during_staging_validation",
            "failed",
            False,
            False,
            False,
            "failed",
            "failed",
        ),
        ("before_promotion", "failed", True, False, False, "failed", "failed"),
        ("during_promotion", "failed", True, False, False, "failed", "failed"),
        ("after_promotion", "failed", False, True, False, "failed", "failed"),
        (
            "before_completed_transition",
            "failed",
            False,
            True,
            False,
            "failed",
            "failed",
        ),
        (
            "during_completed_transition",
            "failed",
            False,
            True,
            False,
            "failed",
            "failed",
        ),
        (
            "after_completed_transition",
            "completed",
            False,
            True,
            False,
            "completed_receipt_missing",
            "completed",
        ),
        (
            "before_temporary_receipt_write",
            "completed",
            False,
            True,
            False,
            "completed_receipt_missing",
            "receipt_publication_completed",
        ),
        (
            "during_temporary_receipt_write",
            "completed",
            False,
            True,
            False,
            "completed_receipt_missing",
            "receipt_publication_completed",
        ),
        (
            "before_receipt_rename",
            "completed",
            False,
            True,
            False,
            "completed_receipt_missing",
            "receipt_publication_completed",
        ),
        (
            "during_receipt_rename",
            "completed",
            False,
            True,
            False,
            "completed_receipt_missing",
            "receipt_publication_completed",
        ),
        (
            "after_receipt_publication",
            "completed",
            False,
            True,
            True,
            "completed_receipt_valid",
            "receipt_publication_completed",
        ),
    ],
)
def test_failure_boundaries_preserve_receipt_last_contract(
    tmp_path: Path,
    boundary: str,
    state: str,
    staging: bool,
    canonical: bool,
    receipt: bool,
    publication_status: str,
    last_kind: str,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    executor = SyntheticFinalExecutor()

    def inject(name: str) -> None:
        if name == boundary:
            raise InjectedFailure(name)

    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=executor,
        failure_injector=inject,
    )
    execution_request = request(tmp_path, auth)
    with pytest.raises(InjectedFailure, match=boundary):
        service.execute_final_once(execution_request)

    access_id = access_id_for_authorization(auth)
    reconstruction = reconstruct_final_access_ledger(service.store, access_id)
    events = service.list_events(access_id)
    output = Path(auth.output_dir)
    staged = tuple(tmp_path.glob(".*.final-staging-*"))
    event_payload = events[-1].event_payload.to_value()

    assert reconstruction["record"]["state"] == state
    assert bool(staged) is staging
    assert output.exists() is canonical
    assert (output / FINAL_RECEIPT_NAME).exists() is receipt
    assert reconstruction["publication_status"] == publication_status
    assert reconstruction["publishable_success"] is receipt
    assert isinstance(event_payload, dict)
    assert event_payload["kind"] == last_kind
    assert executor.calls == [
        "materialize",
        "score_providers",
        "assess_reachability",
        "build_artifacts",
    ]

    before_retry_calls = tuple(executor.calls)
    with pytest.raises(VPMValidationError, match="terminal state|transition"):
        service.execute_final_once(execution_request)
    assert tuple(executor.calls) == before_retry_calls


def test_missing_or_invalid_receipt_blocks_claims_reports_and_repair(
    tmp_path: Path,
) -> None:
    import json

    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    service = FinalAccessService(
        store=InMemoryVideoActionSetStore(),
        final_executor=SyntheticFinalExecutor(),
    )
    service.execute_final_once(request(tmp_path, auth))
    output = Path(auth.output_dir)
    evaluation = FinalEvaluationResultDTO.from_dict(
        json.loads((output / FINAL_EVALUATION_NAME).read_text(encoding="utf-8"))
    )
    receipt_path = output / FINAL_RECEIPT_NAME
    receipt_path.unlink()

    access_id = access_id_for_authorization(auth)
    missing = reconstruct_final_access_ledger(service.store, access_id)
    assert missing["publication_status"] == "completed_receipt_missing"
    assert load_published_receipt(output) is None
    with pytest.raises(VPMValidationError, match="receipt"):
        build_final_claim_registry(
            receipt=None,  # type: ignore[arg-type]
            evaluation_result=evaluation,
            claim_rules={"claims": []},
        )
    with pytest.raises(VPMValidationError, match="receipt"):
        generate_final_report(
            receipt=None,  # type: ignore[arg-type]
            evaluation_result=evaluation,
            claim_registry={},
        )
    assert not receipt_path.exists()

    receipt_path.write_text('{"invalid":true}', encoding="utf-8")
    invalid_bytes = receipt_path.read_bytes()
    invalid = reconstruct_final_access_ledger(service.store, access_id)
    assert invalid["publication_status"] == "completed_receipt_invalid"
    with pytest.raises(VPMValidationError):
        load_published_receipt(output)
    assert receipt_path.read_bytes() == invalid_bytes
