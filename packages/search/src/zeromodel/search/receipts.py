from __future__ import annotations

from zeromodel.artifacts import ArtifactRef, ArtifactStore

from zeromodel.search.dto import RelationSearchReceiptDTO
from zeromodel.search.errors import SearchReplayMismatchError
from zeromodel.search.persistence import (
    load_relation_search_receipt,
    load_relation_search_request,
    load_relation_search_result,
    store_relation_search_receipt,
)
from zeromodel.search.ranking import search_relation


def build_relation_search_receipt(
    *,
    store: ArtifactStore,
    request_ref: ArtifactRef,
    result_ref: ArtifactRef,
) -> tuple[RelationSearchReceiptDTO, ArtifactRef]:
    request = load_relation_search_request(store, request_ref)
    result = load_relation_search_result(store, result_ref)
    receipt = RelationSearchReceiptDTO(
        request_ref=request_ref,
        result_ref=result_ref,
        readout_ref=request.readout_ref,
        corpus_ref=request.corpus_ref,
        required_checks=RelationSearchReceiptDTO.REQUIRED,
        result_id=result.result_id,
    )
    return receipt, store_relation_search_receipt(store, receipt)


def replay_relation_search(
    *,
    store: ArtifactStore,
    receipt_ref: ArtifactRef,
) -> object:
    receipt = load_relation_search_receipt(store, receipt_ref)
    request = load_relation_search_request(store, receipt.request_ref)
    fresh, fresh_ref, _ = search_relation(
        store=store, request=request, persist_result=True
    )
    if fresh.result_id != receipt.result_id:
        raise SearchReplayMismatchError("replayed search result diverged from receipt")
    if fresh_ref is None or fresh_ref.artifact_id != receipt.result_ref.artifact_id:
        raise SearchReplayMismatchError("replayed result ref diverged from receipt")
    return fresh
