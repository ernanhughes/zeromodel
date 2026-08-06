from __future__ import annotations

import numpy as np

from zeromodel.artifacts import ArtifactRef, ArtifactStore
from zeromodel.core.matrix_blob import MatrixBlob

from zeromodel.search.dto import (
    RelationSearchHitDTO,
    RelationSearchRequestDTO,
    RelationSearchResultDTO,
    canonical_dto_bytes,
)
from zeromodel.search.errors import (
    CandidateGenerationError,
    RepresentationMismatchError,
    SearchValidationError,
)
from zeromodel.search.persistence import (
    load_matrix_blob,
    load_relation_readout_aggregate,
    load_representation_batch_aggregate,
    store_relation_search_request,
    store_relation_search_result,
)
from zeromodel.search.projection import compiled_from_aggregate

ORDERING_WITH_COSINE = "relation_distance_asc;cosine_distance_asc;artifact_id_asc"
ORDERING_RELATION_ONLY = "relation_distance_asc;artifact_id_asc"


def _cosine_distances(source: np.ndarray, targets: np.ndarray) -> np.ndarray:
    source_norm = float(np.linalg.norm(source))
    target_norms = np.linalg.norm(targets, axis=1)
    if source_norm == 0.0 or np.any(target_norms == 0.0):
        raise SearchValidationError(
            "cosine distance is undefined for zero-norm representations"
        )
    similarity = (targets @ source) / (target_norms * source_norm)
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def _query_vector(store: ArtifactStore, ref: ArtifactRef) -> np.ndarray:
    blob = load_matrix_blob(store, ref)
    array = np.asarray(blob.to_array(), dtype=np.float64)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise SearchValidationError(
            "query representation must be a vector or one-row matrix"
        )
    if not np.isfinite(array).all():
        raise SearchValidationError("query representation must be finite")
    return array


def search_relation(
    *,
    store: ArtifactStore,
    request: RelationSearchRequestDTO,
    persist_result: bool = True,
) -> tuple[RelationSearchResultDTO, ArtifactRef | None, ArtifactRef]:
    if request.candidate_generation_spec_ref is not None:
        raise CandidateGenerationError(
            "candidate generation is not implemented in zeromodel-search v1"
        )
    request_ref = store_relation_search_request(store, request)
    readout_aggregate = load_relation_readout_aggregate(request.readout_ref, store)
    corpus = load_representation_batch_aggregate(request.corpus_ref, store)
    if (
        corpus.spec.representation_spec_id
        != readout_aggregate.representation_spec.representation_spec_id
    ):
        raise RepresentationMismatchError(
            "corpus representation spec does not match readout"
        )
    runtime = compiled_from_aggregate(readout_aggregate)
    query = _query_vector(store, request.query_representation_blob_ref)
    query_projected = runtime.project_one(
        query,
        representation_spec_id=corpus.spec.representation_spec_id,
    )
    total = len(corpus.batch.item_refs)
    excluded = {ref.artifact_id for ref in request.exclude_refs}
    if len(excluded) != len(request.exclude_refs):
        raise SearchValidationError("exclude refs must be unique")
    if not excluded.issubset({ref.artifact_id for ref in corpus.batch.item_refs}):
        raise SearchValidationError("excluded artifacts must belong to the corpus")
    available = tuple(
        index
        for index, ref in enumerate(corpus.batch.item_refs)
        if ref.artifact_id not in excluded
    )
    if not available:
        result = RelationSearchResultDTO(
            request_ref=request_ref,
            readout_ref=request.readout_ref,
            corpus_ref=request.corpus_ref,
            relation_contract_ref=readout_aggregate.readout.relation_contract_ref,
            representation_spec_ref=readout_aggregate.readout.representation_spec_ref,
            query_projected_coordinates=tuple(float(v) for v in query_projected),
            hits=(),
            total_candidates=total,
            evaluated_candidates=0,
            ordering_contract=ORDERING_WITH_COSINE
            if request.include_cosine_comparison
            else ORDERING_RELATION_ONLY,
        )
        return (
            result,
            store_relation_search_result(store, result) if persist_result else None,
            request_ref,
        )

    matrix = corpus.matrix[np.asarray(available, dtype=np.int64)]
    projected = runtime.project_many(
        matrix, representation_spec_id=corpus.spec.representation_spec_id
    )
    deltas = projected - query_projected
    distances = np.max(np.abs(deltas), axis=1)
    refs = tuple(corpus.batch.item_refs[index] for index in available)
    ref_ids = np.asarray([ref.artifact_id for ref in refs])
    cosine = (
        _cosine_distances(query, matrix) if request.include_cosine_comparison else None
    )
    relation_order = sorted(
        range(len(refs)), key=lambda i: (float(distances[i]), refs[i].artifact_id)
    )
    relation_rank_by_pos = {pos: rank + 1 for rank, pos in enumerate(relation_order)}
    cosine_rank_by_pos: dict[int, int] = {}
    if cosine is not None:
        cosine_order = sorted(
            range(len(refs)), key=lambda i: (float(cosine[i]), refs[i].artifact_id)
        )
        cosine_rank_by_pos = {pos: rank + 1 for rank, pos in enumerate(cosine_order)}
        order = sorted(
            range(len(refs)),
            key=lambda i: (float(distances[i]), float(cosine[i]), refs[i].artifact_id),
        )
    else:
        order = relation_order
    wanted = min(request.k, len(order))
    hits = []
    coordinate_ids = readout_aggregate.relation_contract.coordinate_ids
    for rank, pos in enumerate(order[:wanted], start=1):
        abs_deltas = np.abs(deltas[pos])
        dominant_index = int(np.argmax(abs_deltas)) if abs_deltas.size else 0
        cosine_rank = cosine_rank_by_pos.get(pos) if cosine is not None else None
        rank_shift = None if cosine_rank is None else cosine_rank - rank
        hits.append(
            RelationSearchHitDTO(
                artifact_ref=refs[pos],
                rank=rank,
                relation_distance=float(distances[pos]),
                predicted_coordinates=tuple(float(v) for v in projected[pos]),
                coordinate_deltas=tuple(float(v) for v in deltas[pos]),
                cosine_distance=None if cosine is None else float(cosine[pos]),
                cosine_rank=cosine_rank,
                relation_rank=relation_rank_by_pos[pos],
                rank_shift=rank_shift,
                dominant_coordinate_id=coordinate_ids[dominant_index],
                dominant_coordinate_delta=float(deltas[pos][dominant_index]),
            )
        )
    _ = ref_ids  # documents that row refs, not indexes, are the public identity.
    result = RelationSearchResultDTO(
        request_ref=request_ref,
        readout_ref=request.readout_ref,
        corpus_ref=request.corpus_ref,
        relation_contract_ref=readout_aggregate.readout.relation_contract_ref,
        representation_spec_ref=readout_aggregate.readout.representation_spec_ref,
        query_projected_coordinates=tuple(float(v) for v in query_projected),
        hits=tuple(hits),
        total_candidates=total,
        evaluated_candidates=len(available),
        ordering_contract=ORDERING_WITH_COSINE
        if request.include_cosine_comparison
        else ORDERING_RELATION_ONLY,
        metadata={
            "coordinate_space": "robust_scaled_predicted_relation_coordinates",
            "candidate_generation": "exact_exhaustive",
        },
    )
    result_ref = store_relation_search_result(store, result) if persist_result else None
    return result, result_ref, request_ref


def make_query_blob(store: ArtifactStore, values: object) -> ArtifactRef:
    return store.put(
        "zeromodel.core.matrix_blob",
        canonical_dto_bytes(
            MatrixBlob.from_array(
                values, dtype="float64", metadata={"role": "query_representation"}
            )
        ),
        {"role": "query_representation"},
    )
