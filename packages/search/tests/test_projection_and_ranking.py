from __future__ import annotations

import numpy as np
import pytest

from zeromodel.core.matrix_blob import MatrixBlob
from zeromodel.search.dto import RelationSearchRequestDTO
from zeromodel.search.errors import RepresentationMismatchError, SearchValidationError
from zeromodel.search.persistence import (
    load_relation_readout_aggregate,
    store_matrix_blob,
)
from zeromodel.search.projection import compiled_from_aggregate
from zeromodel.search.ranking import search_relation


def test_projection_rejects_wrong_representation_identity(search_fixture):
    aggregate = load_relation_readout_aggregate(
        search_fixture["readout_ref"], search_fixture["store"]
    )
    runtime = compiled_from_aggregate(aggregate)

    with pytest.raises(RepresentationMismatchError):
        runtime.project_one([0.0, 1.0], representation_spec_id="different")


def test_exact_search_relation_overrides_cosine_and_reports_dominant_coordinate(
    search_fixture,
):
    store = search_fixture["store"]
    request = RelationSearchRequestDTO(
        readout_ref=search_fixture["readout_ref"],
        corpus_ref=search_fixture["batch_ref"],
        query_artifact_ref=search_fixture["items"][0],
        query_representation_blob_ref=search_fixture["query_ref"],
        k=3,
        exclude_refs=(search_fixture["items"][0],),
        include_cosine_comparison=True,
    )

    result, result_ref, request_ref = search_relation(store=store, request=request)

    assert result_ref is not None
    assert request_ref.artifact_id == result.request_ref.artifact_id
    assert [hit.rank for hit in result.hits] == [1, 2, 3]
    assert result.hits[0].artifact_ref == search_fixture["items"][2]
    assert result.hits[0].cosine_rank != 1
    assert result.hits[0].rank_shift is not None
    assert any(hit.dominant_coordinate_id == "y" for hit in result.hits)


def test_empty_corpus_after_exclusions_returns_empty_result(search_fixture):
    request = RelationSearchRequestDTO(
        readout_ref=search_fixture["readout_ref"],
        corpus_ref=search_fixture["batch_ref"],
        query_representation_blob_ref=search_fixture["query_ref"],
        k=10,
        exclude_refs=search_fixture["items"],
    )

    result, _, _ = search_relation(store=search_fixture["store"], request=request)

    assert result.total_candidates == 4
    assert result.evaluated_candidates == 0
    assert result.hits == ()


def test_query_nan_is_rejected(search_fixture):
    bad_ref = store_matrix_blob(
        search_fixture["store"],
        MatrixBlob.from_array(
            np.asarray([0.0, 1.0], dtype=np.float64), dtype="float64"
        ),
    )
    request = RelationSearchRequestDTO(
        readout_ref=search_fixture["readout_ref"],
        corpus_ref=search_fixture["batch_ref"],
        query_representation_blob_ref=bad_ref,
        k=1,
    )
    # MatrixBlob itself rejects NaN, so use a wrong-shaped finite query for runtime validation.
    wrong_shape = store_matrix_blob(
        search_fixture["store"],
        MatrixBlob.from_array(
            np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64), dtype="float64"
        ),
    )
    request = RelationSearchRequestDTO(
        readout_ref=request.readout_ref,
        corpus_ref=request.corpus_ref,
        query_representation_blob_ref=wrong_shape,
        k=1,
    )
    with pytest.raises(SearchValidationError):
        search_relation(store=search_fixture["store"], request=request)
