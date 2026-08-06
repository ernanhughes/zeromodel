from __future__ import annotations

from zeromodel.core.matrix_blob import MatrixBlob
from zeromodel.navigation.dto import (
    NavigationTileDTO,
    TileCoverageDTO,
    TilePointerDTO,
    TraversalRequestDTO,
    compute_tile_id,
)
from zeromodel.search.dto import RelationSearchRequestDTO
from zeromodel.search.navigation_rule import RelationTraversalRule
from zeromodel.search.persistence import (
    load_relation_readout_aggregate,
    store_matrix_blob,
)
from zeromodel.search.ranking import search_relation
from zeromodel.search.receipts import (
    build_relation_search_receipt,
    replay_relation_search,
)
from zeromodel.search.views import build_relation_search_vpm


def test_readout_aggregate_validates(search_fixture):
    aggregate = load_relation_readout_aggregate(
        search_fixture["readout_ref"], search_fixture["store"]
    )

    assert (
        aggregate.representation_spec.representation_spec_id
        == search_fixture["spec"].representation_spec_id
    )


def test_receipt_replays_exact_result(search_fixture):
    request = RelationSearchRequestDTO(
        readout_ref=search_fixture["readout_ref"],
        corpus_ref=search_fixture["batch_ref"],
        query_representation_blob_ref=search_fixture["query_ref"],
        k=2,
    )
    result, result_ref, request_ref = search_relation(
        store=search_fixture["store"], request=request
    )
    assert result_ref is not None
    receipt, receipt_ref = build_relation_search_receipt(
        store=search_fixture["store"],
        request_ref=request_ref,
        result_ref=result_ref,
    )

    replayed = replay_relation_search(
        store=search_fixture["store"], receipt_ref=receipt_ref
    )

    assert replayed.result_id == result.result_id
    assert receipt.result_id == result.result_id


def test_search_vpm_binds_result(search_fixture):
    request = RelationSearchRequestDTO(
        readout_ref=search_fixture["readout_ref"],
        corpus_ref=search_fixture["batch_ref"],
        query_representation_blob_ref=search_fixture["query_ref"],
        k=2,
    )
    result, _, _ = search_relation(store=search_fixture["store"], request=request)

    vpm = build_relation_search_vpm(result=result, resolver=search_fixture["store"])

    assert vpm.provenance["result_id"] == result.result_id
    assert "relation_closeness" in vpm.source.metric_ids


def test_navigation_rule_selects_nearest_child(search_fixture):
    store = search_fixture["store"]
    child_a_ref = store_matrix_blob(
        store, MatrixBlob.from_array([1.0, 0.0], dtype="float64")
    )
    child_b_ref = store_matrix_blob(
        store, MatrixBlob.from_array([0.0, 0.2], dtype="float64")
    )
    children = (
        TilePointerDTO(
            "leaf", search_fixture["items"][1].artifact_id, child_a_ref.artifact_id
        ),
        TilePointerDTO(
            "leaf", search_fixture["items"][2].artifact_id, child_b_ref.artifact_id
        ),
    )
    coverage = TileCoverageDTO("fixture", "root", child_count=2, leaf_count=2)
    tile = NavigationTileDTO(
        tile_id=compute_tile_id(
            depth=0, coverage=coverage, children=children, tie_rule="relation"
        ),
        depth=0,
        coverage=coverage,
        children=children,
        tie_rule="relation",
    )
    rule = RelationTraversalRule(
        resolver=store,
        readout_ref=search_fixture["readout_ref"],
        representation_spec_ref=search_fixture["spec_ref"],
    )
    request = TraversalRequestDTO(
        request_id="nav",
        attributes=(("query_blob_ref", search_fixture["query_ref"].artifact_id),),
    )

    selection = rule.select_child(request, tile, children)

    assert selection.selected == children[1]
    assert selection.tie_resolution == "distance_then_target_id"
