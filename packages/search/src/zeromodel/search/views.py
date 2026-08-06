from __future__ import annotations

from zeromodel.core import LayoutRecipe, ScoreTable, VPMArtifact, build_vpm

from zeromodel.search.dto import RelationSearchResultDTO
from zeromodel.search.persistence import load_relation_contract
from zeromodel.artifacts import ArtifactResolver


def build_relation_search_vpm(
    *,
    result: RelationSearchResultDTO,
    resolver: ArtifactResolver,
) -> VPMArtifact:
    contract = load_relation_contract(resolver, result.relation_contract_ref)
    coordinate_ids = contract.coordinate_ids
    metric_ids = (
        "relation_closeness",
        "relation_distance",
        *[f"abs_delta:{coord}" for coord in coordinate_ids],
    )
    values = []
    row_ids = []
    for hit in result.hits:
        row_ids.append(hit.artifact_ref.artifact_id)
        values.append(
            [
                1.0 / (1.0 + hit.relation_distance),
                hit.relation_distance,
                *[abs(value) for value in hit.coordinate_deltas],
            ]
        )
    if not values:
        values = [[0.0 for _ in metric_ids]]
        row_ids = ["empty-result"]
    table = ScoreTable(
        values=values,
        row_ids=row_ids,
        metric_ids=metric_ids,
        metadata={
            "kind": "relation_search_result",
            "result_id": result.result_id,
            "distance_metric": "raw distances are shown directly; relation_closeness is the larger-is-stronger companion metric",
        },
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "relation-search-nearest-first",
            "row_order": {
                "kind": "lexicographic",
                "keys": [{"metric_id": "relation_closeness", "direction": "desc"}],
                "tie_break": "row_id",
            },
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(
        table,
        recipe,
        provenance={
            "kind": "relation_search_vpm",
            "result_id": result.result_id,
            "request_ref": result.request_ref.artifact_id,
            "readout_ref": result.readout_ref.artifact_id,
            "corpus_ref": result.corpus_ref.artifact_id,
            "parents": [result.result_id],
        },
    )
