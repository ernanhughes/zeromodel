from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zeromodel.artifacts import ArtifactRef, ArtifactResolver
from zeromodel.navigation.dto import (
    NavigationTileDTO,
    TilePointerDTO,
    TraversalRequestDTO,
    TraversalRuleDescriptorDTO,
)
from zeromodel.navigation.rules import ChildSelection

from zeromodel.search.persistence import (
    load_matrix_blob,
    load_relation_readout_aggregate,
)
from zeromodel.search.projection import compiled_from_aggregate


@dataclass(frozen=True, slots=True)
class RelationTraversalRule:
    """Navigation rule selecting the child nearest to a projected query vector.

    The traversal request must carry `query_blob_ref` as an artifact id. Child
    `order_key` values must be artifact ids for matrix blobs containing child
    representation vectors. Ties break by stable child target id.
    """

    resolver: ArtifactResolver
    readout_ref: ArtifactRef
    representation_spec_ref: ArtifactRef
    query_attribute: str = "query_blob_ref"

    def descriptor(self) -> TraversalRuleDescriptorDTO:
        return TraversalRuleDescriptorDTO(
            rule_kind="relation_traversal_rule",
            parameters=(
                ("readout_ref", self.readout_ref.artifact_id),
                ("representation_spec_ref", self.representation_spec_ref.artifact_id),
                ("query_attribute", self.query_attribute),
            ),
        )

    def select_child(
        self,
        request: TraversalRequestDTO,
        tile: NavigationTileDTO,
        children: tuple[TilePointerDTO, ...],
    ) -> ChildSelection:
        query_blob_id = request.attributes_map.get(self.query_attribute)
        if not query_blob_id:
            return ChildSelection(None, children, (), "missing_query")
        readout = load_relation_readout_aggregate(self.readout_ref, self.resolver)
        runtime = compiled_from_aggregate(readout)
        query = load_matrix_blob(
            self.resolver,
            ArtifactRef("zeromodel.core.matrix_blob", query_blob_id),
        ).to_array()
        query_vector = np.asarray(
            query[0] if query.ndim == 2 and query.shape[0] == 1 else query,
            dtype=np.float64,
        )
        query_projected = runtime.project_one(
            query_vector,
            representation_spec_id=readout.representation_spec.representation_spec_id,
        )
        scored = []
        for child in children:
            child_blob = load_matrix_blob(
                self.resolver,
                ArtifactRef("zeromodel.core.matrix_blob", child.order_key),
            ).to_array()
            child_vector = np.asarray(
                child_blob[0]
                if child_blob.ndim == 2 and child_blob.shape[0] == 1
                else child_blob,
                dtype=np.float64,
            )
            child_projected = runtime.project_one(
                child_vector,
                representation_spec_id=readout.representation_spec.representation_spec_id,
            )
            distance = float(np.max(np.abs(child_projected - query_projected)))
            scored.append((distance, child.target_id, child))
        if not scored:
            return ChildSelection(None, children, (), "no_children")
        scored.sort(key=lambda item: (item[0], item[1]))
        best_distance = scored[0][0]
        tied = tuple(item[2].target_id for item in scored if item[0] == best_distance)
        return ChildSelection(
            selected=scored[0][2],
            eligible=children,
            tie_candidates=tied if len(tied) > 1 else (),
            tie_resolution="distance_then_target_id",
        )
