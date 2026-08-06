from __future__ import annotations

import numpy as np
import pytest

from zeromodel.artifacts import ArtifactRef, InMemoryArtifactStore
from zeromodel.core.matrix_blob import MatrixBlob
from zeromodel.search.dto import (
    RelationContractDTO,
    RelationCoordinateBatchDTO,
    RelationCoordinateSpecDTO,
    RelationFitSpecDTO,
    RepresentationBatchDTO,
    RepresentationSpecDTO,
)
from zeromodel.search.persistence import (
    load_relation_coordinate_batch_aggregate,
    load_representation_batch_aggregate,
    store_matrix_blob,
    store_relation_contract,
    store_relation_coordinate_batch,
    store_representation_batch,
    store_representation_spec,
)
from zeromodel.search.projection import compile_relation_readout


def item_ref(name: str) -> ArtifactRef:
    import hashlib

    return ArtifactRef(
        "example.item", "sha256:" + hashlib.sha256(name.encode()).hexdigest()
    )


@pytest.fixture
def search_fixture():
    store = InMemoryArtifactStore()
    spec = RepresentationSpecDTO(
        provider_id="fixture",
        model_id="linear",
        model_revision="r1",
        dimensions=2,
        dtype="float64",
        pooling_policy="none",
        normalization_policy="none",
        preprocessing_contract_id="raw-vectors",
    )
    spec_ref = store_representation_spec(store, spec)
    items = tuple(item_ref(name) for name in ("query", "a", "b", "c"))
    embeddings = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 0.2], [0.0, 2.0]], dtype=np.float64
    )
    matrix_ref = store_matrix_blob(
        store, MatrixBlob.from_array(embeddings, dtype="float64")
    )
    batch = RepresentationBatchDTO(
        representation_spec_ref=spec_ref,
        matrix_blob_ref=matrix_ref,
        item_refs=items,
        matrix_shape=embeddings.shape,
        matrix_dtype="float64",
    )
    batch_ref = store_representation_batch(store, batch)
    contract = RelationContractDTO(
        relation_id="structure",
        version="1",
        subject_kind="fixture",
        coordinates=(
            RelationCoordinateSpecDTO("x", "first coordinate", "measured-x"),
            RelationCoordinateSpecDTO("y", "second coordinate", "measured-y"),
        ),
    )
    contract_ref = store_relation_contract(store, contract)
    values = embeddings.copy()
    values_ref = store_matrix_blob(
        store, MatrixBlob.from_array(values, dtype="float64")
    )
    coord_batch = RelationCoordinateBatchDTO(
        relation_contract_ref=contract_ref,
        values_blob_ref=values_ref,
        item_refs=items,
        values_shape=values.shape,
        values_dtype="float64",
    )
    coord_ref = store_relation_coordinate_batch(store, coord_batch)
    rep_agg = load_representation_batch_aggregate(batch_ref, store)
    coord_agg = load_relation_coordinate_batch_aggregate(coord_ref, store)
    readout, readout_ref = compile_relation_readout(
        store=store,
        representations=rep_agg,
        coordinates=coord_agg,
        fit_spec=RelationFitSpecDTO(alpha=0.0, minimum_relation_scale=0.1),
    )
    return {
        "store": store,
        "spec": spec,
        "spec_ref": spec_ref,
        "items": items,
        "batch_ref": batch_ref,
        "contract": contract,
        "contract_ref": contract_ref,
        "readout": readout,
        "readout_ref": readout_ref,
        "query_ref": store_matrix_blob(
            store, MatrixBlob.from_array(embeddings[0], dtype="float64")
        ),
    }
