from __future__ import annotations

from zeromodel.search.dto import (
    RelationContractDTO,
    RelationCoordinateBatchDTO,
    RelationCoordinateSpecDTO,
    RelationFitSpecDTO,
    RelationReadoutArtifactDTO,
    RelationSearchHitDTO,
    RelationSearchReceiptDTO,
    RelationSearchRequestDTO,
    RelationSearchResultDTO,
    RepresentationBatchDTO,
    RepresentationSpecDTO,
)
from zeromodel.search.errors import (
    CandidateGenerationError,
    RelationContractMismatchError,
    RelationReadoutIntegrityError,
    RepresentationMismatchError,
    SearchReplayMismatchError,
    SearchValidationError,
)
from zeromodel.search.navigation_rule import RelationTraversalRule
from zeromodel.search.persistence import (
    load_relation_readout_aggregate,
    validate_relation_readout_aggregate,
)
from zeromodel.search.projection import (
    CompiledRelationReadout,
    compile_relation_readout,
)
from zeromodel.search.ranking import search_relation
from zeromodel.search.receipts import (
    build_relation_search_receipt,
    replay_relation_search,
)
from zeromodel.search.views import build_relation_search_vpm

SEARCH_PACKAGE_VERSION = "1.2.0"

__all__ = [
    "SEARCH_PACKAGE_VERSION",
    "CandidateGenerationError",
    "CompiledRelationReadout",
    "RelationContractDTO",
    "RelationContractMismatchError",
    "RelationCoordinateBatchDTO",
    "RelationCoordinateSpecDTO",
    "RelationFitSpecDTO",
    "RelationReadoutArtifactDTO",
    "RelationReadoutIntegrityError",
    "RelationSearchHitDTO",
    "RelationSearchReceiptDTO",
    "RelationSearchRequestDTO",
    "RelationSearchResultDTO",
    "RelationTraversalRule",
    "RepresentationBatchDTO",
    "RepresentationMismatchError",
    "RepresentationSpecDTO",
    "SearchReplayMismatchError",
    "SearchValidationError",
    "build_relation_search_receipt",
    "build_relation_search_vpm",
    "compile_relation_readout",
    "load_relation_readout_aggregate",
    "replay_relation_search",
    "search_relation",
    "validate_relation_readout_aggregate",
]
