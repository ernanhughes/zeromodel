# zeromodel-search

`zeromodel-search` compiles and applies deterministic relation-specific
readouts over declared frozen representations. Given measured relation
coordinates, a fitted readout can rank identified candidates in the declared
relation space and produce an inspectable, replayable result.

Search does not generate embeddings. It consumes representations governed by a
declared representation contract.

Search does not decide whether a relation is scientifically useful. It compiles
and applies a readout for a relation the caller has declared and measured.

## Install

```powershell
python -m pip install zeromodel==1.2.0 zeromodel-artifacts==1.2.0 zeromodel-navigation==1.2.0 zeromodel-search==1.2.0
```

## Architecture

```text
frozen representation batch
        +
declared relation coordinate batch
        +
ridge fit spec
        ↓
RelationReadoutArtifactDTO
        ↓
exact relation ranking over identified corpus rows
        ↓
RelationSearchResultDTO
        ↓
VPM inspection + replay receipt
```

The package stores DTOs through `zeromodel.artifacts`. Numeric arrays are
stored as `zeromodel.core.MatrixBlob` artifacts. The public result exposes
stable `ArtifactRef`s, not matrix row indexes.

## Minimal Example

```python
import numpy as np

from zeromodel.artifacts import InMemoryArtifactStore
from zeromodel.search import RelationFitSpecDTO, compile_relation_readout, search_relation

store = InMemoryArtifactStore()

# Build and store RepresentationSpecDTO, RepresentationBatchDTO,
# RelationContractDTO, and RelationCoordinateBatchDTO, then resolve them as
# aggregates through zeromodel.search.persistence helpers.
representations = ...
coordinates = ...
readout, readout_ref = compile_relation_readout(
    store=store,
    representations=representations,
    coordinates=coordinates,
    fit_spec=RelationFitSpecDTO(alpha=1.0),
)

request = ...
result, result_ref, request_ref = search_relation(store=store, request=request)
```

## Representation Identity

Representation compatibility is stronger than matching dimensions. Provider,
model, revision, dtype, pooling policy, normalization policy, and preprocessing
contract are identity-bearing. Two 768-dimensional encoders are incompatible
unless their declared representation identities match.

## Relation Contracts

A relation contract declares coordinate order, measurement contracts, scaling,
and distance. Version one supports robust median/IQR scaling and Chebyshev
distance only. Coordinate order is semantic: reordering coordinates changes the
relation identity.

## Readout Compilation

Compilation fits independent deterministic ridge projections from frozen
embeddings into robust-scaled relation coordinates. The persisted readout binds
the representation contract, relation contract, fit spec, training batches, and
parameter `MatrixBlob`s. Loading validates aggregate closure and never refits or
repairs corrupted state.

## Relation Versus Cosine Ranking

Exact search ranks by relation distance first. If cosine comparison is enabled,
cosine is diagnostic and breaks only relation-distance ties:

```text
1. relation distance ascending
2. optional cosine distance ascending
3. stable artifact identity ascending
```

The returned ranking coordinates are robust-scaled predicted relation
coordinates. Natural-unit restoration is available from the compiled runtime,
but result ordering is defined in scaled space.

## VPM Inspection

`build_relation_search_vpm()` materializes a VPM with relation closeness,
relation distance, and per-coordinate absolute mismatch. Raw distances remain
visible; `relation_closeness = 1 / (1 + distance)` is included for the
larger-is-stronger VPM view.

## Receipts And Replay

`build_relation_search_receipt()` binds the persisted request, result, readout,
corpus, and required closure checks. `replay_relation_search()` re-executes the
exact search and fails if the result identity diverges.

A valid Search receipt proves deterministic execution and aggregate closure. It
does not prove that the relation labels, encoder, or downstream use are correct.

## Navigation Integration

`RelationTraversalRule` implements Navigation's existing `TraversalRule`
protocol. Navigation does not import Search. The adapter selects the child whose
declared representation blob is nearest to the projected query vector, with
ties resolved by stable child target id.

## Candidate Generation

Version one is exact search only. Candidate filtering and approximate nearest
neighbour indexing are not implemented and are not claimed. There is no
validated recall guarantee under cosine candidate filtering.

## Scientific Boundary

The implementation operationalises the RELATE mechanism. It does not rerun,
reinterpret, or replace the preserved RELATE benchmark evidence.

## Explicit Exclusions

This package does not claim universal semantic search, universal improvement
over cosine, automatic relation discovery, causal reasoning, correctness of
relation labels, correctness of upstream embeddings, calibrated confidence,
safe autonomous retrieval, scalable approximate nearest-neighbour performance,
general code understanding, or general RAG improvement.

