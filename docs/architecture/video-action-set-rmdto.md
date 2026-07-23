# Video Action-Set RMDTO Slice

This document describes the RMDTO boundary for the video action-set domain. The
first slice introduced the benchmark identity aggregate; the second slice adds
episode plans and the sealed final split-plan envelope; the third slice adds the
database-first observation, MatrixBlob, and provenance ledger.

The dependency path is:

```text
Runtime -> Facade -> Engine -> Service -> Store -> ORM -> Database
```

DTOs are the only application objects allowed to cross Store boundaries.

## Responsibilities

- Runtime composes Facades, Engines, Services, and Stores.
- Facade exposes domain capabilities and delegates to the Engine.
- Engine coordinates Services and contains no persistence, parsing, hashing, or ORM logic.
- Service owns identity document parsing and Store calls.
- Store protocol is DTO-only and defines persistence semantics.
- Store implementations own ORM-to-DTO and DTO-to-ORM mapping.
- ORM classes represent database rows only.

## Allowed imports

- Core Runtime may import domain layers and the in-memory Store.
- Domain Services may import DTOs, pure contracts, Store protocols, standard library helpers, and domain validation errors.
- SQL Store implementations may import DTOs, Store protocols, SQLAlchemy sessions, and ORM classes.
- ORM modules may import SQLAlchemy mapping primitives and local ORM base classes.

## Forbidden imports

- Domain modules must not import SQLAlchemy, ORM modules, SQL Stores, or database sessions.
- Core Runtime must not import `zeromodel.db` or SQLAlchemy.
- ORM modules must not import Services, Engines, Facades, Runtime, or legacy benchmark modules.
- ORM objects must never leave Store implementations.
- Database creation must not happen at import time.

## Store composition

`build_runtime()` uses `InMemoryVideoActionSetStore` by default. This keeps core
ZeroModel SQLAlchemy-free and suitable for the existing minimal dependency
surface.

SQLite persistence is explicit:

```powershell
pip install -e .[persistence]
```

Use `zeromodel.db.runtime.build_sqlite_runtime(...)` when persistence is needed.
Schema creation happens only when `initialize_schema=True` or when callers invoke
the schema helper directly.

## Compatibility strategy

The legacy `zeromodel.video_action_set_benchmark` module keeps exporting
`BenchmarkIdentity`, `load_identity`, and the identity-owned constants. Its
`load_identity()` function delegates through Runtime, Facade, Engine, Service,
and Store, while downstream monolith functions continue accepting the same
immutable identity shape.

## ORM scope

The benchmark identity has six persisted fields. The Store persists those fields
directly rather than storing a JSON copy of the DTO. Not every nested value in
future aggregates should become an ORM table; relational structure should be
introduced only where it supports query, integrity, or lifecycle needs.

## Episode-Plan Aggregate

The aggregate ownership chain is:

```text
BenchmarkIdentity
    -> EpisodePlan
    -> SealedFinalSplitPlan
```

An episode plan cannot be stored until its benchmark identity exists. The root
seed digest in the plan seed lineage is persistence metadata derived from the
plan contract, so the Store uses it to enforce ownership without adding a new
serialized plan field.

Individual episode plans have queryable ORM columns for seed digest, split,
ordinal, family, disposition, source rows, derived seed identity, frame count,
and plan digest. Those fields support deterministic listing and integrity
checks. Nested frame plans, seed lineage, family contracts, and family
interventions remain canonical JSON because they do not yet have independent
query, lifecycle, integrity, or deduplication requirements.

The sealed final split-plan row stores only the envelope metadata, counts, ID
manifest, and sealed digest. It does not duplicate every episode payload. The
SQL Store reconstructs the sealed DTO by reading the envelope row and episode
rows for the same seed and split, ordered by ordinal and episode ID, then
validating the reconstructed DTO digest.

The legacy generator remains authoritative for the scientific plan derivation in
this slice. It still chooses source rows, derives seeds, constructs family
interventions, plans frames, and validates deterministic regeneration. The RMDTO
boundary records and validates the deterministic plan contract so those concerns
can be extracted later behind the same Store and Runtime surface.

Final materialization remains prohibited. The sealed final split-plan is a
plan-only artifact; persistence does not make final observations renderable,
scorable, or materialized.

Persistence records the deterministic plan contract; it does not become the
authority that invents or repairs scientific plans.

## Observation Ledger Aggregate

The durable observation ownership chain is:

```text
BenchmarkIdentity
    -> EpisodePlan
    -> Observation
    -> ObservationOperationChain
    -> ObservationOperation
```

`MatrixBlob` stores canonical binary frame payloads separately from observation
metadata. Identical pixel arrays deduplicate by `MatrixBlob.blob_id`; frame
ownership fields such as split, episode ID, frame ID, sequence number, family,
and expected action are observation fields and must not be included in
MatrixBlob identity-bearing metadata. Benchmark frame blobs use payload metadata
only, including the frame-pixel kind and historical pixel digest.

`MatrixBlob` predates the RMDTO package but already satisfies the immutable DTO
contract. It is the only Store-boundary exception to the `*DTO` naming pattern,
and Stores may accept or return it directly.

The observation ledger preserves three distinct identities:

- `observation_pixel_digest`: historical raw contiguous pixel-byte digest.
- provider raw digest: `ImageObservation` domain-separated digest including shape.
- `matrix_blob_id`: MatrixBlob identity over version, dtype, shape, metadata,
  quantization, and canonical bytes.

Operation chains are stored as ordered relational provenance rows. Operation
parameters remain canonical JSON, while operation names, versions, input
digests, output digests, parameter digests, operation digests, and chain digests
are queryable. SQL Stores apply split, episode, family, event type,
denominator-class, pixel-presence, operation, input-digest, and output-digest
filters in SQL predicates and joins, not by loading the table and filtering in
Python.

Batch observation writes are atomic. Stores validate existing benchmark identity
and episode-plan ownership, MatrixBlob identity, observation identity, sequence
uniqueness, pixel digest, provider descriptor digest, and operation-chain digest
before committing. Stores do not repair conflicting scientific records.

Durable benchmark paths use SQLite as the preferred substrate and write public
JSON and JSONL artifacts as projections of Store-returned DTOs. Compatibility
exports keep the legacy observation record keys and do not add MatrixBlob IDs.
Final split materialization remains prohibited: the final split is plan-only,
and Stores must not persist final observations or final MatrixBlob rows.

Python and NumPy remain authoritative for rendering, pixel transformations,
splicing, corruption, temporal mutation, operation replay, digest computation,
and scientific validation. SQLite is authoritative for durable identity,
relationships, ordering, deduplication, transactionality, provenance traversal,
and audit retrieval.

Use the database to organize, relate, constrain, and retrieve scientific
evidence. Do not use it to invent, normalize, or repair scientific evidence.

## Future aggregates

Future slices can add DTOs, Stores, and persistence mappings for:

- verification reports;
- mutation audits.

Those later slices should preserve the same dependency direction and must not
reconstruct the monolith inside a new package.

**Provider evaluation and policy-impact verification** (implemented, Stage 2D)
is the first such slice: it turns one external-provider evaluation run into an
immutable, database-backed aggregate that references existing observations by
`frame_id` rather than duplicating pixels or provenance. See
[provider-evaluation-rmdto.md](provider-evaluation-rmdto.md) for its aggregate
ownership, identity model, and Store semantics.
