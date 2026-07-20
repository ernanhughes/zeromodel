# Video Action-Set RMDTO Slice

This document describes the RMDTO boundary for the video action-set domain. The
first slice introduced the benchmark identity aggregate; the second slice adds
episode plans and the sealed final split-plan envelope.

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

## Future aggregates

Future slices can add DTOs, Stores, and persistence mappings for:

- observations and MatrixBlob references;
- operation chains;
- verification reports;
- mutation audits.

Those later slices should preserve the same dependency direction and must not
reconstruct the monolith inside a new package.
