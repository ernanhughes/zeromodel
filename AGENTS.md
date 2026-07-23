# ZeroModel Agent Guide

## Purpose

ZeroModel is a small Python package for building deterministic Visual Policy Map (VPM) artifacts from scored tables, then consuming those artifacts through views, spatial transforms, comparisons, gates, and higher-level assessment modules.

The package is split into nine namespace-package distributions, each under its
own `packages/<key>/` directory with its own `pyproject.toml`, `src/`, and
`tests/`. `package-boundaries.toml` at the repo root is the authoritative,
machine-readable source for package names, namespaces, source roots, and
allowed internal dependency edges — treat it as ground truth over any prose
description, including this one.

Keep the core artifact kernel conservative. Most new behavior should be added
in consumer packages (analysis, observation, vision, video) first, not by
widening `zeromodel.core`'s contract.

## Repo Map

- `packages/core/src/zeromodel/core`: `ScoreTable`, `LayoutRecipe`, `VPMArtifact`, normalization, ordering, artifact identity, `VPMPolicyLookup`. No internal dependencies.
- `packages/analysis/src/zeromodel/analysis`: views, spatial/manifold logic, learning/training/critic projections, PHOS gates, tracker-export adapters. Depends on core.
- `packages/observation/src/zeromodel/observation`: observation DTOs, visual address contracts, deployment bindings. Depends on core.
- `packages/vision/src/zeromodel/vision`: deterministic visual index and visual policy reader. Depends on core, observation.
- `packages/video/src/zeromodel/video`: video policy, arcade fixture, video action-set RMDTO (Runtime/Facade/Engine/Service/Store) domain. Depends on core, observation.
- `packages/sqlalchemy/src/zeromodel/persistence/sqlalchemy`: explicit SQLAlchemy persistence runtime and Store implementations. Depends on core, video.
- `packages/artifacts/src/zeromodel/artifacts`: content-addressed artifact storage and reference identity; storage/content-reference authority. Depends on core.
- `packages/trust/src/zeromodel/trust`: signature envelopes, trust receipts, revocation. Depends on core, artifacts. Leaf package — does not depend on navigation.
- `packages/navigation/src/zeromodel/navigation`: artifact navigation over the artifacts store. Depends on core, artifacts. Leaf package — does not depend on trust. Search is not yet implemented here.
- `tests/`: repository-wide and cross-package contract tests, run alongside each package's own `packages/*/tests/`.
- `integration_tests/`: cross-package production behavior requiring multiple distributions together.
- `research/`: benchmark, evidence, and unpromoted experimental machinery; excluded from the production package contract and from the fast/quality gates.
- `docs/examples/`: usage-oriented examples that usually map cleanly to tests.
- `docs/research/`: hypothesis and framing docs; do not treat these as implementation proof.

## Core Invariants

- `ScoreTable` values must remain finite, rectangular, and aligned to stable `row_ids` and `metric_ids`.
- `LayoutRecipe` is explicit. Ordering and normalization should stay deterministic.
- `VPMArtifact.artifact_id` is content-derived. Any semantic change to artifact payload changes identity.
- Consumers can add provenance, derived summaries, or alternative orderings, but should not mutate source evidence.
- When adding new public APIs, prefer exposing them through the owning package's `__init__.py` only after tests exist. There is no root `zeromodel/__init__.py` compatibility surface to fall back on — it has been removed, and `package-boundaries.toml`'s `forbidden_roots` keeps it that way.

## Review Priorities

When reviewing or extending this repo, check these first:

1. Package/version consistency between each package's `packages/<key>/pyproject.toml`, `package-boundaries.toml`, README, and release docs.
2. Public flags that look configurable but do not change behavior.
3. Provenance and artifact identity stability after serialization or derived-view construction.
4. Whether README claims are backed by tests or by committed example fixtures.

## Fast Commands

```powershell
python scripts/run_fast_tests.py
pytest packages/core/tests/test_artifact_kernel.py -q
pytest packages/analysis/tests/test_spatial.py packages/analysis/tests/test_manifold.py -q
python -m build packages/core
```

## Test Execution Policy

ZeroModel has four execution tiers.

### Fast tests

The default repository validation command is:

```powershell
python scripts/run_fast_tests.py
```

The complete fast suite has a hard 120-second budget (`FAST_SUITE_BUDGET_SECONDS` in `scripts/run_fast_tests.py`) and runs `tests/`, `integration_tests/`, and every `packages/*/tests/` directory with the marker expression `not slow and not external and not research`.

During implementation:

1. Run only directly affected fast tests.
2. Run the complete fast suite once after implementation.
3. Do not repeatedly rerun an unchanged command.
4. Do not add exhaustive, end-to-end, materialization, or complete mutation work to the fast tier.

### Integration and slow tests

Integration tests and slow tests require explicit human authorization for the
exact command being run.

Do not run any of these unless the user explicitly requests it:

```powershell
python -m pytest -q --run-integration -m integration
python -m pytest -q --run-slow -m slow
```

Tests marked with both tiers require both opt-in flags.

Integration tests include complete benchmark materialization, exhaustive observation universes, complete mutation audits, installed-wheel tests, historical regeneration, and long-running cross-product validation.

When integration coverage is relevant, report the exact suggested command but do not execute it.

### Scientific/manual checks

Scientific and manual checks include development, calibration, selection, or
final split builds, runtime profiling, provider-equivalence audits, canonical
provider audits, evidence-completeness checks over preserved outputs, reference
closure, and mutation audits.

Coding agents must not execute integration tests, slow tests, scientific builds,
benchmarks, or mutation audits unless the user explicitly authorizes that exact
execution. Agents should implement the code and provide operator commands or
scripts for long-running validation. For multi-step long validation, create or
update a script under `scripts/` and leave execution to the operator.

## Video Action-Set RMDTO Policy

- Domain Services never import ORM or SQLAlchemy.
- Store protocols use DTOs only.
- Store implementations own ORM-to-DTO and DTO-to-ORM mapping.
- ORM objects never leave Store implementations.
- Core Runtime uses the in-memory Store by default.
- Database Runtime construction is explicit.
- Do not add database creation at import time.
- Do not combine structural extraction with scientific behavior changes.
- Do not add new legacy quality exceptions for extracted modules.
- New durable aggregates require DTO, Store protocol, in-memory Store, SQL Store, and focused tests.
- Nested values become separate ORM tables only when they require independent query, lifecycle, integrity, or deduplication.
- Canonical JSON payloads must be validated through DTO reconstruction when read from persistence.
- Stores must not repair or normalize conflicting scientific records.
- Batch Store writes must be atomic.
- A behavior-preserving extraction must reduce the corresponding monolith quality ceiling.
- Integration-tier persistence tests may be authored but must not be executed without explicit human authorization.
- Durable benchmark runs prefer SQLite Runtime construction.
- Durable exports should be projections of Store-returned DTOs.
- Large binary arrays must use MatrixBlob and content-addressed deduplication.
- MatrixBlob identity metadata must describe the payload, not the owning frame, episode, split, or sequence.
- Provenance and operation-chain queries belong in relational Store predicates and joins.
- Scientific rendering, transformation, mutation, replay, and digest computation stay in Python/NumPy.
- Legacy final split materialization remains prohibited in every Store. The only
  permitted final observation write path is the explicit final-access service
  path with a durable access record in `running` state.
- Batch observation writes must be atomic.
- Progress observer exceptions intentionally propagate during split builds.
  Before such a failure, SQLite episode plans plus observations, matrix blobs,
  and observation operation chains may already be durable; split JSONL,
  split-manifest, and family-closure artifacts are not completion markers until
  all required files for the split exist. Same-directory retry is allowed only
  for unchanged code and inputs before current-split filesystem completion
  artifacts exist; otherwise require a fresh output directory.

## Working Style

- Prefer additive changes over renaming existing public APIs.
- Add focused tests beside the affected module.
- If behavior is intentionally constrained by the artifact contract, document that constraint instead of adding misleading options.

## Final Action-Set Agent Boundary

- Agents may edit final-access DTOs, stores, CLI preflight, scripts, templates,
  and synthetic tests.
- Agents must not approve a final protocol, create a live final authorization,
  run final execution, materialize the final split, or choose scientific
  thresholds/providers/operating points.
- Preflight is read-only and must not reserve an authorization or access final
  observations.
- Reservation is irreversible. Retry, resume, overwrite, force, alternate-plan,
  provider, threshold, and operating-point switches are not supported for final
  execution.
- Final execution uses a fresh schema-marked finalization database. Never run it
  against, migrate, or relabel a historical Stage 8 database.
- A completed ledger state is not a receipt. Claims require the canonical receipt
  published after artifact promotion, terminal completion, reconstruction, and
  receipt-bound digest revalidation.
- Possession of writable finalization-database access, an approved authorization
  and protocol, and a registered executor is authority within the local trust
  scope. Digest, event-chain, and compare-and-swap checks remain mandatory.
