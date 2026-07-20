# ZeroModel Agent Guide

## Purpose

ZeroModel is a small Python package for building deterministic Visual Policy Map (VPM) artifacts from scored tables, then consuming those artifacts through views, spatial transforms, comparisons, gates, and higher-level assessment modules.

The package is intentionally split into:

- `zeromodel.artifact`: core immutable artifact kernel and identity rules.
- `zeromodel.*` consumers: views, spatial/manifold logic, learning/training/critic projections, rendering, bundling, composition, and controllers.

Keep the artifact kernel conservative. Most new behavior should be added in consumer modules first, not by widening the core artifact contract.

## Repo Map

- `zeromodel/__init__.py`: public import surface.
- `zeromodel/artifact.py`: `ScoreTable`, `LayoutRecipe`, `VPMArtifact`, normalization, ordering, artifact identity.
- `zeromodel/views.py`: dense-table policy lenses via `ViewProfile` and `build_view`.
- `zeromodel/spatial.py`: top-left mass optimization and optimized views.
- `zeromodel/manifold.py`: temporal series over optimized panels.
- `zeromodel/learning.py`: before/after/held-out/regression learning evidence.
- `zeromodel/training.py`: checkpoint-level training telemetry assessment.
- `zeromodel/critic.py`: critic/evidence/policy risk assessment.
- `zeromodel/compose.py`, `compare.py`, `hierarchy.py`, `edge.py`, `controller.py`: downstream consumers over VPM fields/artifacts.
- `zeromodel/render.py`: dependency-light PNG/SVG output.
- `zeromodel/bundle.py`: `.vpm` bundle serialization.
- `zeromodel/metrics.py`: metric alias packing and `ScoreTable` helpers.
- `zeromodel/adapters/`: tracker export adapters.
- `tests/`: module-level contract tests plus a broad capability smoke test.
- `docs/examples/`: usage-oriented examples that usually map cleanly to tests.
- `docs/research/`: hypothesis and framing docs; do not treat these as implementation proof.

## Core Invariants

- `ScoreTable` values must remain finite, rectangular, and aligned to stable `row_ids` and `metric_ids`.
- `LayoutRecipe` is explicit. Ordering and normalization should stay deterministic.
- `VPMArtifact.artifact_id` is content-derived. Any semantic change to artifact payload changes identity.
- Consumers can add provenance, derived summaries, or alternative orderings, but should not mutate source evidence.
- When adding new public APIs, prefer exposing them through `zeromodel/__init__.py` only after tests exist.

## Review Priorities

When reviewing or extending this repo, check these first:

1. Packaging/version consistency between `pyproject.toml`, `zeromodel/__init__.py`, README, and release docs.
2. Public flags that look configurable but do not change behavior.
3. Provenance and artifact identity stability after serialization or derived-view construction.
4. Whether README claims are backed by tests or by committed example fixtures.

## Fast Commands

```powershell
python scripts/run_fast_tests.py
pytest tests/test_artifact_kernel.py -q
pytest tests/test_views.py tests/test_spatial.py tests/test_manifold.py -q
python -m build
```

## Test Execution Policy

ZeroModel has two test tiers.

### Fast tests

The default repository validation command is:

```powershell
python scripts/run_fast_tests.py
```

The complete fast suite has a hard 60-second budget.

During implementation:

1. Run only directly affected fast tests.
2. Run the complete fast suite once after implementation.
3. Do not repeatedly rerun an unchanged command.
4. Do not add exhaustive, end-to-end, materialization, or complete mutation work to the fast tier.

### Integration tests

Integration tests require explicit human authorization.

Do not run any of these unless the user explicitly requests it:

```powershell
pytest --run-integration
pytest --run-slow
pytest -m integration
```

Integration tests include complete benchmark materialization, exhaustive observation universes, complete mutation audits, installed-wheel tests, historical regeneration, and long-running cross-product validation.

When integration coverage is relevant, report the exact suggested command but do not execute it.

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

## Working Style

- Prefer additive changes over renaming existing public APIs.
- Add focused tests beside the affected module.
- If behavior is intentionally constrained by the artifact contract, document that constraint instead of adding misleading options.
