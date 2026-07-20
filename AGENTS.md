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
python scripts/check_quality.py
pytest -q
pytest tests/test_artifact_kernel.py -q
pytest tests/test_views.py tests/test_spatial.py tests/test_manifold.py -q
python -m build
```

## Code Quality Policy

Codex and other agents must follow these rules:

1. Run `python scripts/check_quality.py` for quality validation.
2. Do not auto-format the entire repository.
3. Do not increase a legacy exception ceiling.
4. New modules must satisfy the hard limits.
5. Do not combine structural refactors with behavioral changes.
6. Do not run integration tests without explicit human authorization.
7. During refactors, preserve public imports and deterministic outputs.
8. Run the bounded fast suite only once after implementation.
9. Do not repeatedly rerun unchanged commands.
10. Report any integration command that a human may run later, but do not execute it.

## Working Style

- Prefer additive changes over renaming existing public APIs.
- Add focused tests beside the affected module.
- If behavior is intentionally constrained by the artifact contract, document that constraint instead of adding misleading options.
