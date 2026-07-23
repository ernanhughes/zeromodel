# ZeroModel Current Architecture Package Inventory

**Status: current architecture inventory** (not a historical migration snapshot; the nine-package split described here is the present state of `main`, not a plan).

Generator version: `2.0.0`
Baseline commit: `dab3494087c2b65012a2a6e28a6b7a6130b0de82`
Generated (UTC): `2026-01-01T00:00:00Z`

Generated artifacts:

- `docs/architecture/package-module-map-1.0.13.csv`
- `docs/architecture/package-import-graph-1.0.13.json`
- `docs/architecture/package-dependency-findings-1.0.13.md`

## Module Count By Classification

- analysis: 22
- artifacts: 17
- core: 11
- examples: 27
- navigation: 7
- observation: 4
- research: 64
- sqlalchemy: 12
- tooling: 118
- trust: 6
- video: 60
- vision: 3

## Production Packages

Production source is discovered exclusively from the `source_root` entries in `package-boundaries.toml` (the authoritative package configuration), one row per configured package:

- `zeromodel` (`zeromodel.core`) - packages/core/src
- `zeromodel-analysis` (`zeromodel.analysis`) - packages/analysis/src
- `zeromodel-observation` (`zeromodel.observation`) - packages/observation/src
- `zeromodel-vision` (`zeromodel.vision`) - packages/vision/src
- `zeromodel-video` (`zeromodel.video`) - packages/video/src
- `zeromodel-sqlalchemy` (`zeromodel.persistence.sqlalchemy`) - packages/sqlalchemy/src
- `zeromodel-artifacts` (`zeromodel.artifacts`) - packages/artifacts/src
- `zeromodel-trust` (`zeromodel.trust`) - packages/trust/src
- `zeromodel-navigation` (`zeromodel.navigation`) - packages/navigation/src

The historical monolithic root (`package-boundaries.toml`'s `forbidden_roots = ["zeromodel"]`) is not scanned by this script and is never reported as current production implementation.

## Package Build And Data Inventory

Each production package under `packages/*/` ships its own `pyproject.toml`, distribution name, and version. `package-boundaries.toml` declares `release_version = "1.0.13"` as the coordinated release-candidate version across all nine packages; see the individual package manifests under `packages/*/pyproject.toml` for exact per-package dependency declarations. The repository root `pyproject.toml` no longer declares a `[project]` section or builds a distribution of its own; it only holds shared tool configuration (pytest, ruff, mypy) that spans all nine packages via `pythonpath`/`mypy_path` entries under `packages/*/src`.

## Domain Boundary Inventory

The RMDTO target path is Runtime -> Facade -> Engine -> Service -> Store protocol -> Store implementation -> ORM. SQLAlchemy ownership is isolated under `packages/sqlalchemy/src/zeromodel/persistence/sqlalchemy`; video runtime and stores live under `packages/video/src/zeromodel/video` and are expected to stay SQLAlchemy-free at the domain-service layer. Suspicious and forbidden observed edges are ranked in the dependency findings document.

## Architecture Comparison

Allowed target graph (derived from `package-boundaries.toml` `depends_on`): analysis->core; artifacts->core; core->(none); navigation->artifacts,core; observation->core; sqlalchemy->core,video; trust->artifacts,core; video->core,observation; vision->core,observation.

Observed classification graph: `{"analysis": ["core"], "artifacts": ["core"], "navigation": ["artifacts", "core"], "observation": ["core"], "sqlalchemy": ["core", "video"], "trust": ["artifacts", "core"], "video": ["core", "observation"], "vision": ["core", "observation"]}`.

Forbidden observed edge count: `0`.
