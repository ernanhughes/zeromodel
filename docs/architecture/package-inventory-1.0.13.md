# ZeroModel 1.0.13 Package Inventory

Baseline commit: `216b5150ad314ffcfccebd809ffdd88ac5231e3b`

Generated artifacts:

- `docs/architecture/package-module-map-1.0.13.csv`
- `docs/architecture/package-import-graph-1.0.13.json`
- `docs/architecture/package-dependency-findings-1.0.13.md`

## Module Count By Classification

- examples: 26
- tooling: 115

## Public Root API

`zeromodel/__init__.py` currently re-exports symbols from core, analysis, observation, vision, video, and research/evidence modules. The approved package architecture removes this compatibility surface instead of preserving aliases. See the CSV `public_symbols` and inbound test/example columns for defining-module and consumer evidence.

## Package Build And Data Inventory

Current `pyproject.toml` discovers `zeromodel*`, ships the monolithic `zeromodel` distribution at version `1.0.12`, declares NumPy as the only base runtime dependency, and puts SQLAlchemy, Torch, TorchVision, Transformers, and Pillow behind optional extras. `tool.pytest.ini_options.pythonpath = ["."]` means tests can rely on repository-root imports that future wheels must not assume.

## Domain Boundary Inventory

The RMDTO target path is Runtime -> Facade -> Engine -> Service -> Store protocol -> Store implementation -> ORM. Current SQLAlchemy ownership is isolated under `zeromodel/db`; `zeromodel/runtime.py` and `zeromodel/stores` are classified as video and should remain SQLAlchemy-free. Suspicious and forbidden proposed edges are ranked in the dependency findings document.

## Split Analysis

| current module | responsibility fragment | target module | target package | symbols to move | inbound callers | identity/schema risk | recommended split order |
|---|---|---|---|---|---|---|---|
| zeromodel | root compatibility re-exports | package-local `__init__.py` files | core/analysis/observation/vision/video/sqlalchemy | all current `__all__` entries | tests and examples using `from zeromodel import ...` | high: root API removal changes import identity | remove root re-exports after package-local public APIs are declared |

## Architecture Comparison

Allowed target graph: analysis->core; observation->core; vision->observation/core; video->observation/core; sqlalchemy->video/core; research->any production package.

Observed proposed classification graph: `{}`.

Forbidden proposed edge count: `0`.
