# ZeroModel 1.0.13 SQLAlchemy Package Validation

Stage commit: containing commit for the SQLAlchemy isolation stage.

Prior validated package commits:

- Core: `c827a36b6498990f2d7eb10e8ec4fc6a584fb502`
- Analysis: `e3ac0457e06bab9e7ae407ffa1d1fe1acbe9cabd`
- Observation: `0b6a4698633e55e99326488d4dbf77b1c266c560`
- Vision and video closure: `f251ea80d028f73fdd843fcf0ca22b4173b72b08`

## Package Boundary

Distribution: `zeromodel-sqlalchemy==1.0.13`

Namespace: `zeromodel.persistence.sqlalchemy`

Runtime dependencies declared by `packages/sqlalchemy/pyproject.toml`:

- `numpy>=1.23`
- `SQLAlchemy>=2.0,<3`
- `zeromodel==1.0.13`
- `zeromodel-video==1.0.13`

`zeromodel-observation` is not declared directly because the SQLAlchemy package
does not import `zeromodel.observation`. It is installed transitively through
`zeromodel-video==1.0.13` for the clean wheel environment.

The package does not import analysis, vision, research fixtures, examples,
Torch, TorchVision, Transformers, or Pillow.

## Module Inventory

- `zeromodel.persistence.sqlalchemy.__init__`
- `zeromodel.persistence.sqlalchemy.db.__init__`
- `zeromodel.persistence.sqlalchemy.db.orm.__init__`
- `zeromodel.persistence.sqlalchemy.db.orm.base`
- `zeromodel.persistence.sqlalchemy.db.orm.video_action_set`
- `zeromodel.persistence.sqlalchemy.db.runtime`
- `zeromodel.persistence.sqlalchemy.db.session`
- `zeromodel.persistence.sqlalchemy.db.stores.__init__`
- `zeromodel.persistence.sqlalchemy.db.stores.video_action_set`
- `zeromodel.persistence.sqlalchemy.db.stores.video_action_set_observation`
- `zeromodel.persistence.sqlalchemy.video_action_set_final_admin_cli`
- `zeromodel.persistence.sqlalchemy.video_action_set_final_cli`

## Public API

`zeromodel.persistence.sqlalchemy.__all__` exports only:

- `FINALIZATION_SCHEMA_VERSION`
- `SqlAlchemyVideoActionSetStore`
- `build_finalization_sqlite_runtime`
- `build_sqlite_runtime`
- `create_database_engine`
- `create_schema`
- `create_session_factory`
- `initialize_finalization_authority`
- `sqlite_database_url`
- `verify_finalization_authority`

ORM classes and mapper helpers remain importable only from internal
implementation modules and are not part of the top-level public surface.

## ORM Tables

- `video_action_set_benchmark_identity`
- `video_action_set_episode_plan`
- `video_action_set_sealed_split_plan`
- `matrix_blob`
- `video_action_set_finalization_schema`
- `video_action_set_final_evaluation_protocol`
- `video_action_set_final_access_authorization`
- `video_action_set_final_access_record`
- `video_action_set_final_access_event`
- `video_action_set_observation`
- `video_action_set_observation_operation_chain`
- `video_action_set_observation_operation`
- `video_action_set_observation_operation_input`

## Validation Results

Focused package tests:

```powershell
$env:PYTHONPATH='packages/core/src;packages/observation/src;packages/video/src;packages/sqlalchemy/src'
python -m pytest packages/sqlalchemy/tests -q
```

Result: `6 passed`.

Focused quality:

```powershell
python -m ruff check packages/sqlalchemy/src packages/sqlalchemy/tests
python -m ruff format --check packages/sqlalchemy/src packages/sqlalchemy/tests
python -m mypy packages/sqlalchemy/src
```

Results: ruff passed, format passed, mypy passed for 12 source files.

Package boundaries:

```powershell
python scripts/check_package_boundaries.py
```

Result: passed for 112 production modules.

Repository quality gate:

```powershell
python scripts/check_quality.py
```

Result: passed with the governed path set unchanged. No new legacy quality
exceptions were added for SQLAlchemy's extracted legacy Store aggregate.

Wheel builds:

```powershell
python -m build packages/core
python -m twine check packages/core/dist/*
python -m build packages/observation
python -m twine check packages/observation/dist/*
python -m build packages/video
python -m twine check packages/video/dist/*
python -m build packages/sqlalchemy
python -m twine check packages/sqlalchemy/dist/*
```

Result: all four wheels and source distributions built and passed `twine check`.

Clean virtual environment:

```powershell
python -m venv build/sqlalchemy-isolation-venv
build/sqlalchemy-isolation-venv/Scripts/python.exe -m pip install pytest
build/sqlalchemy-isolation-venv/Scripts/python.exe -m pip install packages/core/dist/zeromodel-1.0.13-py3-none-any.whl
build/sqlalchemy-isolation-venv/Scripts/python.exe -m pip install packages/observation/dist/zeromodel_observation-1.0.13-py3-none-any.whl
build/sqlalchemy-isolation-venv/Scripts/python.exe -m pip install packages/video/dist/zeromodel_video-1.0.13-py3-none-any.whl
build/sqlalchemy-isolation-venv/Scripts/python.exe -m pip install packages/sqlalchemy/dist/zeromodel_sqlalchemy-1.0.13-py3-none-any.whl
build/sqlalchemy-isolation-venv/Scripts/python.exe -m pip check
build/sqlalchemy-isolation-venv/Scripts/python.exe -m pytest -q packages/sqlalchemy/tests
```

Results:

- `pip check`: no broken requirements.
- SQLAlchemy installed version: `2.0.51`.
- Installed import paths came from
  `build/sqlalchemy-isolation-venv/Lib/site-packages`.
- Package-local tests from installed wheels: `6 passed`.

Wheel content check:

- Wheel: `packages/sqlalchemy/dist/zeromodel_sqlalchemy-1.0.13-py3-none-any.whl`
- File count: 16
- Every file is under `zeromodel/persistence/sqlalchemy/**` or
  `zeromodel_sqlalchemy-1.0.13.dist-info/**`.

## Behavioral Coverage

The SQLAlchemy package-local tests prove:

- import has no database file, engine, session, or schema side effects;
- schema creation is explicit through `create_schema`;
- SQLite foreign keys are enabled;
- Store methods accept and return video DTOs, not ORM rows;
- SQL Store identity and episode-plan behavior matches the in-memory Store;
- unknown benchmark identity and conflicting plan failures are preserved;
- batch episode-plan writes roll back atomically on conflict;
- MatrixBlob rows are deduplicated by content identity and reject conflicting
  payloads;
- observation operation-chain query filters work through SQL predicates;
- filtered observation listing preserves DTO output;
- file-backed SQLite databases can be reopened by a fresh engine/session;
- tampered sealed split rows are rejected by DTO digest validation;
- runtime composition works with and without explicit schema initialization.

## Historical Test Classification

Historical SQL persistence tests remain repository-level validation fixtures:

- `tests/test_video_identity_sql_store.py`
- `tests/test_video_episode_plan_sql_store.py`
- `tests/test_video_observation_sql_store.py`
- `tests/test_video_final_access_transactions.py`
- `tests/test_video_final_schema_and_scripts.py`
- `tests/integration/test_video_finalization_*.py`

The new package-local test suite is the clean-wheel validation surface for
`zeromodel-sqlalchemy`. Integration-tier persistence tests were not executed.

## Remaining Defect

`python scripts/run_fast_tests.py` currently stops before SQLAlchemy coverage on
`tests/integration/test_video_provider_measurement_real.py`, which imports
`zeromodel.video.domains.video_action_set.provider_measurement`. That module was
reclassified to research in the video package closure and is unrelated to this
SQLAlchemy adapter isolation. This stage does not change that historical test.

## Next Stage

The SQLAlchemy package is independently buildable, installable, importable, and
testable. The next permitted stage is final cross-package integration and release
readiness validation; publication is still out of scope for this stage.
