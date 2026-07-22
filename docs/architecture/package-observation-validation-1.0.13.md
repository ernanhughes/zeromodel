# ZeroModel 1.0.13 Observation Package Validation

## Scope

This report covers only the `packages/observation` validation stage. It does not
validate vision, video, SQLAlchemy, or cross-package integration.

## Commits

- Validated core commit:
  `c827a36b6498990f2d7eb10e8ec4fc6a584fb502`
- Validated analysis commit:
  `e3ac0457e06bab9e7ae407ffa1d1fe1acbe9cabd`
- Observation validation implementation commit: recorded by the commit
  containing this report.

## Observation Modules

The observation wheel owns these modules under the implicit `zeromodel`
namespace:

- `zeromodel.observation.__init__`
- `zeromodel.observation.deployment_binding`
- `zeromodel.observation.visual_address`
- `zeromodel.observation.visual_address_manifest`

## Public API

`zeromodel.observation.__all__` is explicit and provider-neutral:

- `DEPLOYMENT_BINDING_VERSION`
- `DeploymentBinding`
- `IMAGE_OBSERVATION_VERSION`
- `ImageObservation`
- `PrototypeBinding`
- `VISUAL_ADDRESS_CONTRACT_VERSION`
- `VISUAL_ADDRESS_DECISION_VERSION`
- `VISUAL_ADDRESS_MANIFEST_VERSION`
- `VisualAddressContract`
- `VisualAddressDecision`
- `VisualAddressManifest`
- `VisualAddressProvider`

The API does not re-export core objects, concrete vision readers, video policy
readers, SQL persistence objects, or former root-API compatibility aliases.

## Runtime Dependencies

The observation runtime dependency set is:

- Python `>=3.10`
- `numpy>=1.23`
- `zeromodel==1.0.13`

The core wheel used for validation was:

```text
packages/core/dist/zeromodel-1.0.13-py3-none-any.whl
```

The observation wheel filename was:

```text
packages/observation/dist/zeromodel_observation-1.0.13-py3-none-any.whl
```

## Schema-Version Constants

Preserved version constants:

- `IMAGE_OBSERVATION_VERSION = "zeromodel-image-observation/v1"`
- `VISUAL_ADDRESS_CONTRACT_VERSION = "zeromodel-visual-address-contract/v1"`
- `VISUAL_ADDRESS_DECISION_VERSION = "zeromodel-visual-address-decision/v1"`
- `VISUAL_ADDRESS_MANIFEST_VERSION = "zeromodel-visual-address-manifest/v1"`
- `DEPLOYMENT_BINDING_VERSION = "zeromodel-deployment-binding/v1"`

## Preserved Identities

Package-local tests preserve deterministic identity behavior for:

- `ImageObservation.raw_digest`
- `VisualAddressContract.digest`
- `VisualAddressDecision.digest`
- `VisualAddressManifest.manifest_id`
- `DeploymentBinding.binding_id`
- `PrototypeBinding` canonical manifest payloads
- `MatrixBlob` linkage in visual-address manifests

Identity payloads remain based on explicit canonical bytes or sorted JSON data,
not module paths, repository paths, memory addresses, hostnames, or filesystem
ordering. No golden identity was updated for this stage.

## Migrated Tests

Observation-owned historical tests were moved or rewritten as package-local
tests:

- `tests/test_deployment_binding.py` ->
  `packages/observation/tests/test_deployment_binding.py`
- Contract-only assertions from `tests/test_visual_address.py` ->
  `packages/observation/tests/test_visual_address_contracts.py`

New observation package tests:

- `packages/observation/tests/test_observation_api_isolation.py`
- `packages/observation/tests/test_visual_address_contracts.py`

The remaining `tests/test_visual_address.py` behavior is classified for the
vision stage because it exercises concrete visual indexing,
`VisualSignReader`, `DeterministicVisualAddressProvider`, and visual policy
reader behavior.

## Test Results

Source-tree validation:

```text
PYTHONPATH=packages/observation/src;packages/core/src python -m pytest -q packages/observation/tests
32 passed in 0.30s
```

Clean wheel validation:

```text
OBSERVATION_WHEEL_PATH=packages/observation/dist/zeromodel_observation-1.0.13-py3-none-any.whl `
  build/observation-isolation-venv/Scripts/python.exe -m pytest -q packages/observation/tests
32 passed in 0.35s
```

`pip check` passed in the clean environment.

## Import Locations

Both packages resolved inside the clean validation environment:

```text
C:\Projects\zeromodel\build\observation-isolation-venv\Lib\site-packages\zeromodel\core\__init__.py
C:\Projects\zeromodel\build\observation-isolation-venv\Lib\site-packages\zeromodel\observation\__init__.py
```

The forbidden-import test verifies that importing `zeromodel.observation` loads
`zeromodel.core` and does not load `zeromodel.analysis`, `zeromodel.vision`,
`zeromodel.video`, `zeromodel.persistence`, `sqlalchemy`, `torch`,
`torchvision`, `transformers`, or `PIL`.

## Wheel Contents

The observation wheel contains only these entries:

- `zeromodel/observation/__init__.py`
- `zeromodel/observation/deployment_binding.py`
- `zeromodel/observation/visual_address.py`
- `zeromodel/observation/visual_address_manifest.py`
- `zeromodel_observation-1.0.13.dist-info/METADATA`
- `zeromodel_observation-1.0.13.dist-info/WHEEL`
- `zeromodel_observation-1.0.13.dist-info/top_level.txt`
- `zeromodel_observation-1.0.13.dist-info/RECORD`

Rejected content assertions cover `zeromodel/__init__.py`, copied core files,
analysis, vision, video, persistence, root tests, research, examples, docs, and
scripts.

## Boundary And Quality

Package-boundary validation passed:

```text
python scripts/check_package_boundaries.py
Package boundary check passed: 118 production modules
```

Observation quality checks passed:

```text
python -m ruff check packages/observation/src packages/observation/tests
python -m ruff format --check packages/observation/src packages/observation/tests
python -m mypy packages/observation/src
```

Scoped repository quality checks passed with observation included:

```text
python scripts/check_quality.py
Quality checks passed
```

## Historical Test Classification

- Moved to observation: deployment binding, image observation DTOs, visual
  address contract DTOs, decision DTOs, provider protocol, manifest and
  prototype-binding contracts.
- Vision test: concrete visual indexing, `VisualSignReader`, deterministic
  visual address providers, visual policy reader behavior.
- Video test: temporal policy behavior, observation provenance, video provider
  measurement, observation universe behavior.
- SQL persistence test: observation ledger persistence and SQL store behavior.
- Research test: benchmark provider measurements, FAR/FRR-style evaluation, and
  evidence-generation scripts.

## Downstream Import Changes

No downstream production import change was required. Existing production
consumers already import observation DTOs and protocols from
`zeromodel.observation.*`. Stale historical root tests that import concrete
visual provider behavior remain classified for the vision or video stage.

## Remaining Defects

- The `setuptools` license-table and license-classifier deprecation warnings
  remain from the package metadata style and can be modernized later.
- Vision isolation is the next package-validation stage.
