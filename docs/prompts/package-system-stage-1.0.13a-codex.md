# Codex Prompt — Stage 1.0.13A: Workspace and Lightweight Core

You are implementing Stage 1.0.13A of the ZeroModel multi-package overhaul.

This stage establishes the package workspace, creates the independently buildable
`zeromodel` core distribution, removes the monolithic root API, and introduces
machine-enforced package boundaries.

Do not extract analysis, observation, vision, video, or SQLAlchemy packages in
this stage except where a small split is required to remove a forbidden dependency
from an approved core module.

## Repository

```text
https://github.com/ernanhughes/zeromodel
```

## Preconditions

Do not begin until all of the following exist and have been reviewed:

```text
docs/architecture/package-system-1.0.13.md
docs/architecture/package-inventory-1.0.13.md
docs/architecture/package-module-map-1.0.13.csv
docs/architecture/package-import-graph-1.0.13.json
docs/architecture/package-dependency-findings-1.0.13.md
```

The inventory review must contain no unresolved Blocker for Stage 1.0.13A.

If a module required for the complete core workflow is classified `undecided`,
stop and report the architectural blocker. Do not guess.

## Branch strategy

Work on a dedicated stage branch based on:

```text
integration/package-system-1.0.13
```

The stage pull request targets that integration branch, not `main`.

Record the exact base commit in the implementation report.

## Authoritative rules

Read and follow:

```text
docs/architecture/package-system-1.0.13.md
docs/architecture/code-quality-policy.md
docs/architecture/video-action-set-rmdto.md
package inventory artifacts
scripts/check_architecture.py
scripts/check_quality.py
quality-baseline.toml
.github/workflows/python.yml
```

The architecture contract wins over implementation convenience.

## Stage objective

After this stage, the repository must contain an independently buildable core
package at:

```text
packages/core/
```

with public imports under:

```python
from zeromodel.core import ...
```

A clean installation of the core wheel must support:

```text
scored rows
    -> deterministic VPM artifact
    -> stable identity and source mapping
    -> bundle or lightweight rendering
    -> exact bounded row address
    -> policy action and evidence
```

The core wheel must require only the Python standard library and NumPy at runtime.

The current root API must no longer exist:

```python
from zeromodel import ScoreTable
from zeromodel import VisualSignReader
from zeromodel import VideoPolicyReader
```

Do not replace it with compatibility aliases.

## Required deliverables

At minimum, this stage must add or update:

```text
packages/core/pyproject.toml
packages/core/README.md
packages/core/src/zeromodel/core/
packages/core/tests/
package-boundaries.toml
requirements/dev.txt
requirements/release.txt
scripts/check_architecture.py
scripts/check_quality.py
scripts/check_distribution_contents.py
scripts/check_workspace_versions.py
.github/workflows/python.yml
docs/architecture/package-transition-ledger-1.0.13.md
```

Add tests for every new reusable script.

The exact moved modules are governed by the reviewed inventory, not by this prompt's
candidate list.

## 1. Establish the workspace

### 1.1 Root project configuration

Convert the repository-root `pyproject.toml` into workspace and tool configuration.
It must no longer describe a publishable `zeromodel` distribution.

Preserve and adapt relevant tool configuration for:

- pytest;
- Ruff;
- mypy;
- repository markers;
- any existing quality tooling.

Do not leave a root build configuration that can accidentally publish a second
`zeromodel` distribution.

Move repository development and release tooling dependencies into explicit files:

```text
requirements/dev.txt
requirements/release.txt
```

Avoid undeclared environment assumptions. Keep package runtime dependencies in
the package's own `pyproject.toml`.

### 1.2 Core package metadata

Create `packages/core/pyproject.toml` with:

```text
distribution name: zeromodel
version: 1.0.13
requires-python: >=3.10
runtime dependency: numpy>=1.23
```

Preserve appropriate project metadata, license, classifiers, repository URLs, and
README linkage from the current distribution.

Use a single package-local version source. Recommended shape:

```text
packages/core/src/zeromodel/core/_version.py
```

with setuptools dynamic version loading. `zeromodel.core.__version__` may be
public. There must be no `zeromodel.__version__` compatibility surface.

### 1.3 Implicit namespace

There must be no:

```text
zeromodel/__init__.py
packages/core/src/zeromodel/__init__.py
```

The `zeromodel` namespace must be implicit under PEP 420.

Add a test that inspects source roots and the built wheel for a forbidden namespace
initializer.

## 2. Introduce `package-boundaries.toml`

Create the machine-readable package manifest described by the architecture
contract.

For Stage A, include all six final packages, even though only core is active.
Represent package state explicitly, for example:

```toml
[packages.core]
state = "active"

[packages.analysis]
state = "planned"
```

Do not infer package activation from whether a directory happens to exist.

The manifest must also represent transitional root modules that remain on the
integration branch. Each transitional module must have:

- current path;
- final owner package;
- removal stage;
- reason it remains;
- whether it may appear in a built wheel.

No transitional root module may appear in the core wheel.

The manifest is the single authority for:

- distributions;
- namespaces;
- source roots;
- allowed dependencies;
- package state;
- synchronized release version;
- transitional ownership.

Architecture and version scripts must read it rather than duplicate package lists.

## 3. Extract the core kernel

Use the reviewed module map to move every approved core module or approved core
fragment into:

```text
packages/core/src/zeromodel/core/
```

Use history-preserving moves where a complete file moves. Do not copy a module and
leave the old implementation in place.

Candidate responsibilities include:

- artifact DTOs and validation;
- canonical identity and digest mechanics;
- matrix blob;
- score-table metric helpers;
- deterministic artifact construction;
- minimal views required to construct usable artifacts;
- bundle round trip;
- lightweight rendering;
- exact bounded policy lookup and decision evidence;
- small dependency-free portable policy representation when approved by inventory.

The inventory is authoritative about exact files and split requirements.

### 3.1 Split mixed modules correctly

When a current module contains both core and advanced behavior:

1. move the core responsibility into a focused core module;
2. leave the advanced behavior in its transitional current location;
3. update the advanced code to import the public core object;
4. do not create a compatibility wrapper at the old core path;
5. preserve identity, schema versions, and behavior through tests.

Do not force advanced behavior into core merely to avoid a split.

### 3.2 Core public API

Create a deliberate, small public surface in:

```text
packages/core/src/zeromodel/core/__init__.py
```

Re-export only objects owned by core.

Add an explicit public API test that verifies the expected export set and fails on
accidental additions.

Do not re-export analysis, observation, vision, video, persistence, research, or
transitional objects.

## 4. Remove the monolithic root API

Delete the current:

```text
zeromodel/__init__.py
```

Update repository production code, tests, examples, and executable scripts that
import core symbols through the root namespace.

Replace imports with ownership-explicit paths, such as:

```python
from zeromodel.core import ScoreTable
from zeromodel.core.policy import VPMPolicyLookup
```

Choose the package public surface when it is deliberately exported. Use a public
submodule when importing a specialized public object. Do not import private
modules across future package boundaries.

Prohibited:

```python
# No compatibility root export.
from zeromodel import ScoreTable

# No old-path shim.
from zeromodel.artifact import ScoreTable

# No dynamic alias registration.
sys.modules["zeromodel.artifact"] = ...
```

Repository documentation that is executed or copied as an installation example
must be updated in this stage. Broad narrative documentation can be completed in
Stage H, but no live example may teach a removed import.

## 5. Update all inbound core imports

After moving core modules, scan the entire repository and update imports from:

```text
zeromodel/
scripts/
examples/
tests/
```

Do not stop after core tests pass. Transitional advanced modules must consume the
new core package instead of importing removed root modules.

Run an AST check that proves no import of a removed core module path remains.

Record any intentionally deferred import in the transition ledger. An import is
not deferrable merely because it appears only in a slow test.

## 6. Preserve deterministic behavior

Moving identity-bearing code must not change:

- canonical serialization;
- artifact IDs;
- matrix blob IDs;
- digest domain separators;
- schema version constants;
- source mapping;
- bundle payloads;
- policy decisions;
- Lua plan identity if retained in core;
- error and rejection semantics.

Before editing behavior, capture the existing golden values already encoded in
tests. Do not regenerate expected identities to match accidental changes.

Add or retain equivalence tests that construct representative artifacts through
the new imports and assert the existing identities and serialized forms.

If a golden value must change for a reason other than import location, stop and
report it as out of scope.

## 7. Package-aware architecture tooling

Refactor `scripts/check_architecture.py` to discover active source roots from
`package-boundaries.toml`.

It must support both:

- active package roots; and
- explicitly declared transitional root modules.

It must enforce at least:

- unique module ownership;
- no overlapping namespace files;
- allowed inter-package edges;
- no local import cycles;
- no production imports from tests, examples, or research;
- no tracked heavyweight dependency in core;
- no sibling-private import across declared package owners;
- no forbidden `zeromodel/__init__.py`;
- no undeclared production module;
- no transitional module in a production wheel.

Do not hard-code a second package graph in the script.

Add focused tests for manifest parsing, invalid edges, duplicate ownership,
namespace initializers, transitional modules, and the current valid repository.

## 8. Preserve the quality ratchet

Update `scripts/check_quality.py` and `quality-baseline.toml` for moved paths.

Rules:

- preserve existing limits;
- do not raise a numeric ceiling;
- do not drop a file-specific exception merely because the file moved;
- map a moved exception to its new path with the same or lower ceiling;
- new core modules must satisfy the current new-code limits;
- split files should reduce, not reproduce, oversized responsibilities;
- no syntax, cycle, or forbidden-edge check may be exempted.

Add a test that proves a path move cannot silently escape a quality ceiling.

## 9. Distribution-content validation

Create `scripts/check_distribution_contents.py` driven by
`package-boundaries.toml`.

For a built wheel it must verify:

- distribution name and version;
- owned namespace subtree only;
- no `zeromodel/__init__.py`;
- no transitional root module;
- no sibling package subtree;
- expected package metadata;
- deterministic, sorted diagnostics.

Add unit tests using small synthetic wheel archives. Do not test only the happy
path.

## 10. Version validation

Create `scripts/check_workspace_versions.py` driven by
`package-boundaries.toml`.

In Stage A it must verify:

- core reports `1.0.13`;
- manifest reports `1.0.13`;
- no active package has a different version;
- planned package metadata is not required until the package becomes active;
- internal dependency pins, when present, equal `1.0.13`.

The script must naturally extend to all six packages in later stages.

## 11. Testing structure

Create package-local core tests under:

```text
packages/core/tests/
```

Move tests whose complete responsibility is core. Split mixed tests rather than
copying them.

Keep cross-package or transitional tests outside the core package and update their
imports.

At minimum, core tests must cover:

- artifact construction;
- validation;
- deterministic identity;
- matrix blob round trip and tamper rejection;
- source/cell mapping;
- basic views;
- bundle round trip;
- lightweight rendering;
- exact policy lookup;
- public API exports;
- import isolation;
- wheel namespace ownership.

## 12. Clean-wheel proof

Build the core distribution:

```text
python -m build packages/core --outdir build/dist/core
python -m twine check build/dist/core/*
```

Create a genuinely clean virtual environment outside the repository tree.

Install the wheel, not the source tree:

```text
python -m pip install <core-wheel>
python -m pip check
```

From a temporary directory outside the repository, prove:

```python
import sys
import zeromodel.core

assert zeromodel.core.__version__ == "1.0.13"

for forbidden in (
    "sqlalchemy",
    "torch",
    "torchvision",
    "transformers",
    "PIL",
    "zeromodel.analysis",
    "zeromodel.observation",
    "zeromodel.vision",
    "zeromodel.video",
    "zeromodel.persistence.sqlalchemy",
):
    assert forbidden not in sys.modules
```

Copy the core test directory to a temporary location outside the repository and
run the package tests against the installed wheel. Ensure neither the repository
root nor `packages/core/src` appears on `sys.path`.

An editable install is not acceptable evidence for the clean-wheel gate.

## 13. Transitional repository test configuration

Advanced modules remain temporarily in their current source locations until later
stages. Keep them testable on the integration branch without shipping them in the
core wheel.

Requirements:

- transitional source visibility must be explicit in repository test tooling;
- core wheel tests must not see transitional source paths;
- transitional modules must be listed in the transition ledger and manifest;
- no release workflow may package transitional root modules;
- Stage H must have a mechanical check requiring the transitional list to be empty.

Do not add runtime `sys.path` manipulation inside production code.

## 14. Transition ledger

Create:

```text
docs/architecture/package-transition-ledger-1.0.13.md
```

It must list every production Python module remaining outside `packages/*` after
Stage A with:

```text
current path
current import
final package
final namespace
removal stage
known forbidden-edge risk
reason it remains
```

Also list removed old core module paths and prove that no shim remains.

The ledger must be generated or validated against the manifest so it cannot drift
silently.

## 15. CI changes

Update `.github/workflows/python.yml`.

Path filters must include at least:

```text
packages/**
zeromodel/**
integration_tests/**
research/**
requirements/**
package-boundaries.toml
pyproject.toml
scripts/**
tests/**
examples/**
docs/architecture/**
.github/workflows/python.yml
```

Replace root editable installation assumptions.

The workflow must include:

1. repository quality and architecture checks;
2. current bounded fast-suite execution for transitional code;
3. core package unit tests on Python 3.10, 3.11, and 3.12;
4. core sdist and wheel build;
5. Twine metadata validation;
6. distribution-content validation;
7. clean-wheel import isolation;
8. package tests against the installed wheel;
9. workspace version validation;
10. the existing Lua fixture if its implementation remains supported.

Failure logs may be uploaded as artifacts. Do not add self-modifying CI or commits
from workflows.

## 16. Release safety

The existing release tooling assumes one distribution. Stage A must prevent it
from publishing an incomplete multi-package workspace.

Choose one explicit safe behavior:

- make release preparation fail with a clear message while planned packages remain
  inactive; or
- add a manifest-driven completeness gate that fails until all six packages are
  active and Stage H conditions are met.

Do not make a partial core-only 1.0.13 publication possible through the normal
release command.

Document this temporary release lock in the transition ledger.

## 17. Validation commands

Run all relevant checks, including at least:

```text
python scripts/check_architecture.py
python scripts/check_quality.py
python scripts/check_workspace_versions.py
python scripts/run_fast_tests.py
python -m pytest -q packages/core/tests
python -m build packages/core --outdir build/dist/core
python -m twine check build/dist/core/*
python scripts/check_distribution_contents.py build/dist/core/*.whl
```

Run the clean virtual-environment wheel proof described above.

Run tests on Python 3.10, 3.11, and 3.12 when the environment permits. CI must
cover the complete matrix even if the local environment provides only one version.

Record exact commands, pass/fail status, and any environment limitation.

## 18. Prohibited shortcuts

Do not:

- preserve the root API;
- create old-path shim modules;
- copy core code instead of moving or splitting it;
- publish transitional modules in the core wheel;
- add SQLAlchemy, Pillow, Torch, TorchVision, or Transformers to core;
- import optional sibling packages from `zeromodel.core.__init__`;
- use repository-root `PYTHONPATH` as clean-wheel evidence;
- regenerate golden IDs to hide a regression;
- raise quality ceilings;
- hard-code package lists independently in multiple scripts;
- begin extracting analysis, observation, vision, video, or SQLAlchemy wholesale;
- delete research or advanced modules merely because their package stage has not
  happened yet;
- weaken existing scientific or RMDTO tests.

## 19. Required implementation report

Return a report with:

### Baseline

- base branch;
- base commit;
- inventory commit used.

### Workspace changes

- files added;
- root configuration changes;
- manifest summary.

### Core extraction

- files moved;
- files split;
- new target modules;
- public export list;
- removed old import paths.

### Boundary proof

- observed core external dependencies;
- observed core internal package edges;
- wheel content summary;
- namespace initializer check;
- import-isolation result.

### Determinism proof

- golden identities exercised;
- schema/version constants preserved;
- bundle and policy equivalence results.

### Transitional debt

- count of remaining root production modules;
- owner stage for each category;
- release lock status.

### Validation

- exact commands;
- results;
- environment limitations.

### Deviations

List every deviation from this prompt or the architecture contract. If none, say
`None`.

Stop after Stage 1.0.13A. Do not begin Stage B.
