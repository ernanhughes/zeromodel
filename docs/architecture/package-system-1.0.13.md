# ZeroModel 1.0.13 Package System Architecture

**Status:** Proposed architecture contract  
**Release:** `1.0.13`  
**Repository model:** one repository, multiple independently buildable Python distributions  
**Compatibility policy:** breaking source and import changes are permitted  
**Release policy:** no intermediate extraction stage is publishable

## 1. Decision

ZeroModel 1.0.13 will replace the current single-distribution package with a
workspace of independently installable capability packages that share the
implicit `zeromodel` namespace.

The package system will contain:

1. `zeromodel` — the lightweight, complete artifact and bounded-policy core;
2. `zeromodel-analysis` — deterministic analysis and verification over core artifacts;
3. `zeromodel-observation` — provider-neutral observation and address contracts;
4. `zeromodel-vision` — concrete deterministic visual-address runtime implementations;
5. `zeromodel-video` — temporal policy and video action-set domain functionality;
6. `zeromodel-sqlalchemy` — SQLAlchemy and SQLite persistence adapters.

Research experiments, benchmark orchestration, evidence generation, and failed or
unpromoted provider implementations will not be shipped as production packages.
They will live under `research/` and may depend on production packages. Production
packages must never import from `research/`.

This is an architectural extraction, not a compatibility migration. The current
root re-export surface will be removed rather than redirected through compatibility
aliases.

## 2. Why this change is required

The current distribution declares a small NumPy dependency surface while the
`zeromodel` namespace now contains several distinct systems:

- deterministic VPM artifacts and identities;
- bounded policy lookup and portable consumers;
- spatial analysis and finite verification;
- provider-neutral observation contracts;
- deterministic and learned visual addressing;
- temporal video policy;
- the video action-set RMDTO domain;
- SQLAlchemy persistence;
- scientific benchmark and evidence machinery.

These concerns have different dependency, stability, testing, and release needs.
Keeping them in one distribution makes the root API misleading, makes optional
capabilities appear foundational, and allows accidental imports to erase the
intended domain boundaries.

The existing RMDTO direction remains valid:

```text
Runtime -> Facade -> Engine -> Service -> Store protocol -> Store implementation
```

The package split raises that rule to the distribution level. The domain package
owns DTOs, services, and protocols. The persistence package owns ORM mappings,
sessions, and SQL Store implementations.

## 3. Goals

ZeroModel 1.0.13 must provide all of the following:

- a small but genuinely functional `zeromodel` core distribution;
- explicit package ownership for every production module;
- an acyclic distribution dependency graph;
- no SQLAlchemy, Torch, Transformers, or Pillow dependency in core;
- provider-neutral observation contracts separated from concrete providers;
- video domain logic separated from database implementation;
- research code separated from supported runtime code;
- independently buildable and independently testable wheels;
- deterministic identity and schema behavior preserved across moves;
- package boundaries enforced by executable architecture checks;
- synchronized `1.0.13` versions for the initial multi-package release.

## 4. Non-goals

The 1.0.13 package-system work will not:

- preserve `from zeromodel import ...` imports;
- provide deprecation aliases for old module paths;
- retain the existing monolithic wheel as a compatibility package;
- publish intermediate extraction stages;
- redesign scientific algorithms merely because their files move;
- promote benchmark-only or negative-result research providers into production;
- introduce a plugin registry before a concrete package requires one;
- split into multiple Git repositories;
- adopt independent package versions in the first release.

## 5. Workspace layout

The final repository layout is:

```text
packages/
├── core/
│   ├── pyproject.toml
│   ├── README.md
│   ├── src/zeromodel/core/
│   └── tests/
├── analysis/
│   ├── pyproject.toml
│   ├── README.md
│   ├── src/zeromodel/analysis/
│   └── tests/
├── observation/
│   ├── pyproject.toml
│   ├── README.md
│   ├── src/zeromodel/observation/
│   └── tests/
├── vision/
│   ├── pyproject.toml
│   ├── README.md
│   ├── src/zeromodel/vision/
│   └── tests/
├── video/
│   ├── pyproject.toml
│   ├── README.md
│   ├── src/zeromodel/video/
│   └── tests/
└── sqlalchemy/
    ├── pyproject.toml
    ├── README.md
    ├── src/zeromodel/persistence/sqlalchemy/
    └── tests/

research/
├── visual/
├── video/
├── benchmarks/
└── evidence/

integration_tests/
examples/
scripts/
docs/
pyproject.toml
package-boundaries.toml
```

The repository-root `pyproject.toml` is workspace and tool configuration. It is
not a publishable ZeroModel distribution after the split. Every published
package builds from its own package directory.

## 6. Shared namespace rule

`zeromodel` becomes a PEP 420 implicit namespace.

There must be no file at:

```text
zeromodel/__init__.py
```

and no package may ship:

```text
src/zeromodel/__init__.py
```

Each distribution owns one non-overlapping subtree:

| Distribution | Owned import subtree |
|---|---|
| `zeromodel` | `zeromodel.core` |
| `zeromodel-analysis` | `zeromodel.analysis` |
| `zeromodel-observation` | `zeromodel.observation` |
| `zeromodel-vision` | `zeromodel.vision` |
| `zeromodel-video` | `zeromodel.video` |
| `zeromodel-sqlalchemy` | `zeromodel.persistence.sqlalchemy` |

No two distributions may contain the same import package or write files into the
same owned subtree.

Public imports must identify capability ownership:

```python
from zeromodel.core import ScoreTable, VPMArtifact, build_vpm
from zeromodel.analysis import SpatialOptimizer
from zeromodel.observation import VisualAddressProvider
from zeromodel.vision import VisualSignReader
from zeromodel.video import VideoPolicyReader
from zeromodel.persistence.sqlalchemy import SqlAlchemyVideoActionSetStore
```

The following compatibility surface is intentionally removed:

```python
from zeromodel import ScoreTable
from zeromodel import VisualSignReader
from zeromodel import VideoPolicyReader
```

## 7. Distribution responsibilities

### 7.1 `zeromodel` / `zeromodel.core`

The core is the smallest complete expression of ZeroModel.

A core-only installation must support this complete path:

```text
scored rows
    -> deterministic VPM artifact
    -> stable identity and source mapping
    -> serialization or rendering
    -> exact bounded row address
    -> policy action and evidence
```

Core owns:

- immutable artifact primitives;
- `ScoreTable`, `LayoutRecipe`, `VPMArtifact`, cells, and regions;
- deterministic canonicalization, identity, and digest mechanics;
- `MatrixBlob` and its identity contract;
- deterministic VPM construction;
- basic source and explicit views required to build usable artifacts;
- bundle round-trip support;
- lightweight rendering that does not require a heavyweight optional stack;
- exact bounded policy lookup and its decision DTOs;
- dependency-free portable policy representation where it remains small and stable;
- core exceptions, scalar types, and protocols required by the above.

Candidate current modules include `artifact`, `bundle`, `matrix_blob`, `metrics`,
`policy_lookup`, `render`, and the minimal parts of `views`. The inventory stage
must verify each candidate and split modules when only part belongs in core.

Core does not own:

- spatial optimization, pattern discovery, manifolds, or experiment analysis;
- generic observation-to-address contracts;
- image encoders or visual-address providers;
- temporal video systems;
- SQLAlchemy or database sessions;
- benchmark orchestration or evidence adjudication;
- application-wide service locators that compose optional packages.

Core runtime dependencies are limited to the Python standard library and NumPy.
Importing `zeromodel.core` must not import any optional ZeroModel package or any
tracked heavyweight external dependency.

### 7.2 `zeromodel-analysis` / `zeromodel.analysis`

Analysis owns deterministic transformations, diagnostics, and verification that
operate on core artifacts but are not required to construct or address one.

Expected responsibilities include:

- artifact and field comparison;
- fuzzy field composition;
- hierarchy and pyramid operations;
- PHOS packing and concentration analysis;
- spatial optimization;
- pattern detection and discovered views;
- decision manifolds and inflection analysis;
- Q-derived policy diagnostics;
- finite policy property checking and verification artifacts;
- policy transition evidence;
- critic, learning, and training-progress artifact builders.

Analysis may depend on core. Core must not depend on analysis.

### 7.3 `zeromodel-observation` / `zeromodel.observation`

Observation defines provider-neutral contracts between runtime observations and
policy addresses.

Observation owns:

- immutable observation DTOs;
- address contracts and address decisions;
- provider protocols;
- calibration and rejection contracts that are provider-neutral;
- prototype and address manifests;
- deployment binding between policy, provider, calibration, and source scope;
- policy-reader composition that delegates an accepted address to core lookup,
  when the composition remains provider-neutral.

Observation must not import Torch, Transformers, Pillow, concrete image encoders,
video benchmark machinery, SQLAlchemy, or ORM classes.

Observation may depend on core. Core must not depend on observation.

### 7.4 `zeromodel-vision` / `zeromodel.vision`

Vision owns supported concrete image-to-address runtime implementations.

Expected responsibilities include:

- deterministic visual feature extraction;
- deterministic visual index construction;
- visual sign reading;
- normalized-pixel and local deterministic providers that have been promoted to
  supported runtime capabilities;
- image-specific calibration implementations;
- image preprocessing required by supported providers.

Vision must depend on observation contracts rather than defining a competing
provider seam.

Learned provider adapters with heavyweight dependencies should not be forced into
the base vision distribution. A future adapter such as
`zeromodel-vision-huggingface` may own Torch and Transformers integrations after
its runtime contract is stable. Until then, learned baselines remain research.

Benchmark harnesses, architecture showdowns, calibration sweeps, and failed or
unpromoted systems belong under `research/visual/`, not in the production wheel.

### 7.5 `zeromodel-video` / `zeromodel.video`

Video owns supported temporal policy and video action-set domain behavior.

Video owns:

- video frame and clip DTOs;
- frame-source protocols and in-memory implementations;
- temporal evidence, policy decisions, and traces;
- video policy readers;
- video action-set DTO aggregates;
- Runtime, Facade, Engine, Service, and Store protocols for the video domain;
- deterministic scientific planning and materialization mechanics that are
  promoted runtime behavior rather than benchmark-only orchestration;
- in-memory Store implementations;
- package-specific CLI entry points that call supported video services.

Video must not import SQLAlchemy, ORM modules, database sessions, or SQL Store
implementations. ORM objects must never cross into this package.

Experimental local-correlation, discriminative, joint-evidence, provider-showdown,
and benchmark-only implementations remain research unless the inventory and
review process identifies a separately supported runtime component.

### 7.6 `zeromodel-sqlalchemy` / `zeromodel.persistence.sqlalchemy`

The SQLAlchemy package is an adapter package.

It owns:

- ORM declarations;
- SQLAlchemy session and engine construction;
- SQLite URL and schema helpers;
- SQL Store implementations;
- DTO-to-ORM and ORM-to-DTO mapping;
- transaction, ownership, ordering, and relational integrity behavior;
- persistence integration tests.

It may depend on core and video. It must implement Store protocols declared by
the video domain. It must not move domain services or scientific computation
into the persistence package.

SQLite remains authoritative for durable identity, relationships, ordering,
deduplication, transactionality, and retrieval. Python and NumPy remain
authoritative for scientific derivation, rendering, transformation, replay, and
digest computation.

## 8. Dependency graph

The only allowed production distribution edges are:

```text
zeromodel-analysis ---------> zeromodel

zeromodel-observation ------> zeromodel
          ^
          |
zeromodel-vision -----------> zeromodel
          ^
          |
zeromodel-video ------------> zeromodel
          ^
          |
zeromodel-sqlalchemy -------> zeromodel-video
zeromodel-sqlalchemy -------> zeromodel
```

In table form:

| Package | May depend on |
|---|---|
| core | standard library, NumPy |
| analysis | core |
| observation | core |
| vision | core, observation |
| video | core, observation |
| sqlalchemy | core, video |

Research may depend on any production package. Examples and integration tests may
depend on any packages they demonstrate. Production packages may not depend on
research, examples, integration tests, or repository scripts.

Circular distribution dependencies are forbidden.

## 9. Cross-package data rule

Package boundaries are crossed only by deliberate public objects:

- immutable DTOs;
- protocols;
- stable identifiers and digests;
- standard-library scalar and collection types;
- explicitly owned NumPy arrays or `MatrixBlob` values;
- serialized bytes when serialization is the contract.

The following may not cross package boundaries:

- ORM entities;
- SQLAlchemy sessions or queries;
- private implementation classes;
- mutable internal caches;
- benchmark fixture objects;
- filesystem-specific state unless the public protocol explicitly models it;
- imports from a sibling package's private module.

A sibling package may import only from the owning package's documented public
surface or a documented public submodule. Imports from names beginning with an
underscore are forbidden across distributions.

## 10. Production versus research

A module belongs in a published package only when all of the following are true:

1. its runtime responsibility is stable and named;
2. it is not tied to one benchmark fixture or one evidence package;
3. its public inputs, outputs, rejection behavior, and identity behavior are defined;
4. it has package-local tests independent of research fixtures;
5. its dependencies are appropriate for the owning distribution;
6. the project is willing to maintain it as a supported capability.

A module belongs under `research/` when any of the following are true:

- it orchestrates a benchmark or parameter sweep;
- it compares named experimental systems;
- it generates claim evidence or adjudication reports;
- it is coupled to the arcade fixture;
- it implements an unpromoted or negative-result provider;
- it exists to reproduce a paper, report, or one-off scientific result;
- its public contract changes with the experiment.

Moving a module to research does not diminish its scientific value. It clarifies
that the module is evidence machinery rather than supported runtime API.

## 11. Package boundary manifest

Stage 1.0.13A will introduce a repository-root `package-boundaries.toml` as the
machine-readable authority for package ownership and allowed dependencies.

The intended shape is:

```toml
schema_version = 1
release_version = "1.0.13"

[packages.core]
distribution = "zeromodel"
namespace = "zeromodel.core"
source_root = "packages/core/src"
depends_on = []

[packages.analysis]
distribution = "zeromodel-analysis"
namespace = "zeromodel.analysis"
source_root = "packages/analysis/src"
depends_on = ["core"]

[packages.observation]
distribution = "zeromodel-observation"
namespace = "zeromodel.observation"
source_root = "packages/observation/src"
depends_on = ["core"]

[packages.vision]
distribution = "zeromodel-vision"
namespace = "zeromodel.vision"
source_root = "packages/vision/src"
depends_on = ["core", "observation"]

[packages.video]
distribution = "zeromodel-video"
namespace = "zeromodel.video"
source_root = "packages/video/src"
depends_on = ["core", "observation"]

[packages.sqlalchemy]
distribution = "zeromodel-sqlalchemy"
namespace = "zeromodel.persistence.sqlalchemy"
source_root = "packages/sqlalchemy/src"
depends_on = ["core", "video"]
```

Architecture tooling must read this manifest. Package ownership and allowed
edges must not be duplicated as unrelated hard-coded lists across scripts.

## 12. Versioning

All initial distributions use the synchronized version `1.0.13`:

```text
zeromodel                 1.0.13
zeromodel-analysis        1.0.13
zeromodel-observation     1.0.13
zeromodel-vision          1.0.13
zeromodel-video           1.0.13
zeromodel-sqlalchemy      1.0.13
```

Production package dependencies use exact synchronized pins for this release:

```toml
dependencies = [
    "zeromodel==1.0.13",
]
```

The release tooling must verify that:

- every package version equals the release version;
- every internal dependency references the same release version;
- every expected distribution was built;
- no unexpected distribution was built.

Independent versioning is deferred until package boundaries and release cadence
have proven stable.

## 13. Public API policy

Every package has a deliberately small `__init__.py` in its owned subtree.

For example:

```text
packages/core/src/zeromodel/core/__init__.py
packages/video/src/zeromodel/video/__init__.py
```

These files may re-export public objects owned by that package. They must not
re-export objects from optional sibling packages merely for convenience.

The namespace root has no initializer and no public exports.

Public API tests must enumerate the intended exports. Adding a new export is a
deliberate API change, not an automatic consequence of creating a module.

## 14. Build and installation requirements

Each package must independently pass:

```text
python -m build <package-directory>
python -m twine check <package-directory>/dist/*
install wheel into a clean virtual environment
pip check
import the package's public namespace
run package-local tests against the installed wheel
```

Tests that claim wheel isolation must not add the repository root or a package
`src` directory to `PYTHONPATH`.

Every wheel must be inspected to confirm that it contains only its owned subtree.
For example, the `zeromodel-video` wheel must not contain `zeromodel/core`,
`zeromodel/vision`, or `zeromodel/persistence` files.

## 15. Test layers

The repository will use four explicit test layers.

### 15.1 Package unit tests

Located in each package's `tests/` directory. They test only the owning package
and declared dependencies.

### 15.2 Package wheel tests

Build and install one wheel in a clean environment, then run import and behavior
smoke tests against installed files.

### 15.3 Integration tests

Located under `integration_tests/`. They install the required package combination
and test cross-package behavior such as:

- core + analysis;
- core + observation + vision;
- core + observation + video;
- core + observation + video + SQLAlchemy.

### 15.4 Research tests

Located with the research system or in a clearly marked research test area. They
may use benchmark fixtures and generated evidence but do not define the production
package contract.

## 16. Import isolation requirements

At minimum, CI must prove:

```python
import sys
import zeromodel.core

for forbidden in (
    "sqlalchemy",
    "torch",
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

Equivalent probes must verify that:

- observation does not import concrete vision or video implementations;
- video does not import SQLAlchemy;
- production packages do not import research;
- package imports perform no database creation, model loading, network access, or
  expensive computation.

## 17. Architecture tooling

The existing architecture checker must evolve from one hard-coded source root to
workspace discovery driven by `package-boundaries.toml`.

It must enforce:

- unique module ownership;
- allowed inter-package import edges;
- no import cycles;
- no production imports from tests, examples, or research;
- no SQLAlchemy imports outside the SQLAlchemy package;
- no tracked heavyweight dependencies in packages that do not declare them;
- no sibling-private imports;
- no `zeromodel/__init__.py` in any wheel source root;
- no overlapping wheel contents.

The current quality ratchet remains in force. Package extraction is not permission
to increase line ceilings or hide legacy debt under new paths.

## 18. CI requirements

The Python workflow must ultimately include:

1. repository quality and architecture checks;
2. package build matrix for all six distributions;
3. clean-wheel import and `pip check` validation;
4. Python 3.10, 3.11, and 3.12 package test coverage;
5. integration combinations defined in section 15.3;
6. Lua fixture validation if the Lua consumer remains supported;
7. distribution-content inspection;
8. synchronized-version verification.

Workflow path filters must include `packages/**`, `research/**`,
`integration_tests/**`, `package-boundaries.toml`, and all workspace build scripts.

## 19. Staged implementation strategy

The package overhaul is implemented through an integration branch so `main` is
not left in a partially extracted state.

```text
main
  |
  +-- integration/package-system-1.0.13
          |
          +-- Stage A: workspace and core
          +-- Stage B: analysis
          +-- Stage C: observation contracts
          +-- Stage D: vision
          +-- Stage E: video
          +-- Stage F: SQLAlchemy
          +-- Stage G: research separation
          +-- Stage H: final integration and release readiness
```

Stage pull requests target the integration branch. The completed integration
branch receives one final adversarial review and one final PR to `main`.

No package is published and no release tag is created from an intermediate stage.
Temporary transitional source paths are permitted only on the integration branch,
must be explicitly recorded, and must be eliminated before Stage H completes.

## 20. Stage definitions

### Stage 1.0.13A — Workspace and core

- introduce the workspace manifest and package-aware architecture tooling;
- create `packages/core`;
- move the approved core kernel using history-preserving moves;
- remove the root re-export API;
- update repository imports of moved core objects;
- build and test the core wheel independently;
- establish transitional test configuration for unextracted modules;
- record every remaining temporary root module.

### Stage 1.0.13B — Analysis

- create `packages/analysis`;
- move deterministic analysis and verification modules;
- update imports and tests;
- prove analysis depends inward on core only;
- remove analysis modules from transitional root paths.

### Stage 1.0.13C — Observation

- create `packages/observation`;
- move provider-neutral DTOs, protocols, manifests, and bindings;
- eliminate duplicate provider seams;
- prove observation has no concrete provider dependencies.

### Stage 1.0.13D — Vision

- create `packages/vision`;
- move supported deterministic image-to-address runtime behavior;
- move experimental, benchmark-only, and unpromoted visual systems to research;
- prove no heavyweight learned stack is imported by base vision.

### Stage 1.0.13E — Video

- create `packages/video`;
- move temporal policy and video action-set domain layers;
- preserve RMDTO direction and immutable identities;
- move in-memory Stores with the domain;
- prove video imports no SQLAlchemy or ORM implementation.

### Stage 1.0.13F — SQLAlchemy

- create `packages/sqlalchemy`;
- move ORM, session, engine, schema, and SQL Store implementations;
- retain DTO-only Store boundaries;
- run SQLite reconstruction, transaction, and concurrency integration tests.

### Stage 1.0.13G — Research separation

- move benchmark orchestration and evidence machinery under `research/`;
- classify every experimental provider as promoted runtime or research;
- ensure production wheels contain no benchmark fixtures or evidence packages.

### Stage 1.0.13H — Final integration

- eliminate all transitional root modules and path configuration;
- build every sdist and wheel;
- run clean-wheel package and integration matrices;
- verify synchronized versions and internal pins;
- update README, release notes, and installation guidance;
- run adversarial external review;
- merge the integration branch only after all blockers are resolved.

## 21. Inventory gate

No production module moves before the repository inventory is reviewed.

The inventory must classify every current Python module as exactly one of:

- `core`;
- `analysis`;
- `observation`;
- `vision`;
- `video`;
- `sqlalchemy`;
- `research`;
- `examples`;
- `tooling`;
- `delete`;
- `split`;
- `undecided`.

A `split` classification must identify the responsibilities and target packages.
An `undecided` classification blocks the extraction stage that would otherwise
move the module.

The inventory must also record public exports, inbound and outbound imports,
external dependencies, tests, CLI entry points, schema and digest ownership, and
known architecture exceptions.

## 22. Definition of done

ZeroModel 1.0.13 package-system work is complete only when:

- all six distributions build independently;
- all distributions report version `1.0.13`;
- the namespace root has no `__init__.py`;
- the root `from zeromodel import ...` API no longer exists;
- every production module has exactly one package owner;
- no transitional root production modules remain;
- the architecture checker reports no forbidden distribution edges;
- core imports only the standard library and NumPy;
- observation is provider-neutral;
- vision contains no unpromoted benchmark systems;
- video imports no SQLAlchemy;
- SQLAlchemy implements video-owned Store protocols using DTO boundaries;
- production packages import no research code;
- every wheel contains only its owned namespace subtree;
- package-local tests pass against installed wheels;
- integration combinations pass on Python 3.10, 3.11, and 3.12;
- the Lua consumer remains green if retained;
- README and release documentation describe the new installation model;
- an adversarial review finds no unresolved blocker;
- the final integration PR is green and mergeable.

## 23. Architectural authority

When implementation convenience conflicts with this contract, the contract wins
unless it is explicitly amended in the same reviewed change.

The inventory may refine module placement, but it may not silently change:

- the six-distribution model;
- the implicit namespace rule;
- the dependency direction;
- the DTO-only persistence boundary;
- the production/research separation;
- the removal of compatibility re-exports;
- the synchronized `1.0.13` release version.
