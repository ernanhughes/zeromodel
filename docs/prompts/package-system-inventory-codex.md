# Codex Prompt — ZeroModel 1.0.13 Package Inventory

You are performing the mandatory inventory pass for the ZeroModel 1.0.13
multi-package overhaul.

This is an evidence-gathering and classification task. Do not move production
modules, change public imports, or begin the package extraction.

## Repository

```text
https://github.com/ernanhughes/zeromodel
```

Work from the latest reviewed branch containing:

```text
docs/architecture/package-system-1.0.13.md
```

Record the exact baseline commit SHA in every generated inventory artifact.

## Authoritative documents

Read these before making any classification:

```text
docs/architecture/package-system-1.0.13.md
docs/architecture/code-quality-policy.md
docs/architecture/video-action-set-rmdto.md
pyproject.toml
zeromodel/__init__.py
scripts/check_architecture.py
scripts/check_quality.py
.github/workflows/python.yml
```

The package-system architecture contract is authoritative. The inventory may
refine module placement, identify required splits, and expose contradictions. It
must not silently replace the approved six-package model.

## Approved target packages

```text
core         -> distribution zeromodel
analysis     -> distribution zeromodel-analysis
observation  -> distribution zeromodel-observation
vision       -> distribution zeromodel-vision
video        -> distribution zeromodel-video
sqlalchemy   -> distribution zeromodel-sqlalchemy
```

Additional classifications are:

```text
research
examples
tooling
delete
split
undecided
```

Every current Python module must receive exactly one classification. A `split`
classification must identify each responsibility and target. An `undecided`
classification must state the concrete missing fact that prevents a decision.

## Objective

Create a complete, mechanically grounded map of the current repository so the
package extraction can be implemented without cosmetic boundaries, hidden
cycles, duplicated APIs, or accidental optional dependencies.

The inventory must answer:

1. Which modules exist?
2. What does each module own?
3. Which modules import which other modules?
4. Which external dependencies does each module import?
5. Which symbols does each module expose publicly?
6. Which tests and examples depend on each module?
7. Which modules own persisted schemas, stable versions, digests, or identity behavior?
8. Which modules are runtime capability versus benchmark or evidence machinery?
9. Which modules must be split before they can move cleanly?
10. Which current dependencies contradict the approved target graph?

## Required deliverables

Commit all of the following:

```text
docs/architecture/package-inventory-1.0.13.md
docs/architecture/package-module-map-1.0.13.csv
docs/architecture/package-import-graph-1.0.13.json
docs/architecture/package-dependency-findings-1.0.13.md
```

If reusable analysis code is required, also add:

```text
scripts/analyze_package_inventory.py
tests/test_analyze_package_inventory.py
```

Do not add a one-off script without tests when its output becomes architecture
evidence.

## Inventory scope

Inventory all Python files under at least:

```text
zeromodel/
scripts/
examples/
tests/
```

Also inspect:

```text
.github/workflows/
docs/architecture/
docs/research/
docs/results/
quality-baseline.toml
pyproject.toml
```

Non-Python files do not require one module-map row, but their relationship to
packaging, tests, claims, benchmark evidence, and runtime behavior must be
reported where relevant.

## Mechanical analysis requirements

Use AST-based analysis for normal Python imports. Do not infer the graph only
from text search.

The analysis must resolve:

- absolute imports;
- relative imports;
- imports from package `__init__.py` files;
- sibling imports;
- imports of tracked external packages;
- imports from tests, examples, scripts, or research;
- type-checking-only imports;
- local imports inside functions;
- optional imports guarded by `try/except`;
- imports guarded by `TYPE_CHECKING`;
- dynamic imports through `importlib` where statically discoverable.

Report dynamic imports that cannot be resolved statically.

Do not treat `TYPE_CHECKING` or function-local imports as irrelevant. Record the
edge kind so reviewers can decide whether the package dependency is runtime,
type-only, or deferred.

## Module map schema

`docs/architecture/package-module-map-1.0.13.csv` must contain one row per current
Python module with these columns in this exact order:

```text
path,module,lines,classification,target_distribution,target_namespace,responsibility,public_symbols,inbound_internal_count,outbound_internal_count,external_dependencies,test_paths,example_paths,cli_entry_points,identity_or_schema_ownership,move_action,confidence,rationale,blocking_questions
```

Column rules:

- `path`: repository-relative POSIX path;
- `module`: current import name;
- `lines`: physical line count;
- `classification`: one approved classification;
- `target_distribution`: exact distribution or blank when not production;
- `target_namespace`: exact target import subtree or blank when not production;
- `responsibility`: one concise dominant responsibility;
- `public_symbols`: semicolon-separated deliberate or de facto public symbols;
- `external_dependencies`: semicolon-separated top-level distributions;
- `test_paths`: semicolon-separated direct tests;
- `example_paths`: semicolon-separated direct examples;
- `cli_entry_points`: declared or direct CLI surfaces;
- `identity_or_schema_ownership`: version constants, digest behavior, DTO schema,
  persisted schema, or `none`;
- `move_action`: `move`, `split`, `retain-tooling`, `move-research`, `delete`, or
  `investigate`;
- `confidence`: `high`, `medium`, or `low`;
- `rationale`: evidence-based classification reason;
- `blocking_questions`: blank unless a concrete decision blocker exists.

Escape CSV correctly. Do not place unescaped commas into fields.

## Import graph schema

`docs/architecture/package-import-graph-1.0.13.json` must be deterministic JSON
with sorted keys and arrays. Use this top-level shape:

```json
{
  "schema_version": 1,
  "baseline_commit": "<sha>",
  "generated_at_utc": "<iso-8601>",
  "modules": {},
  "edges": [],
  "strongly_connected_components": [],
  "unresolved_dynamic_imports": [],
  "external_dependencies": {}
}
```

Each module record must include:

```json
{
  "path": "zeromodel/example.py",
  "classification": "core",
  "target_namespace": "zeromodel.core.example",
  "lines": 100
}
```

Each edge must include:

```json
{
  "importer": "zeromodel.a",
  "imported": "zeromodel.b",
  "line": 12,
  "kind": "runtime|type-checking|deferred|optional|dynamic",
  "resolved": true
}
```

Strongly connected components must include only components containing a genuine
cycle. Include one concrete cycle path for each component.

## Public API inventory

The Markdown inventory must separately list:

- every symbol exported by the current `zeromodel/__init__.py`;
- the symbol's defining module;
- its proposed target import;
- whether it is retained, renamed, split, moved to research, or deleted;
- tests and examples that currently use the root export;
- any symbol whose apparent public status exists only because the root initializer
  imports it.

Do not assume that every current root export should survive. The architecture
explicitly removes root compatibility re-exports.

## External dependency inventory

Report all imported third-party top-level modules and map them to the package or
research area that currently requires them.

At minimum, explicitly investigate:

```text
numpy
sqlalchemy
torch
torchvision
transformers
PIL
pytest
```

Distinguish runtime dependencies from test, dev, research, and optional imports.

Identify imports that cause an optional capability to become an import-time
requirement of an otherwise lightweight module.

## Package-data and build inventory

Inspect current packaging configuration and report:

- packages discovered by setuptools;
- non-Python package data;
- console scripts or other entry points;
- optional dependency groups;
- test assumptions created by root `pythonpath` configuration;
- imports that work only because the repository root is on `sys.path`;
- expected wheel contents;
- files currently shipped that should become research-only;
- release and CI scripts that assume one distribution or one `dist/` directory.

Run a current wheel build and inspect the archive contents. Record the exact
command and the relevant observations. Do not modify release metadata in this
inventory task.

## Domain-boundary inventory

For `zeromodel.domains.video_action_set`, `zeromodel.stores`, and `zeromodel.db`,
map the current RMDTO path:

```text
Runtime -> Facade -> Engine -> Service -> Store protocol -> Store implementation -> ORM
```

Report every violation or suspicious edge, including:

- domain imports of SQLAlchemy;
- domain imports of ORM or SQL Store implementations;
- ORM imports of services or runtimes;
- ORM objects escaping Stores;
- filesystem or benchmark orchestration embedded in reusable domain mechanics;
- runtime composition that imports persistence eagerly.

Do not change the architecture during this task. Record findings only.

## Production-versus-research adjudication

For every visual and video provider, benchmark, selector, calibration sweep, and
evidence generator, answer:

1. Is this a supported runtime implementation or experiment machinery?
2. Is it coupled to the arcade fixture?
3. Is its contract stable independently of a benchmark?
4. Does a package-local test exist without research fixtures?
5. Does the claims audit promote, withhold, or refute the capability?
6. Should it move to production, move to research, be split, or be deleted?

Negative-result and unpromoted implementations should default to `research`
unless concrete runtime evidence supports production ownership.

## Split analysis

Do not force a module into one package when it contains multiple responsibilities.

For every `split` classification, provide a table with:

```text
current module
responsibility fragment
target module
target package
symbols to move
inbound callers
identity/schema risk
recommended split order
```

Pay special attention to:

- the current root `__init__.py`;
- application-wide runtime composition;
- benchmark modules that also contain reusable mechanics;
- visual dataset and benchmark contracts mixed with orchestration;
- video action-set files mixing DTOs, scientific derivation, filesystem I/O,
  orchestration, and CLI behavior;
- persistence modules that contain domain validation;
- modules above current quality ceilings.

## Dependency findings document

`docs/architecture/package-dependency-findings-1.0.13.md` must rank findings:

```text
Blocker
High
Medium
Low
```

Each finding must include:

- a short title;
- exact paths and import edges;
- why it conflicts with the target package graph;
- the extraction stage affected;
- the smallest structurally correct remedy;
- whether it blocks Stage 1.0.13A.

Do not pad the report with generic style advice.

## Architecture comparison

Compare the observed graph with the allowed graph:

```text
analysis -> core
observation -> core
vision -> observation, core
video -> observation, core
sqlalchemy -> video, core
research -> any production package
```

Produce a package-level adjacency matrix for both:

1. the proposed classification graph; and
2. the allowed target graph.

List every proposed forbidden edge.

## Validation

Run the repository's existing checks without weakening them:

```text
python scripts/check_quality.py
python scripts/run_fast_tests.py
python -m build
python -m twine check dist/*
```

If the environment lacks a required tool, install the declared development or
release dependency and rerun. Record exact commands and results.

Also validate generated artifacts:

- JSON parses;
- CSV has one header and one row per inventoried Python file;
- paths are unique;
- modules are unique;
- classifications use only allowed values;
- every production classification has a target distribution and namespace;
- every `split` row has a split plan;
- every `undecided` row has a blocking question;
- generated output is deterministic across two consecutive runs except for the
  explicitly recorded generation timestamp.

## Prohibited shortcuts

Do not:

- move or rename production modules;
- add compatibility wrappers;
- alter version numbers;
- relax quality limits;
- suppress import cycles;
- classify by filename alone;
- call benchmark code production merely because it is under `zeromodel/`;
- call code research merely because its current evidence is negative when it
  also provides a stable supported runtime contract;
- silently omit modules that fail parsing;
- treat tests as proof of package ownership without examining responsibility;
- use repository-root import behavior as proof that a future wheel will work.

## Final response

Return:

1. the baseline commit;
2. files added or changed;
3. module count by classification;
4. cycle count;
5. forbidden proposed edge count;
6. Stage 1.0.13A blockers;
7. validation commands and results;
8. unresolved decisions requiring architectural adjudication.

Stop after the inventory deliverables. Do not begin Stage 1.0.13A.
