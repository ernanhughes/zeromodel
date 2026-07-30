# Changelog

## 1.2.0 - Unreleased

See the [ZeroModel 1.2.0 release posture](docs/releases/1.2.0.md).

ZeroModel 1.2.0 establishes the coordinated foundation for Visual AI Computing.

### Product and architecture

* Rewrites the repository README around the exact bounded 1.2.0 headline claim.
* Adds a system-wide package and repository architecture map.
* Introduces the five-state evidence ladder prominently.
* Links major capabilities to their implementation proof, demonstration programme, and explicit boundary.
* Defines the coordinated relationship between `zeromodel`, `zeromodel-demos`, `zeromodel.org`, and `zeromodel.github.io`.

### Package system

* Coordinates eleven distributions at version `1.2.0`:

  * `zeromodel`
  * `zeromodel-analysis`
  * `zeromodel-observation`
  * `zeromodel-vision`
  * `zeromodel-perception`
  * `zeromodel-observer`
  * `zeromodel-video`
  * `zeromodel-sqlalchemy`
  * `zeromodel-artifacts`
  * `zeromodel-trust`
  * `zeromodel-navigation`
* Updates all internal package dependency pins to `1.2.0`.
* Updates public package documentation links to `https://zeromodel.org/`.
* Makes the observer package an explicit part of the coordinated architecture and install surface.
* Updates release validation and release documentation for the eleven-package topology.

### Claim boundary

* Retains the claims audit as the authoritative evidence record.
* Preserves unsupported, refuted, and insufficient-observability findings as first-class results.
* Positions 1.2.0 as the **foundation for Visual AI Computing**, not as proof of general open-world Visual AI.

## 1.1.0 - 2026-07-27

See the [ZeroModel 1.1.0 release notes](docs/releases/1.1.0.md).

ZeroModel 1.1.0 is the first release of the package-based architecture. It incorporates the namespace split originally prepared under the unpublished 1.0.13 release candidate, together with the complete perception, promotion, activation, rollback, evidence-compiler, and release-validation work completed afterward.

### Package architecture

* Replaces the previous monolithic repository layout with ten coordinated Python distributions:

  * `zeromodel`
  * `zeromodel-analysis`
  * `zeromodel-observation`
  * `zeromodel-vision`
  * `zeromodel-perception`
  * `zeromodel-video`
  * `zeromodel-sqlalchemy`
  * `zeromodel-artifacts`
  * `zeromodel-trust`
  * `zeromodel-navigation`
* Removes the legacy root import compatibility API.
* Establishes package namespaces such as:

  * `zeromodel.core`
  * `zeromodel.analysis`
  * `zeromodel.observation`
  * `zeromodel.vision`
  * `zeromodel.perception`
  * `zeromodel.video`
  * `zeromodel.persistence.sqlalchemy`
  * `zeromodel.artifacts`
  * `zeromodel.trust`
  * `zeromodel.navigation`
* Makes `package-boundaries.toml` the authoritative source for package identity, ownership, dependencies, publication eligibility, integration-test location, and coordinated release version.
* Adds package-boundary, dependency-graph, namespace-overlap, artifact-manifest, public-API, wheel, source-distribution, and clean-install validation.
* Reclassifies stale provider-measurement benchmark coverage under research ownership rather than restoring unsupported production provider-measurement modules.

### Perception and evidence

* Adds the `zeromodel-perception` package.
* Adds deterministic action schemas, dataset manifests, partitions, extracted fields, visual evidence artifacts, expected surfaces, discrepancy surfaces, and transition records.
* Adds transition discovery and conformance evaluation.
* Adds weighted evidence models, baseline inference, calibrated translators, candidate validation, candidate promotion, review, certification, materialization, activation, and rollback contracts.
* Adds evidence-compiled visual contracts that search a bounded candidate family and preserve explicit negative results when the representation is insufficient.
* Adds component, relation, value, and identity evaluation for visual-transition evidence.
* Preserves insufficient-observability results instead of silently claiming recovery.
* Adds the visual-transition regression suite as a required release gate.

### Observation ledger and provenance

* Adds immutable observation records backed by DTO-only store boundaries.
* Adds content-addressed matrix-blob deduplication.
* Adds provider observation descriptors.
* Adds ordered observation operation chains.
* Adds individual provenance operations.
* Adds in-memory and SQLAlchemy-backed observation stores.
* Adds restart-safe SQLite retrieval and relational ownership validation.

### Promotion activation and rollback

* Adds P18G governed perception promotion activation.
* Adds P18H bounded SQLite persistence for:

  * active promotion state;
  * activation receipts;
  * rollback plans;
  * rollback admissions;
  * rollback receipts;
  * execution attempts;
  * explicitly ordered rollback operations.
* Adds immutable rollback-plan identity.
* Adds governed rollback admission.
* Adds atomic inverse-plan execution.
* Requires the current active state to exactly match the activated state named by the rollback plan.
* Makes successful rollback execution idempotent.
* Adds restart-safe activation and rollback state.
* Adds corruption detection and schema-version validation.
* Adds activation-receipt, rollback-plan, and execution identity checks.
* Adds cross-instance SQLite coordination for the bounded in-process reference implementation.
* Rejects unsupported relaxed activation and rollback policies.

### Release validation

* Replaces stale references to `integration_tests/` with the authoritative integration root:

  * `tests/integration`
* Separates bounded fast tests, package-local tests, cross-package integration tests, and visual-transition regression tests.
* Adds structured release-gate outcomes:

  * `passed`
  * `failed`
  * `missing`
  * `zero_tests`
  * `timed_out`
  * `setup_failed`
  * `skipped_optional`
* Prevents missing directories and zero-test collections from being treated as successful release gates.
* Adds perception to the complete package validation topology.
* Adds release validation for all ten package distributions.
* Adds wheel and source-distribution builds.
* Adds Twine metadata checks.
* Adds clean-environment wheel installation.
* Adds public API smoke checks.
* Adds frozen release-candidate evidence and package manifests.
* Adds a bounded timeout for the visual-transition regression suite.

### Quality and CI

* Expands Ruff, mypy, architecture, package-boundary, and quality-report coverage across the package workspace.
* Adds package-local GitHub Actions workflows.
* Adds package-integration and repository-quality workflows.
* Improves failure reporting by preserving downloadable quality logs.
* Fixes import ordering, `__all__` ordering, unused assignments, and lambda-assignment lint failures found during final release preparation.
* Validates source-tree behavior and installed-wheel behavior independently.

### Validation summary

The 1.1.0 release-candidate workflow validates:

* repository quality checks;
* Ruff formatting;
* Ruff lint;
* mypy;
* package boundaries;
* architecture rules;
* code-quality limits;
* bounded fast tests;
* package-local tests;
* cross-package integration tests;
* visual-transition regression tests;
* wheel builds;
* source-distribution builds;
* Twine metadata;
* clean-environment installation;
* public API imports;
* release-report generation.

Research suites remain excluded from the production release verdict unless explicitly promoted into a required release gate.

### Claim boundary

ZeroModel 1.1.0 establishes bounded package construction, installation, deterministic artifacts, named runtime behavior, visual-transition evidence contracts, SQLite-backed activation and rollback, and the integration contracts exercised by the release gates.

It does not claim:

* general artificial intelligence;
* open-world visual recognition;
* arbitrary image understanding;
* universal transformation invariance;
* general formal verification;
* distributed activation or rollback;
* semantic safety recovery;
* production authorization;
* high-availability guarantees;
* planet-scale performance;
* constrained-device performance without named hardware measurements.

See `docs/claims-audit.md` for the authoritative public claim boundary.

## 1.0.12 - 2026-07-22

See the [ZeroModel 1.0.12 release notes](docs/releases/1.0.12.md).

## 0.1.1a1 - Unreleased

First TestPyPI release candidate for the clean `zeromodel` package surface.

### Highlights

* Publishes ZeroModel as an alpha package rather than presenting the current surface as a stable 2.x release.
* Keeps the primary claim narrow: deterministic, inspectable Visual Policy Map artifacts for scored data.
* Includes the validated core artifact kernel, dense policy views, spatial optimizer, temporal decision manifold, learning traces, training progress artifacts, tracker-export adapters, critic/evidence risk artifacts, bundles, rendering, and edge gates.
* Adds release validation through source and wheel build checks and `twine check`.
* Adds a manual TestPyPI publishing workflow using GitHub Actions Trusted Publishing.

### Release posture

This release candidate is intentionally alpha. It does not claim planet-scale traversal, automatic semantic view learning, task-level decision accuracy improvement, real-world hallucination detection, or real training-run validation.

See `docs/claims-audit.md` for the claim boundary.
