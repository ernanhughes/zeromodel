# Release process

## Current: ZeroModel 1.2.0

ZeroModel 1.2.0 is a coordinated eleven-distribution release of the Visual AI Computing foundation.

The release includes:

- `zeromodel`
- `zeromodel-analysis`
- `zeromodel-observation`
- `zeromodel-vision`
- `zeromodel-perception`
- `zeromodel-observer`
- `zeromodel-video`
- `zeromodel-sqlalchemy`
- `zeromodel-artifacts`
- `zeromodel-trust`
- `zeromodel-navigation`

[`package-boundaries.toml`](../package-boundaries.toml) is the machine-readable authority for distribution names, namespaces, source roots, internal dependency edges, publication eligibility, and the coordinated release version.

The exact release claim and boundaries are recorded in [`docs/releases/1.2.0.md`](releases/1.2.0.md). The authoritative public evidence posture remains [`docs/claims-audit.md`](claims-audit.md).

## Required validation

Before tagging or publishing `v1.2.0`, run from a clean checkout:

```powershell
python scripts/validate_release_candidate.py
python scripts/run_fast_tests.py
python scripts/check_quality.py
```

The coordinated validator must verify:

- every package declares version `1.2.0`;
- all internal dependencies use the coordinated `1.2.0` pin;
- package metadata agrees with `package-boundaries.toml`;
- bounded fast tests pass;
- package-local tests pass;
- cross-package integration tests pass;
- visual-transition regression tests pass;
- all wheels and source distributions build;
- `twine check` passes;
- clean-environment installation succeeds;
- public API imports succeed;
- release evidence is generated for the exact commit.

The validator writes versioned package manifests and a release-candidate report under:

```text
docs/architecture/package-release-artifacts-1.2.0.json
docs/architecture/package-public-api-1.2.0.csv
docs/architecture/package-release-test-layers-1.2.0.json
docs/results/release-candidate-1.2.0/
```

Generated evidence must not be copied from an older release line.

## Release preparation

The release pull request should contain only deliberate release and positioning changes:

1. coordinated package versions;
2. coordinated internal dependency pins;
3. `package-boundaries.toml` release version;
4. release-validator version and generated paths;
5. root and package README version references;
6. changelog entry;
7. release notes;
8. generated release-candidate evidence after validation.

The preparation PR must not upload to PyPI, create a tag, or create a GitHub release.

## Review and merge

Before merge:

- inspect the complete diff;
- confirm historical `1.0.13` and `1.1.0` evidence records were not rewritten as if they belonged to 1.2.0;
- confirm every public claim links to a proof and a boundary;
- require the package and repository quality workflows to pass;
- preserve failed or unsupported benchmark results;
- verify all generated reports identify the exact release commit.

## Publish

After the release pull request is merged:

1. return to a clean, synchronized `main` checkout;
2. rerun the complete release gate against the merged commit;
3. confirm the built metadata declares `1.2.0` for all eleven distributions;
4. publish the distributions in dependency order;
5. install the published packages into a clean environment;
6. run the public API and bounded smoke checks;
7. create annotated tag `v1.2.0` at the exact validated commit;
8. create the GitHub release and attach the built artifacts and release evidence.

Recommended dependency-aware publication order:

```text
zeromodel
    ↓
zeromodel-analysis
zeromodel-observation
zeromodel-artifacts
    ↓
zeromodel-vision
zeromodel-perception
zeromodel-video
zeromodel-trust
zeromodel-navigation
    ↓
zeromodel-observer
zeromodel-sqlalchemy
```

Parallel publication inside the same level is acceptable only when the package index and automation handle dependency availability reliably.

## Recovery and repeatability

A partially completed publish must be recoverable without reusing immutable versions:

- do not delete or move a published tag;
- do not upload different bytes under an existing package version;
- verify an existing tag resolves to the expected release commit;
- verify an existing GitHub release belongs to the expected tag;
- record publication failures separately from package-validation failures;
- rerun clean-environment installation after recovery.

## Claims boundary

A successful release proves package construction, installation, bounded runtime behavior, and the named integration contracts exercised by the release gates.

It does not establish:

- general or open-world visual recognition;
- arbitrary image understanding;
- scientific provider validity;
- general formal verification;
- semantic safety certification;
- production authorization;
- arbitrary image-transform survival;
- planet-scale traversal;
- constrained-device performance without named hardware measurements.

## Historical release records

Historical records remain evidence for their original release lines and should not be renamed:

- [`docs/releases/1.1.0.md`](releases/1.1.0.md)
- [`docs/architecture/package-system-1.1.0.md`](architecture/package-system-1.1.0.md)
- [`docs/results/release-candidate-1.1.0/`](results/release-candidate-1.1.0/)
- the unpublished 1.0.13 package-split evidence under `docs/architecture/`
- [`docs/releases/1.0.12.md`](releases/1.0.12.md)

The old single-package `scripts/create-release.ps1` workflow is historical and must not be used to publish the coordinated eleven-package system unless it is explicitly rewritten and validated for that topology.
