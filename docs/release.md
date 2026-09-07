# Release process

## Current: ZeroModel 1.3.0

ZeroModel 1.3.0 is a coordinated release with thirteen implementation
distributions plus the metadata-only `zeromodel` umbrella distribution.
For normal users, the public installation path is:

```powershell
python -m pip install zeromodel
```

The release includes:

- `zeromodel`
- `zeromodel-core`
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
- `zeromodel-search`
- `zeromodel-critic`

The root [`VERSION`](../VERSION) file is the only human-edited authority for the coordinated release number.

[`package-boundaries.toml`](../package-boundaries.toml) remains the machine-readable authority for distribution names, namespaces, source roots, internal dependency edges, and publication eligibility. Its `release_version`, every package `pyproject.toml` version, internal dependency pins, and public package-version constants are generated mirrors of `VERSION`.

The exact release claim and boundaries are recorded in [`docs/releases/1.3.0.md`](releases/1.3.0.md). The authoritative public evidence posture remains [`docs/claims-audit.md`](claims-audit.md).

## Changing the release version

Edit only `VERSION`, then synchronize the generated mirrors:

```powershell
Set-Content VERSION "1.3.0"
python scripts/release_version.py sync
python scripts/release_version.py check
```

Do not manually change wheel filenames in GitHub Actions. Package workflows resolve the exact built wheel through `scripts/release_version.py wheel-path` and install coordinated wheel sets through `scripts/release_version.py install`.

The version check rejects:

- package metadata that differs from `VERSION`;
- stale internal `zeromodel-*` dependency pins;
- a stale `package-boundaries.toml` release mirror;
- stale public package-version constants;
- any GitHub Actions workflow containing a semantic version inside a ZeroModel wheel filename.

## Required validation

Before tagging or publishing, run from a clean checkout:

```powershell
python scripts/release_version.py check
python scripts/validate_release_candidate.py
python scripts/run_fast_tests.py
python scripts/check_quality.py
python scripts/validate_packaging_contract.py
```

The coordinated validator must verify:

- every package declares the version in `VERSION`;
- all internal dependencies use the coordinated version pin;
- package metadata agrees with `package-boundaries.toml`;
- bounded fast tests pass;
- package-local tests pass;
- cross-package integration tests pass;
- visual-transition regression tests pass;
- all wheels and source distributions build;
- `twine check` passes;
- clean-environment installation of the umbrella wheel alone succeeds;
- core-only installation does not pull unrelated high-level packages;
- the 1.2.0 `zeromodel` to 1.3.0 umbrella/core upgrade path succeeds;
- public API imports succeed;
- release evidence is generated for the exact commit.

For 1.3.0, the validator writes versioned package manifests and a release-candidate report under:

```text
docs/architecture/package-release-artifacts-1.3.0.json
docs/architecture/package-public-api-1.3.0.csv
docs/architecture/package-release-test-layers-1.3.0.json
docs/results/release-candidate-1.3.0/
```

Generated evidence must not be copied from an older release line.

## Release preparation

The release pull request should contain only deliberate release and positioning changes:

1. the root `VERSION` change;
2. generated package versions and internal dependency pins from `release_version.py sync`;
3. generated release paths and evidence;
4. root and package README version references;
5. changelog entry;
6. release notes;
7. generated release-candidate evidence after validation.

The preparation PR must not upload to PyPI, create a tag, or create a GitHub release.

## Review and merge

Before merge:

- inspect the complete diff;
- confirm `python scripts/release_version.py check` passes;
- confirm no workflow contains a hard-coded versioned wheel filename;
- confirm historical `1.0.13` and `1.1.0` evidence records were not rewritten as if they belonged to 1.2.0;
- confirm every public claim links to a proof and a boundary;
- require the package and repository quality workflows to pass;
- preserve failed or unsupported benchmark results;
- verify all generated reports identify the exact release commit.

## Publish

After the release pull request is merged:

1. return to a clean, synchronized `main` checkout;
2. rerun the complete release gate against the merged commit;
3. confirm the built metadata declares the root `VERSION` for every publishable distribution;
4. publish the implementation distributions in manifest-derived dependency order, then publish the `zeromodel` umbrella last;
5. install the published packages into a clean environment;
6. run the public API and bounded smoke checks;
7. create an annotated `v<VERSION>` tag at the exact validated commit;
8. create the GitHub release and attach the built artifacts and release evidence.

Recommended dependency-aware publication shape:

```text
zeromodel-core
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
zeromodel-search
zeromodel-critic
    ↓
zeromodel
```

Generate the exact order from `package-boundaries.toml`. Parallel publication
inside the same level is acceptable only when the package index and automation
handle dependency availability reliably. The umbrella must publish last because
it pins every supported runtime distribution.

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

The old single-package `scripts/create-release.ps1` workflow is historical and must not be used to publish the coordinated package system unless it is explicitly rewritten and validated for that topology.
