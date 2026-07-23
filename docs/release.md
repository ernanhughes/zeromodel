# Release process

## Current: 1.0.13 nine-package release-candidate validation

ZeroModel 1.0.13 uses a nine-distribution release-candidate workflow before any
publish, tag, or GitHub release action:

```powershell
python scripts/validate_release_candidate.py
python scripts/run_fast_tests.py
python scripts/check_quality.py
```

The validator builds and checks all nine packages declared in
`package-boundaries.toml` (the authoritative package configuration; the
validator fails if its own package list drifts from that file):

- `zeromodel` (core)
- `zeromodel-analysis`
- `zeromodel-observation`
- `zeromodel-vision`
- `zeromodel-video`
- `zeromodel-sqlalchemy`
- `zeromodel-artifacts`
- `zeromodel-trust`
- `zeromodel-navigation`

It writes `docs/architecture/package-release-artifacts-1.0.13.json` and
`docs/architecture/package-public-api-1.0.13.csv`. It does not upload to
TestPyPI or PyPI, create tags, or create a GitHub release. No publication of
the 1.0.13 nine-package release has occurred as of this writing; the workflow
below (Phase 1/Phase 2, `scripts/create-release.ps1`) has not been adapted to
drive a nine-package publish and should not be run against 1.0.13 without
that work first — see "Historical: 1.0.12 single-package workflow" below.

## Historical: 1.0.12 single-package workflow

The following two-phase PowerShell workflow published ZeroModel 1.0.12, when
the repository still shipped one monolithic `zeromodel` distribution. It is
recorded here as history and as a starting point for a future nine-package
publish orchestrator, not as a currently runnable process:
`scripts/create-release.ps1` still reads and rewrites `zeromodel\__init__.py`
and a single root `pyproject.toml` version, both of which no longer exist in
this checkout (each of the nine packages now carries its own
`packages/*/pyproject.toml` and version). Do not run this script against the
current tree until it is updated for the nine-package layout.

```text
Prepare
    ↓
release pull request
    ↓
review + CI + merge
    ↓
Publish
    ↓
final integration validation
    ↓
PyPI upload
    ↓
Git tag
    ↓
GitHub release
```

The workflow is implemented by:

```text
scripts/create-release.ps1
```

The existing `scripts/publish-pypi.ps1` remains the low-level production PyPI uploader. The release orchestrator calls it only after the merged release commit and all release gates have been validated.

## Prerequisites

Run releases from a clean Windows checkout with:

- Git;
- Python 3.10 or later;
- GitHub CLI (`gh`) authenticated for `ernanhughes/zeromodel`;
- permission to push release branches and tags;
- a production PyPI token in `PYPI_API_TOKEN`, or available for secure interactive entry;
- the intended release-notes file already committed on `main`.

Check GitHub authentication with:

```powershell
gh auth status
```

## Phase 1: prepare the release pull request

For ZeroModel 1.0.12:

```powershell
.\scripts\create-release.ps1 `
    -Mode Prepare `
    -Version 1.0.12
```

The prepare phase requires a clean, synchronized `main` branch. It then:

1. creates `release/1.0.12`;
2. updates `pyproject.toml`;
3. updates `zeromodel\__init__.py`;
4. updates the README production install pin;
5. adds the release section to `CHANGELOG.md`;
6. verifies the committed release-notes file;
7. installs release dependencies;
8. runs the repository quality gate;
9. runs the bounded fast suite;
10. runs the release demos;
11. builds the source distribution and wheel;
12. runs `twine check`;
13. commits and pushes the release branch;
14. opens the release pull request.

The prepare phase does not upload to PyPI, create a tag, or create a GitHub release.

### Dry-run preflight

```powershell
.\scripts\create-release.ps1 `
    -Mode Prepare `
    -Version 1.0.12 `
    -DryRun
```

A dry run verifies the repository, branch, synchronization state, commands, and release-notes path without changing anything.

## Review and merge

Review the generated release PR and require the normal `Python package` workflow to pass.

The release PR should contain only release metadata changes:

- canonical version declarations;
- README install pin;
- changelog entry;
- release notes.

Merge the PR before starting the publish phase.

## Phase 2: publish the merged release

Return to `main`, pull the merged release commit, and run:

```powershell
.\scripts\create-release.ps1 `
    -Mode Publish `
    -Version 1.0.12
```

The publish phase requires the merged source tree to already declare `1.0.12`. It then:

1. re-runs the local release gates;
2. runs the bounded video-finalization integration validation against the exact merged commit;
3. confirms `main` is synchronized with `origin/main`;
4. pushes the merged commit if necessary;
5. waits for the GitHub package workflow for that commit;
6. checks whether `zeromodel==1.0.12` already exists on PyPI;
7. invokes `scripts/publish-pypi.ps1` when upload is still required;
8. performs the clean-environment PyPI smoke test;
9. creates and pushes annotated tag `v1.0.12`;
10. creates the GitHub release and attaches the built distributions.

Before the irreversible step, the operator must type:

```text
RELEASE 1.0.12
```

Use `-Yes` only in a controlled non-interactive release environment.

## Recovery and repeatability

The publish phase is designed to be safely repeatable after a partial failure:

- an existing PyPI version is detected and not uploaded again;
- an existing tag is accepted only when it resolves to the expected release commit;
- a conflicting local or remote tag is rejected;
- an existing GitHub release is detected and not recreated.

Do not delete or move a published tag. PyPI versions are immutable and cannot be reused.

## Optional skips

The script exposes narrowly scoped recovery switches:

```text
-SkipQuality
-SkipTests
-SkipDemos
-SkipFinalizationIntegration
-SkipPyPI
-SkipGitHubRelease
```

These switches are not normal release posture. Use them only when the skipped gate has already passed for the exact release commit or when recovering a partially completed release.

## Claims boundary

A successful release proves package construction, installation, bounded runtime behavior, and the named integration contracts exercised by the release gates.

It does not establish:

- open-world visual recognition;
- scientific provider validity;
- general formal verification;
- arbitrary image-transform survival;
- planet-scale traversal;
- constrained-device performance without named hardware measurements.

`docs/claims-audit.md` remains the source of truth for public capability wording.

## Next development version

The nine-package split described above is not future work: it is the current
state of `main` at release-candidate version 1.0.13, and the removal of the
legacy root compatibility import surface is already in effect in this
checkout. There is no pending `2.0.0.dev0` package-architecture line still to
begin. Any future major-version bump should be driven by a new breaking
change identified after the 1.0.13 nine-package release candidate is
published, not by the split itself.
