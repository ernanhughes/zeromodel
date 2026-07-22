# Release process

ZeroModel releases use a two-phase PowerShell workflow:

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

After ZeroModel 1.0.12 is published, begin the package-architecture work under:

```text
2.0.0.dev0
```

The package split is a breaking distribution and import-boundary change, so it belongs to the 2.x development line rather than 1.0.13.
