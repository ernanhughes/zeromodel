# ZeroModel package system — current architecture (post six-package split)

**Status:** Current, governing architecture.
**Supersedes (for "what packages exist and how do they depend on each other"):** [package-system-1.0.13.md](package-system-1.0.13.md), which remains the historical record of the six-distribution split and is not rewritten.
**Release:** `1.0.13` (all nine distributions share this version, per the versioning policy in the historical document, §12).

## 1. Relationship to the six-package document

`package-system-1.0.13.md` describes and governs the **completed split baseline**: `zeromodel`, `zeromodel-analysis`, `zeromodel-observation`, `zeromodel-vision`, `zeromodel-video`, `zeromodel-sqlalchemy`. Every decision in that document about those six packages — the implicit namespace rule, the dependency-direction rule, the DTO-only persistence boundary, the production/research separation — still holds and is not re-litigated here.

This document adds three new distributions on top of that baseline and states, as a single authoritative source, what the *current* nine-package workspace looks like. Where this document and the historical one could be read as disagreeing about which packages currently exist, **this document governs**.

See [adr-artifacts-trust-navigation.md](adr-artifacts-trust-navigation.md) for the decision record explaining *why* these three packages exist and the sequence in which they were built.

## 2. The nine distributions

| # | Distribution | Namespace | Owned subtree |
|---|---|---|---|
| 1 | `zeromodel` | `zeromodel.core` | `zeromodel.core` |
| 2 | `zeromodel-analysis` | `zeromodel.analysis` | `zeromodel.analysis` |
| 3 | `zeromodel-observation` | `zeromodel.observation` | `zeromodel.observation` |
| 4 | `zeromodel-vision` | `zeromodel.vision` | `zeromodel.vision` |
| 5 | `zeromodel-video` | `zeromodel.video` | `zeromodel.video` |
| 6 | `zeromodel-sqlalchemy` | `zeromodel.persistence.sqlalchemy` | `zeromodel.persistence.sqlalchemy` |
| 7 | `zeromodel-artifacts` | `zeromodel.artifacts` | `zeromodel.artifacts` |
| 8 | `zeromodel-trust` | `zeromodel.trust` | `zeromodel.trust` |
| 9 | `zeromodel-navigation` | `zeromodel.navigation` | `zeromodel.navigation` |

Packages 1–6 are the historical baseline (unchanged). Packages 7–9 are new.

## 3. Dependency graph (current)

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

zeromodel-artifacts ---------> zeromodel

zeromodel-trust -------------> zeromodel
zeromodel-trust -------------> zeromodel-artifacts

zeromodel-navigation --------> zeromodel
zeromodel-navigation --------> zeromodel-artifacts
```

| Package | May depend on |
|---|---|
| core | standard library, NumPy |
| analysis | core |
| observation | core |
| vision | core, observation |
| video | core, observation |
| sqlalchemy | core, video |
| artifacts | core |
| trust | core, artifacts |
| navigation | core, artifacts |

Enforced edges specific to the new packages:

- **Nothing depends on `trust`.** Not core, not artifacts, not navigation, not any of the original six. Trust is a leaf consumer, never a dependency.
- **Nothing depends on `navigation`** either, for the same reason.
- **`trust` and `navigation` do not depend on each other.** A secure application composes both at the call site (see `packages/trust/README.md` and `packages/navigation/README.md` for the integration-seam example) without either package importing the other.
- **`zeromodel.artifacts` is the sole storage/identity authority** for the new packages. Trust and Navigation persist their own DTOs through `ArtifactStore`/`ArtifactResolver` rather than defining a second content-addressed repository.

This graph is enforced by `scripts/check_package_boundaries.py` against `package-boundaries.toml`, which is the machine-readable source of truth (not this document — if the two ever disagree, `package-boundaries.toml` is what CI actually checks).

## 4. Package ownership summary

### 4.1 `zeromodel-artifacts` / `zeromodel.artifacts`

The storage and identity kernel the other two new packages build on: `ArtifactRef` (stable content-addressed identity, `sha256:<64 hex>`), the `ArtifactResolver`/`ArtifactStore` protocols, and `InMemoryArtifactStore` (a bounded, process-local reference implementation). It reuses `zeromodel.core.content_identity`'s canonicalization rather than redefining it.

As of this document, Artifacts is a **storage kernel** — see [adr-artifacts-trust-navigation.md](adr-artifacts-trust-navigation.md) §5 for the adapter-first report-compilation capability (`ReportAdapter`, `compile_report()`, etc.) layered on top of it, and its own completion status.

### 4.2 `zeromodel-trust` / `zeromodel.trust`

A cryptographic (Ed25519, via the `cryptography` package) artifact-trust and deployment-authorization kernel. Verifies integrity, authenticity, trust, authorization, and freshness/rollback as separately-preserved decisions, never collapsed into one boolean. Production loading only ever verifies; signing/key-generation are test/authoring-only concerns.

### 4.3 `zeromodel-navigation` / `zeromodel.navigation`

A finite, deterministic hierarchy compiler and traversal engine over identified artifacts — explicitly not search (no similarity/relevance definition). Compiles a root tile → internal navigation tiles → leaf artifact bindings structure and traverses it via a stable `TraversalRule` protocol. Kept structurally distinct from `zeromodel.analysis.hierarchy.build_pyramid()`, which reduces one VPM field into coarser levels of itself — an unrelated, intra-artifact operation that Navigation does not touch.

## 5. Sequencing (how these three packages became complete, and what's still pending)

1. **Artifacts (storage kernel)** — built first; both Trust and Navigation require it.
2. **Trust** and **Navigation** — built in parallel against the Artifacts storage kernel, each independently.
3. **Governance wiring** — all three packages wired into `requirements-dev.txt`, root `pyproject.toml`, `package-boundaries.toml`, `scripts/run_fast_tests.py`, `scripts/check_quality.py`, `scripts/validate_release_candidate.py`, per-package CI workflows, and `.vscode/settings.json`.
4. **Stage 1A hardening** (tracked in [post-split-stage-1a-hardening.md](../reviews/post-split-stage-1a-hardening.md)) — closes integrity gaps an external review identified in the first merged commit: mutable artifact manifests, unverified tile/leaf identity on load, incomplete hierarchy closure, incomplete traversal-receipt identity, and trust issuer-attribution/policy-identity gaps.
5. **Artifacts adapter-first report compilation** (Stage B, complete — see [post-split-stage-b-adapter-artifacts.md](../reviews/post-split-stage-b-adapter-artifacts.md)) — extends `zeromodel.artifacts` with `ReportAdapter`, `AdaptedReportDTO`, `compile_report()`, and related contracts, so external applications can compile typed domain reports into source-bound VPM artifacts without ZeroModel understanding their domain meaning.
6. **Compiled-report aggregate closure** (Stage C, complete — see [post-c203e7a7-aggregate-closure.md](../reviews/post-c203e7a7-aggregate-closure.md) and its two follow-up review-response records) — persists `AdaptedReportDTO` as a first-class artifact, binds it to `CompiledReportArtifactDTO` by a real `ArtifactRef`, and adds `zeromodel.artifacts.aggregate` to resolve and cross-validate the complete five-object aggregate (adapted report, adapter contract, `ScoreTable`, `LayoutRecipe`, `VPMArtifact`) via `ResolvedCompiledReportAggregateDTO` — proving the collection describes one coherent report, not merely that each object's own digest is valid. Also fixes `zeromodel.trust`'s `signature_envelope_id` to be a content-derived identity rather than the raw signature hex.

## 6. What is explicitly not present yet

- **`zeromodel.search` does not exist.** `TraversalRule` in Navigation is a deliberate seam a future Search package can implement with similarity-driven rules against the same protocol; Navigation itself defines no similarity or relevance concept. Search is planned, not started, and should not begin before the Stage 1A hardening, Stage B adapter work, and Stage C aggregate closure above are all complete — those items establish the identity/closure guarantees Search would otherwise have to multiply across an entire corpus.
- No package depends on `zeromodel.search`, obviously, since it doesn't exist; this line exists so a future reader doesn't have to infer that from silence.
