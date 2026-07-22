# ADR: Artifacts, Trust, and Navigation as three new distributions

- **Status:** Accepted. The storage kernel is merged and hardened (Stage 1A) and the adapter-first report-compilation capability on `zeromodel.artifacts` (Stage B) is complete — see [post-split-stage-1a-hardening.md](../reviews/post-split-stage-1a-hardening.md) and [post-split-stage-b-adapter-artifacts.md](../reviews/post-split-stage-b-adapter-artifacts.md).
- **Date:** 2026-07-22
- **Scope:** `zeromodel-artifacts`, `zeromodel-trust`, `zeromodel-navigation` — their existence, dependency direction, ownership boundaries, and relationship to the six-package baseline in [package-system-1.0.13.md](package-system-1.0.13.md)

## Context

Two independent capability requests arrived in the same working session: a cryptographic artifact-trust/deployment-authorization kernel ("Stage 2"), and a finite deterministic hierarchy-compiler/traversal engine over identified artifacts ("Stage 3"). Both were specified as depending on an "Artifacts stage" — a canonical `ArtifactRef` plus a resolver/store protocol — that was assumed to already exist and be merged.

No such package, ADR, or trust model existed anywhere in the repository. `packages/` contained only the six pre-existing distributions; no `ArtifactRef` class, no dedicated resolver/store protocol, and no Stage 0 authority document existed under `docs/`. This was confirmed by exhaustive search (`find`/`grep` across `packages/`, `docs/`, and `git log --all`) before any code was written, specifically to avoid inventing foundational architecture without a decision record.

Building Trust and Navigation directly against ad hoc, package-local identity/storage code each would have produced two incompatible, duplicated content-addressing schemes — exactly the outcome the six-package split was designed to prevent (see `package-system-1.0.13.md` §9, "Cross-package data rule").

## Decision

### 1. Add a minimal `zeromodel-artifacts` distribution first

Before building Trust or Navigation, build a small, storage-only `zeromodel-artifacts` package:

- `ArtifactRef` — a stable identity type: `artifact_kind` (a string) + `artifact_id` (a `sha256:<64 hex>` content digest). Never a filename or display name.
- `ArtifactResolver` / `ArtifactStore` — read/write protocols (`has`, `resolve_canonical_bytes`, `resolve_manifest`, `put`).
- `InMemoryArtifactStore` — a bounded, process-local reference implementation, explicitly not distributed or durable.
- Canonicalization is re-exported from `zeromodel.core.content_identity` (`canonical_json_bytes`, `sha256_digest`), not redefined.

This is tracked informally as "Stage 1" even though it was not separately numbered in the original two stage briefs — it did not exist as a concept before this session, and both Trust and Navigation depend on it.

### 2. `trust → core + artifacts`, `navigation → core + artifacts` — nothing depends on either

Trust and Navigation are each independently built against Artifacts + Core only. Neither depends on the other, and nothing (including Artifacts and Core) depends on either of them. This is enforced by `scripts/check_package_boundaries.py` against `package-boundaries.toml`.

A secure application may compose Navigation and Trust at the call site — e.g., resolve a hierarchy root, verify its authorization through Trust, validate structural closure through Navigation, then traverse — without either package importing the other. See the integration-seam examples in `packages/navigation/README.md` and `packages/trust/README.md`.

### 3. Trust uses standard, audited cryptography only

Ed25519 via the `cryptography` package. No custom cryptographic primitives. Signing and key-generation helpers exist for tests/authoring workflows; production artifact loading only ever verifies. No private key material is ever written to a fixture, report, or source-controlled file (enforced by a dedicated regression test).

### 4. Navigation is not Search

Navigation compiles and traverses a **finite, closed** hierarchy over already-identified artifacts. It defines no similarity or relevance concept. `TraversalRule` is a stable protocol seam a future Search package can implement with similarity-driven routing; the reference implementations Navigation ships (`FixedKeySelectorRule`, `DeclaredPriorityRule`) are deliberately non-search — exact-match or fixed-index routing only.

### 5. Artifacts grows an adapter-first report-compilation capability on top of the storage kernel (Stage B)

A separate design brief ("Stage B") extends `zeromodel.artifacts` so external applications can compile typed domain reports (e.g., a Writer AI-artifact report, a claim-evidence report) into deterministic, source-bound VPM artifacts, via a `ReportAdapter` protocol the *external application* implements — ZeroModel never gains domain-specific knowledge of what "hallucination energy" or "AI-artifact phrasing" mean. See [post-split-stage-b-adapter-artifacts.md](../reviews/post-split-stage-b-adapter-artifacts.md) for the completed implementation record; it does not change the dependency graph above (Artifacts still depends only on Core) and does not replace the existing `ArtifactRef`/`ArtifactStore` APIs, only builds on them.

## Consequences

- The six-package document (`package-system-1.0.13.md`) is preserved as-is and remains authoritative for the six packages it describes; it is not silently rewritten to mention nine packages. [package-system-next.md](package-system-next.md) is the current authority for "how many packages exist and how do they depend on each other."
- An external review of the first merged commit identified real integrity gaps in the storage kernel and the two consumer packages (mutable artifact manifests, unverified tile/leaf identity on load, incomplete hierarchy closure over actual source artifacts, incomplete traversal-receipt identity, and a trust issuer-attribution/policy-identity gap). These are being closed as "Stage 1A" before any Search work begins, precisely because Search would otherwise multiply every one of these assumptions across an entire corpus rather than one hierarchy.
- Search (`zeromodel.search`) is planned but does not exist. It should not begin before Stage 1A and Stage B are both complete.
