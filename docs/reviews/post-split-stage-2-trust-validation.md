# Post-split remediation — Stage 2: `zeromodel-trust` artifact trust and authorization kernel

**Baseline SHA (start of this stage):** `c1ce710db50655a6082567fd3f376c3134095ea2` (branch `main`, working tree clean at start — confirmed via `git branch --show-current`, `git rev-parse HEAD`, `git status --short` before any change).
**Final SHA / working-tree state:** uncommitted at the time of writing — every change described below is staged in the working tree, pending an explicit commit decision (nothing was committed, tagged, published, or pushed in this stage).
**Objective:** build a cryptographic artifact-trust and deployment-authorization kernel as a new, independent distribution, without restoring the root `zeromodel` package, without making `core` or `artifacts` depend back on `trust`, and without inventing cryptography.

## Prerequisite finding and scope decision

The stage brief assumed an already-merged "Artifacts stage" (an `ArtifactRef` + resolver/store protocol package). No such package, ADR, or trust model existed anywhere in the repository (confirmed by exhaustive `find`/`grep` across `packages/`, `docs/`, and `git log --all`). Per explicit user direction, a minimal `zeromodel-artifacts` package was built first (tracked as an implicit "Stage 1") to unblock this stage; Trust and Navigation both depend on it. See [package-boundaries.toml](../../package-boundaries.toml) for the resulting dependency graph.

## Architecture built

- **New distribution:** `zeromodel-trust`, namespace `zeromodel.trust`, `packages/trust/`.
- **Dependency rule enforced:** `trust → core + artifacts` only. Neither `core` nor `artifacts` imports `trust` (enforced by `scripts/check_package_boundaries.py` against `package-boundaries.toml`'s `depends_on` graph — a forbidden edge in either direction fails the check).
- **Cryptography:** Ed25519 via the `cryptography` package exclusively (`cryptography.hazmat.primitives.asymmetric.ed25519`) — no custom cryptographic primitives. Signing/key-generation helpers (`crypto.generate_signing_key`, `crypto.sign_digest`) are explicitly documented and tested as authoring/test-only; `crypto.verify_signature` is the only production-loading concern and never raises on malformed input (returns `False`). No private key material is written to any fixture, report, or source-controlled file — enforced by a dedicated regression test that scans the package tree for PEM private-key markers.
- **Signature coverage:** the signature covers `ArtifactAuthorizationDTO.authorization_id`, itself a canonical digest over artifact digest, artifact kind, adapter/consumer contract, the full deployment scope, policy epoch, validity window, and issuer identity ([dto.py](../../packages/trust/src/zeromodel/trust/dto.py) `authorization_signing_payload`) — never a filename, rendered pixels, or an embedded checksum string alone.

### Files

- `packages/trust/pyproject.toml`, `README.md`
- `packages/trust/src/zeromodel/trust/dto.py` — 11 DTOs (`SignatureEnvelopeDTO`, `SignerIdentityDTO`, `TrustedSignerDTO`, `TrustPolicyDTO`, `TrustPolicyRuleDTO`, `DeploymentScopeDTO`, `ArtifactAuthorizationDTO`, `RevocationRecordDTO`, `TrustVerificationRequestDTO`, `TrustDecisionDTO`) + `TrustFailureCode` (16 members). `ArtifactAuthorizationDTO.authorization_id` is self-validating (recomputed from its own canonical content in `__post_init__`, matching the existing `VPMArtifact.artifact_id` pattern in `zeromodel.core`).
- `packages/trust/src/zeromodel/trust/crypto.py` — Ed25519 sign/verify.
- `packages/trust/src/zeromodel/trust/revocation.py` — `RevocationResolver` protocol, `InMemoryRevocationResolver` (bounded, in-memory, explicitly not network PKI), `IndeterminateRevocationResolver` (models an unreachable/unconfident backend).
- `packages/trust/src/zeromodel/trust/verify.py` — `verify_artifact_for_scope(...)`, factored into one private helper per concern (`_check_integrity`, `_check_signature`, `_check_signer_trust`, `_check_authorization_scope`, `_check_freshness`, `_check_revocations`) so no function mixes more than one decision.
- `packages/trust/src/zeromodel/trust/loading.py` — `ArtifactNotAuthorized` + `require_authorized(decision)`, the fail-closed loading example (does not execute a real controller, makes no safety-certification claim).
- `packages/trust/src/zeromodel/trust/__init__.py` — full public API export (no restricted subset was specified for Trust, unlike Navigation).

## The five decisions, preserved separately

`TrustDecisionDTO` carries `integrity_valid`, `signature_valid`, `signer_known`, `signer_trusted`, `artifact_kind_allowed`, `scope_authorized`, `time_valid`, `epoch_valid`, `not_revoked` as independent booleans, plus a `failure_codes` tuple and a final `decision` of `"authorized"` / `"rejected"` / `"indeterminate"`. `blocks_execution` is `True` for both `"rejected"` and `"indeterminate"`.

`indeterminate` is reached only when every other check passed and *only* revocation resolution could not return a confident answer (`RevocationStatus.INDETERMINATE`) — a definite `REVOKED` status still produces `"rejected"`. This required a real bug fix during development: `signer_trusted` was initially computed by conflating "not CLEAR" with "untrusted," which made an indeterminate revocation backend incorrectly reject rather than report `"indeterminate"`. Caught by `test_indeterminate_revocation_blocks_but_is_distinct_from_rejected` on first test run; fixed by making `signer_trusted` track only definite `REVOKED` status, with indeterminate-vs-revoked routed entirely through the separate `not_revoked` computation.

## Tests

`packages/trust/tests/` — 31 tests, all passing:

- `test_dto_and_crypto.py` (9): self-validating id rejection, non-ed25519 rejection, empty-allowed-kinds rejection, invalid target_kind rejection, invalid decision-string rejection, scope wildcard matching, Ed25519 sign/verify round trip, malformed-input never raises, no private-key material committed.
- `test_trust_api_isolation.py` (3): deliberate public API, import isolation (core + artifacts load, no forbidden siblings including `zeromodel.navigation`), wheel-content check.
- `test_verify_artifact_for_scope.py` (19): valid/authorized, deterministic repeated verification, changed artifact bytes, PNG-vs-canonical-bytes distinction, changed signature, changed authorization field with replayed old signature, missing signature, unknown signer, revoked signer, wrong artifact kind, wrong scope, expired, not-yet-valid, old epoch, revoked authorization, revoked artifact digest, indeterminate revocation, fail-closed loading (raises when not authorized, passes through when authorized).

## Governance integration

Wired into every standard tool (shared with Stage 3 — see the "Combined governance wiring" section in [post-split-stage-3-navigation-validation.md](post-split-stage-3-navigation-validation.md)): `requirements-dev.txt`, root `pyproject.toml` (`pythonpath`, `mypy_path`), `package-boundaries.toml`, `scripts/run_fast_tests.py` (`TEST_ROOTS`), `scripts/check_quality.py` (all three governed path lists), `scripts/validate_release_candidate.py` (`PACKAGES["trust"]`), a dedicated `.github/workflows/trust-package.yml`, and `.vscode/settings.json`.

## Claims boundary (as documented in `packages/trust/README.md`)

Supported: verifying integrity, signature, and declared deployment authorization of an identified artifact under a bounded trust policy.
Explicitly not claimed: secure hardware, remote attestation, PKI deployment, regulatory certification, supply-chain security closure, or tamper-proof storage.

## Validation run this session

```
python -m pytest -q packages/trust/tests          # 31 passed
python -m mypy packages/trust/src                  # Success: no issues found
python -m ruff check packages/trust/src packages/trust/tests    # All checks passed
python -m ruff format --check packages/trust/src packages/trust/tests  # passed (after one format pass)
python scripts/check_quality.py                     # Quality checks passed (all 9 packages)
python scripts/run_fast_tests.py                    # see combined report below
python scripts/validate_release_candidate.py        # Release candidate validation passed
```

## Explicitly not run without further authorization

- Building/publishing to TestPyPI or PyPI (`.github/workflows/publish-testpypi.yml`) — not triggered, no version bump, no tag.
- No git commit, push, or tag was made for this stage's changes.
