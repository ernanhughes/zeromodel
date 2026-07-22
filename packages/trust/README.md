# zeromodel-trust

Artifact trust, authenticity, and deployment-scope authorization kernel for
the ZeroModel workspace.

## Claims boundary

The supported claim is: ZeroModel can verify the integrity, signature, and
declared deployment authorization of an identified artifact under a bounded
trust policy.

This package does **not** claim: secure hardware, remote attestation, PKI
deployment, regulatory certification, supply-chain security closure, or
tamper-proof storage.

## What this checks, as separate decisions

- **Integrity** - do the canonical bytes match the declared artifact digest?
- **Authenticity** - was the artifact's authorization manifest signed by a
  key belonging to the declared signer?
- **Trust** - is that signer accepted under the active trust policy?
- **Authorization** - is this signer permitted to authorize this artifact
  kind for this deployment scope?
- **Freshness / rollback** - is the artifact within its validity window, at
  an acceptable policy epoch, and not revoked?

Each of these is preserved as its own boolean on `TrustDecisionDTO` - never
collapsed into a single opaque pass/fail.

## Cryptography

Ed25519 via the `cryptography` package (a standard, audited library) - no
custom cryptographic primitives. The signature covers a canonical digest of
the artifact's authorization manifest (artifact digest, artifact kind,
deployment scope, policy epoch, validity window, issuer identity) - never
just rendered pixels, a filename, or an embedded checksum string.

Private-key generation/signing helpers in `zeromodel.trust.crypto` exist for
tests and explicit authoring workflows only. Production loading only ever
verifies; it never signs, and no private key material is ever written to a
fixture, report, or source-controlled file.
