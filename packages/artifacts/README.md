# zeromodel-artifacts

Canonical artifact reference, resolution, and content-addressed storage for
the ZeroModel workspace.

This package defines the stable, cross-package `ArtifactRef` identity and the
`ArtifactResolver` / `ArtifactStore` protocols other packages (such as
`zeromodel-trust` and `zeromodel-navigation`) use to persist and resolve their
own artifacts, without each package inventing its own storage layer.

It reuses `zeromodel.core`'s existing canonicalization and digest primitives
(`canonical_json_bytes`, `sha256_digest`) rather than redefining them.
