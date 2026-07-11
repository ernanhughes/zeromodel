# Changelog

## 0.1.0a1 - Unreleased

First TestPyPI release candidate for the clean `zeromodel` package surface.

### Highlights

- Publishes ZeroModel as an alpha package rather than presenting the current surface as a stable 2.x release.
- Keeps the primary claim narrow: deterministic, inspectable Visual Policy Map artifacts for scored data.
- Includes the validated core artifact kernel, dense policy views, spatial optimizer, temporal decision manifold, learning traces, training progress artifacts, tracker-export adapters, critic/evidence risk artifacts, bundles, rendering, and edge gates.
- Adds release validation through source/wheel build checks and `twine check`.
- Adds a manual TestPyPI publishing workflow using GitHub Actions Trusted Publishing.

### Release posture

This release candidate is intentionally alpha. It does not claim planet-scale traversal, automatic semantic view learning, task-level decision accuracy improvement, real-world hallucination detection, or real training-run validation. See `docs/claims-audit.md` for the claim boundary.
