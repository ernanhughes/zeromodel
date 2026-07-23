# ZeroModel 1.0.13 Package Dependency Findings

Baseline commit: `0e56558fad7ba3d2a74ab39d35c7d4ae14357acc`

## Blocker

No blocker forbidden proposed edges detected by the generated classifier.

## High

- Root `zeromodel/__init__.py` imports heavyweight and research-facing modules at import time, contradicting the lightweight core import requirement. Remedy: remove root compatibility exports during Stage 1.0.13A after package-local APIs are declared.

## Medium

- Optional dependencies are declared globally in one distribution, while modules requiring them are intermingled in the `zeromodel` namespace. Remedy: move dependency-owning implementations to vision, research, or sqlalchemy packages before publishing wheels.

## Low

- CI and release scripts assume one distribution and one `dist/` directory. Remedy: replace with workspace-aware build matrix in Stage 1.0.13H.
