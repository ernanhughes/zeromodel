# Stage P16A — Perception Test and CI Integrity

P16A establishes the perception package test suite as a repository-level merge gate before further operational recommendation work.

Historical public API tests no longer assert that an earlier development stage is the current `PERCEPTION_STAGE`. They instead verify the durable symbols, semantics, DTOs, stores, and service contracts introduced by those stages. This prevents every new stage from invalidating all earlier public-contract tests by construction.

The new `perception-package` GitHub Actions workflow builds the core, observation, and perception wheels, installs them into a clean Python 3.12 environment, runs `pip check`, executes the complete `packages/perception/tests` suite, and verifies that imports resolve from the installed wheels.

Canonical local command:

```bash
python -m pytest -q packages/perception/tests
```

P16A does not alter dataset splitting, temporal grouping, statistical health thresholds, lifecycle compatibility, production sequencing, or P17 recommendation behavior. Behavioral failures exposed by the new merge gate must be repaired explicitly rather than hidden or excluded.
