# Measurement Boundary

- Aggregate historical metrics were verified for `system-b-v2`, `r1-local-correlation`, and `stage3-v1`.
- Per-observation top-1 rescoring is supported only where committed observation-level top-1 rows are preserved.
- `stage3-v2` and `stage3-v3-b3` are treated as canonical diagnostics, not historical noisy-utility benchmarks.
- Fixed top-k is unavailable because no provider preserves ordered per-observation rankings.
- Score-gap and conformal analyses are unavailable because no provider preserves complete per-row score vectors.
- Reachability replay is unavailable because no provider proves both frame-level visual beliefs and executed actions.
- The reachability tile remains valid because it is compiled from the declared transition source, not from visual outcomes.
