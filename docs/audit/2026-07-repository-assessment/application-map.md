# Utility Assessment

Strongest supported application: deployment of a closed, reviewed decision table as an immutable artifact for a small deterministic runtime. The likely adopter is an engineer or auditor who needs to version a finite policy, trace every decision to source coordinates, and optionally export a read-only consumer plan.

The scored source is produced by hand rules, a teacher policy, optimizer, or model run. The recurring consumer operation is state-row lookup. The artifact is preferable to rerunning the producer when the producer is expensive, non-deterministic, unavailable at the edge, or needs review before deployment. A VPM is preferable to an ordinary table only when source/view mapping, provenance, bundle identity, rendering, or cross-language plan identity are actually used.

Unknown states fail explicitly. Failure consequences depend on caller policy; the kernel does not provide fallback behavior. Calibration/review is required for the score-producing process and for row coverage. Credible deployment exists for bounded embedded policies and audit trails; open-world visual perception is outside this slice.

Recommended classification: supporting infrastructure. It is coherent and reusable, but its independent product value over a conventional table baseline is not yet proven by human-inspection, operational, or performance evidence.
