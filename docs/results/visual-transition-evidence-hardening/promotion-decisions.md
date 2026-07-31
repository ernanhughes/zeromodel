# Promotion Decisions

| Candidate abstraction | Promote? | Package | Evidence | Reason |
| --------------------- | -------: | ------- | -------- | ------ |
| `VisualTransitionDomain` | No | examples | Used for benchmark orchestration | Domain fixture contract, not runtime evidence. |
| `DomainTransition` | No | examples | Used by arcade/warehouse datasets | Carries privileged benchmark state and renderer semantics. |
| `ValueAnalysisResult` | No | examples | Value benchmark output | Contains domain-specific decoded values and metric fields. |
| `VisualEvidenceRequirement` | No | examples | Compiler research path | Useful but compiler reproduction timed out locally; not promoted without completed evidence. |
| `RepresentationCandidate` | No | examples | Compiler research path | Candidate search remains benchmark/research machinery. |
| `RegionGeometry` | No | examples | Compiler and warehouse adapters | Region constants and geometry are domain/renderer coupled. |
| `compile_requirement` | No | examples | Compiler research path | Selection logic is not yet proven as production-neutral in this run. |
| Action declaration identity | Yes | `zeromodel.perception` | New focused tests and identity generator | Prevents a free action string from sitting outside the evidence chain. |
| Expectation-set identity | Yes | `zeromodel.perception` | New focused tests and identity generator | Prevents confusing the same observations evaluated under different contracts. |
| Visual reader transition trace | Yes | `zeromodel.perception` | Prompt 03 contract present; new focused tests | Preserves `evidence_only` and `policy_executed` without forcing VisualSignReader use. |
| Composite transition analysis identity | Yes | `zeromodel.perception` | New focused tests and identity generator | Binds ordered evidence, declared action, expectation set, and conformance report. |
