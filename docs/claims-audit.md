# ZeroModel claims audit

This document compares the public ZeroModel claims against the current clean `zeromodel` package.

Sources reviewed:

- Public blog post: <https://programmer.ie/post/zeromodel/>
- Current package README
- Current implementation under `zeromodel/`
- Current tests under `tests/`

## Status labels

| Status | Meaning |
|---|---|
| **Validated** | Implemented and covered by explicit tests in this repository. |
| **Implemented / thin evidence** | Code exists, but tests are smoke-level or do not yet prove the stronger wording. |
| **Not validated** | The repo does not currently contain the benchmark, fixture, or reproducible evidence needed for the claim. |
| **Reframe** | The claim should be narrowed because the implementation supports a weaker, more precise version. |

## Claim matrix

| Public claim | Current repo evidence | Status | Notes / required next proof |
|---|---|---|---|
| A VPM is a deterministic spatial view over a table of scored items. | `ScoreTable`, `LayoutRecipe`, `VPMArtifact`, deterministic `artifact_id`, binary-canonical identity bytes, and golden artifact-id test. | **Validated** | This is the strongest centre of the project. Keep this as the primary claim. |
| VPM cells map back to source evidence. | `VPMArtifact.cell()` returns source row, metric, raw value, and normalized value. Covered by `test_cell_maps_view_coordinates_to_source_coordinates`. | **Validated** | This supports inspectability. It does not by itself prove causal explanation. |
| Layout recipes reorganize the matrix task-by-task. | `LayoutRecipe` supports source, lexicographic, and weighted row ordering plus source/explicit column ordering. Tests cover lexicographic ordering and tie-breaking. | **Validated** | More recipes and examples would improve confidence, but the core mechanism is real. |
| The same dense table can support multiple policy views. | `ViewProfile`, `ViewSet`, `build_view()`, and `build_views()` build multiple deterministic VPMs from the same `ScoreTable`. Tests verify people/tree/car/risk views move different rows top-left while preserving the same source digest and cell/source mapping. | **Validated for deterministic multi-view artifacts** | Valid wording: “ZeroModel can build multiple deterministic policy views from the same dense source table while preserving source mapping.” |
| A bounded policy can be compiled into an addressable artifact and read without a model call. | `VPMPolicyLookup` treats rows as discretized states and metrics as actions. Tests cover action argmax, cell/source proof, action-column limiting, alias `SignReader`, deterministic replay, and a tiny arcade-shooter example that clears a wave and beats a random baseline. | **Validated for closed enumerable state/action policies** | Valid wording: “ZeroModel can compile a bounded policy into a deterministic VPM artifact and use runtime state to read the action sign without invoking a model.” Do not claim open-world generalization. |
| A Q-bearing policy can preserve criticality and decision margin as non-action evidence. | `with_q_diagnostics()` adds best-minus-worst `criticality` and best-minus-second-best `decision_margin`; `VPMPolicyLookup.evidence_metric_ids` returns those values while excluding them from argmax. Unit tests verify exact calculations and unchanged action selection across the 112-state example. | **Validated for finite Q-bearing score tables** | Use “VIPER-style criticality” only when the action columns carry Q-values or an equivalent consequence-bearing teacher signal. The original 0/1 arcade policy has a flat best-minus-worst surface; the criticality-first fixture therefore uses a separate Q-bearing teacher. |
| Named row-level properties can be checked exhaustively over a finite policy. | `PolicyPropertyChecker`, `PolicyPropertySpec`, and the `key-value-row-id/v1` decoder evaluate versioned declarative expressions across every source row. Tests cover a passing policy, an exact seeded `FIRE` counterexample, a repaired policy, typed row-ID scalars, JSON-null semantics, and contextual comparison errors. | **Validated for declarative finite row properties** | This is not general formal verification of temporal dynamics, continuous state spaces, liveness, or universal safety. |
| Verification results can become deterministic artifacts linked to the exact checked policy. | `PolicyVerificationReport.to_vpm()` stores checker version, property-spec digest, rows checked, violations, candidates, evidence, and coordinates. Provenance includes a `verifies` parent relation; tests cover deterministic identity and `.vpm` round-trip. | **Validated for finite property-result artifacts** | The artifact identifies which policy and properties were checked. It does not prove authorship, authorization, or that the property set is sufficient. For criticality-aware deployment, production traces cite the enriched artifact actually consumed by the reader. |
| Counterexample, repair, and re-verification can be represented as artifact lineage. | `examples/criticality_verification.py` creates original, unsafe, and repaired policy identities plus failed and passing verification artifacts. The seeded violation localizes to one declared row and action cell. | **Validated for the committed finite fixture** | Automatic repair is not implemented. The example demonstrates a recorded review loop, not autonomous policy correction. |
| A spatial optimizer can derive a view profile for a geometric objective. | `SpatialOptimizer`, `SpatialOptimizationResult`, `optimize_view_profile()`, and `build_optimized_view()` learn non-negative metric weights for an explicit top-left mass objective and emit a normal `ViewProfile`. Tests cover improved objective mass, source digest preservation, table-series input, and validation errors. | **Validated for explicit top-left mass objective** | Valid wording: “ZeroModel can derive a deterministic metric-weight view profile that improves an explicit top-left mass objective.” Do not claim it learns the semantically best view for every task. |
| A sequence of dense panels can become a temporal decision manifold. | `DecisionManifold`, `ManifoldFrame`, `ManifoldTransition`, `ManifoldSummary`, `build_decision_manifold()`, and `find_inflection_points()` build optimized VPM frames across a consistent panel sequence, track mass/weight/order changes, compute a metric graph, and surface inflection indices. Tests cover temporal view shifts, metric graph serialization, threshold/top-k inflection selection, and validation errors. | **Validated for deterministic temporal geometry** | Valid wording: “ZeroModel can summarize a sequence of dense scored panels as a deterministic decision manifold and surface frames with large spatial-view changes.” Do not claim semantic cause or universal change-point discovery. |
| No model at decision time. | `TopLeftGate` consumes an already-built artifact/field and thresholds a top-left region. `VPMPolicyLookup` consumes an already-built state/action artifact and returns an action. Tests cover both consumers. | **Validated for simple gates and closed policy lookup** | Valid wording: “a deployed consumer can make a bounded decision from a prepared artifact without invoking a model.” Avoid implying the artifact independently reasons. |
| Learning can be made visible. | `LearningObservation` and `build_learning_vpm()` require train, held-out, and regression observations before `learned=True`. Tests cover positive learning, tracking-without-heldout, and regression failure. | **Validated for scored traces** | Valid wording: “ZeroModel can make learning visible as deterministic before/after/held-out/regression artifact traces.” This does not prove a model’s internal mechanism changed. |
| Training progress can be visualized. | `TrainingCheckpoint` and `build_training_progress_vpm()` convert checkpoint telemetry into progress VPMs with train progress, held-out progress, regression safety, stability, efficiency, best checkpoint, warnings, and `learned`. Tests cover best checkpoint selection, overfit-like train-without-heldout warnings, regression failure, cell mapping, and validation errors. | **Validated for checkpoint telemetry** | Valid wording: “ZeroModel can turn model-training telemetry into deterministic progress artifacts.” This does not replace TensorBoard/W&B or prove internal model causality. |
| Tracker exports can feed training progress artifacts. | `zeromodel.adapters` parses generic JSON/JSONL/CSV, TensorBoard scalar CSV, W&B history JSONL/CSV/JSON, and Trackio JSON/JSONL/CSV exports into `TrainingCheckpoint` objects. Tests cover TensorBoard scalar grouping, W&B flat rows, Trackio nested JSON, and generic JSONL. | **Validated for exported files** | Valid wording: “ZeroModel can ingest dependency-light tracker exports.” Do not claim live SDK/API integration yet. |
| End-to-end fixtures can reproduce progress artifacts. | `tests/fixtures/training/` contains deterministic TensorBoard-style, W&B-style, Trackio-style, and generic telemetry fixtures. `test_research_readiness_fixtures.py` verifies best-checkpoint selection, warnings, VPM cell mapping, `.vpm` round-trip, and PNG/SVG rendering. | **Validated for synthetic fixtures** | This is research-readiness evidence, not a scale benchmark or real-world tracker validation. Next proof: sanitized real exports from actual runs. |
| Critic/evidence scores can become risk-first artifacts. | `CriticObservation`, `build_critic_vpm()`, and `observations_from_critic_lines()` convert Writer-style critic results and hallucination/policy/evidence scores into VPMs. Tests cover highest-risk ordering, warnings, Writer line-result conversion, cell mapping, and validation errors. | **Validated for scored critic traces** | Valid wording: “ZeroModel can turn critic/evidence/policy scores into deterministic risk-first artifacts for inspection.” This does not mean ZeroModel detects hallucinations by itself. |
| Task-aware top-left concentration. | `phos_sort_pack`, `pack_artifact`, `guarded_pack_artifact`, `top_left_concentration`, `ViewProfile`, `SpatialOptimizer`, and `DecisionManifold`. Guard selection now compares per-fraction improvement rather than raw concentration across different windows. | **Implemented / thin evidence** | The code can organize, measure, optimize, and track concentration over time for explicit objectives, but we still need task benchmarks showing inspection or decision improvement. |
| Compositional visual logic: AND/OR/NOT/XOR/add/subtract. | `zeromodel.compose` implements shape-checked fuzzy operators; tests cover AND/OR/XOR and comparison. | **Validated for numeric field operations** | Reframe as “explicit fuzzy field composition.” Do not call this symbolic reasoning unless a semantic layer is added and tested. |
| Deterministic reproducible provenance. | Artifacts include source and recipe digests, parents, provenance payload, cached identity bytes, and identity mismatch validation. Tests cover golden artifact ID, artifact ID determinism, tamper rejection, policy/verification parent links, and lineage fixtures. | **Validated for artifact identity and declared parent lineage** | Parent links are identity-bearing declarations, not signatures or external trust attestations. |
| Lossless `.vpm` artifact bundles. | `to_bundle()` writes a zip with `manifest.json`; `from_bundle()` validates bundle version and reconstructs `VPMArtifact`. Tests cover policy and verification-artifact identity round-trips. | **Validated** | The current bundle is a JSON manifest inside zip, not PNG metadata. |
| PNG as universal self-describing artifact. | `render.png_bytes()` writes a grayscale PNG image from a field. | **Reframe / not validated** | Current PNG is a rendering/transport image only. It is not self-describing and does not embed manifest/provenance. If we want this claim, add PNG metadata chunks plus round-trip tests. |
| PNG survives image pipelines. | No current tests for resize/crop/JPEG/social-media pipelines. | **Not validated** | Avoid this claim. Lossy transforms will likely destroy exact numeric semantics. |
| Hierarchical pyramids / zoomable navigation. | `build_pyramid()` reduces fields into levels with mean/max/sum. Test covers reduced level shapes. | **Implemented / thin evidence** | Supports hierarchy construction, not planet-scale navigation. Need pointer model, child tile IDs, resolver, traversal benchmark. |
| Planet-scale / infinite memory / 40-hop traversal. | No current benchmark or fixture proving the blog’s latency/scale claims. | **Not validated** | Treat as research ambition until benchmarks are in repo. Required: generated corpus, pyramid builder, resolver, traversal harness, hardware info, raw results. |
| Milliseconds on tiny hardware / 25KB RAM. | `TopLeftGate` and `VPMPolicyLookup` are tiny Python consumers, but there is no microcontroller implementation or benchmark. | **Not validated** | Required: C/Rust or MicroPython consumer, target hardware profile, timing and memory report. |
| Edge ↔ cloud symmetry. | Same artifact/field can be rendered, bundled, inspected, gated, and state-addressed by a policy lookup consumer. | **Implemented / thin evidence** | Need a concrete edge fixture and cloud viewer example using the same artifact bytes. |
| Multi-metric, multi-view by design. | `ScoreTable` supports multiple metrics; `LayoutRecipe` supports different ordering/column recipes; `ViewProfile` makes named policy views explicit; `SpatialOptimizer` can derive a metric-weight profile for a geometric objective; `DecisionManifold` tracks optimized views over time. | **Validated core mechanism** | The 1.0.11 fixture adds source-order and criticality-first views over one Q-bearing source. Next proof: benchmark whether criticality-first inspection improves a human or automated task. |
| Storage-agnostic routing via pointers. | No current pointer/resolver abstraction. | **Not validated** | Add `resolver` interfaces and tests before claiming storage-agnostic routing. |
| Traceable “thought” / 40+ levels. | Current code has provenance, controller signals, learning traces, training progress artifacts, tracker export adapters, end-to-end fixtures, critic evidence artifacts, policy diagnostics, finite verification artifacts, view profiles, spatial optimization, policy lookup, and decision manifolds, but no router frame chain or step graph implementation. | **Reframe / not validated** | Use the specific artifact name rather than “thought” unless strictly metaphorical. |
| Human-compatible explanations. | Cell/source mapping, policy-lookup decisions, evidence metrics, exact property counterexamples, SVG/PNG rendering, view profiles, spatial optimization summaries, temporal manifold summaries, and critic explanation metadata allow inspection. | **Implemented / thin evidence** | Reframe as “inspectable evidence mapping.” Explanation quality requires user-facing viewer tests and examples. |
| Cheap to adopt. | Package requires only NumPy; README shows short install and simple API. | **Supported, not benchmarked** | Add an end-to-end example from metric rows to gate/render/bundle in under 30 lines. |
| Works with your stack. | `metrics.pack_metrics()` accepts common aliases and builds score tables from metric rows; adapters ingest tracker logs; critic, policy diagnostics, finite property checking, views, spatial optimization, and manifolds operate over the shared artifact contract. | **Implemented for exported telemetry, critic traces, finite policies, dense views, spatial objectives, and temporal panels** | Add pandas adapters, real fixture exports, sanitized critic outputs, and optional live SDK integrations if broader wording is needed. |
| Great fits: anomaly detection, safety gates, code review traces, search triage, closed-world policy lookup. | Current package has primitives and early examples for training, learning, critic traces, multi-view artifacts, spatial optimization, temporal manifolds, finite policy lookup, and row-level verification. | **Implemented / thin evidence** | Do not promote finite row properties to safety certification. Create domain-specific fixtures and independent property sets first. |
| Viewer you’ll actually use. | Static site exists, but package has no bundled viewer or click-through pointer graph. | **Implemented as website demo only / thin evidence** | Add screenshot tests or a static demo fixture; add pointer graph before claiming Google-Maps-style navigation. |

## Current honest headline

The repo can honestly claim:

> ZeroModel turns scored data into deterministic, inspectable Visual Policy Map artifacts. For closed enumerable policies, those artifacts support state-addressed lookup without model invocation, Q-bearing evidence metrics, exhaustive named row properties, exact counterexamples, and verification artifacts linked to the exact policy checked. The same artifact contract also supports deterministic views, bundles, spatial and temporal consumers, learning and training traces, and critic evidence maps.

## Claims to avoid until benchmarks exist

Avoid or soften these phrases in README/package copy until the repository contains reproducible evidence:

- “planet-scale”
- “infinite memory”
- “constant-time decision”
- “milliseconds on tiny hardware”
- “25KB RAM”
- “survives image pipelines”
- “watch AI think” as a literal claim
- “visual logic equals reasoning”
- “self-describing PNG” unless metadata chunks are implemented
- “understands why the model learned”
- “connects to every tracker automatically”
- “validated on real training runs” until sanitized real exports are added
- “detects hallucinations by itself”
- “automatically learns the best view for every task” until task benchmarks exist
- “improves decision accuracy” except for narrow tested fixtures
- “finds the forty important steps in any dataset” until manifold benchmarks exist
- “formally verified policy” when only finite row-level properties were checked
- “VIPER criticality” for arbitrary action scores without Q-value or equivalent consequence semantics
- “safe policy” when only the published property set passed

## Recommended validation backlog

1. **Dynamic policy verification** — extend [C60] with transition-system properties and compare finite scans with an SMT-backed checker on the same fixture.
2. **Criticality-weighted representation benchmark** — test [C62] by comparing uniform and criticality-weighted allocation under a fixed artifact budget.
3. **Human inspection study** — test [C22], [C23], and [C24] by comparing source-order, criticality-first, table, and log views for counterexample detection time and accuracy.
4. **PNG metadata round-trip** — test [C49] by embedding manifest/provenance in PNG text chunks and proving decode returns the same artifact ID.
5. **Hierarchy traversal benchmark** — test [C25] and [C36] with a generated corpus, child pointers, resolver, traversal harness, and raw timing report.
6. **Edge benchmark** — test [C33], [C34], and [C35] with a minimal no-model consumer, measured memory, and latency distributions on named hardware.
7. **Decision manifold benchmark** — test [C20], [C29], and [C30] against known synthetic change points and random/scalar baselines.
8. **Signed verification lineage** — extend [C46] by combining artifact and verification identities with Sigstore or in-toto attestations.
9. **Sanitized critic exports** — test [C21], [C42], and [C43] with real critic or RAG-verifier outputs checked into fixtures.
10. **Sanitized real tracker exports** — strengthen [C15], [C16], and [C28]–[C32] with exports from actual training runs.
11. **Examples package** — strengthen [C14], [C21], and [C42] with search-triage, safety-gate, anomaly, and code-review fixtures.
12. **Policy-lookup portability** — test [C33]–[C35] with a C, Rust, or MicroPython consumer over a committed policy artifact.

## Bottom line

The cleaned repo now validates the core artifact abstraction, deterministic multi-view and spatial/temporal consumers, closed-world policy lookup, Q-bearing criticality and decision-margin evidence, exhaustive finite row-property checking, linked verification artifacts, exact counterexample localization, and a counterexample-repair-reverification lineage fixture. It does **not** validate general formal verification, automatic repair, safety certification, criticality-weighted approximation, open-world decision accuracy, tiny-hardware performance, or improved human inspection. The next work should test those claims directly rather than borrowing evidence from the finite arcade fixture.
