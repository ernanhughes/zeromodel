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
| A VPM is a deterministic spatial view over a table of scored items. | `ScoreTable`, `LayoutRecipe`, `VPMArtifact`, deterministic `artifact_id`, canonical JSON identity payload. Covered by `test_build_vpm_is_deterministic`. | **Validated** | This is the strongest centre of the project. Keep this as the primary claim. |
| VPM cells map back to source evidence. | `VPMArtifact.cell()` returns source row, metric, raw value, and normalized value. Covered by `test_cell_maps_view_coordinates_to_source_coordinates`. | **Validated** | This supports inspectability. It does not by itself prove causal explanation. |
| Layout recipes reorganize the matrix task-by-task. | `LayoutRecipe` supports source, lexicographic, and weighted row ordering plus source/explicit column ordering. Tests cover lexicographic ordering and tie-breaking. | **Validated** | More recipes and examples would improve confidence, but the core mechanism is real. |
| No model at decision time. | `TopLeftGate` consumes an already-built artifact/field and thresholds a top-left region. Covered by `test_edge_gate_evaluates_without_model`. | **Validated for simple gates** | Valid wording: “a deployed consumer can make a threshold decision from a prepared artifact without invoking a model.” Avoid implying the artifact independently reasons. |
| Learning can be made visible. | `LearningObservation` and `build_learning_vpm()` require train, held-out, and regression observations before `learned=True`. Tests cover positive learning, tracking-without-heldout, and regression failure. | **Validated for scored traces** | Valid wording: “ZeroModel can make learning visible as deterministic before/after/held-out/regression artifact traces.” This does not prove a model’s internal mechanism changed. |
| Training progress can be visualized. | `TrainingCheckpoint` and `build_training_progress_vpm()` convert checkpoint telemetry into progress VPMs with train progress, held-out progress, regression safety, stability, efficiency, best checkpoint, warnings, and `learned`. Tests cover best checkpoint selection, overfit-like train-without-heldout warnings, regression failure, cell mapping, and validation errors. | **Validated for checkpoint telemetry** | Valid wording: “ZeroModel can turn model-training telemetry into deterministic progress artifacts.” This does not replace TensorBoard/W&B or prove internal model causality. |
| Tracker exports can feed training progress artifacts. | `zeromodel.adapters` parses generic JSON/JSONL/CSV, TensorBoard scalar CSV, W&B history JSONL/CSV/JSON, and Trackio JSON/JSONL/CSV exports into `TrainingCheckpoint` objects. Tests cover TensorBoard scalar grouping, W&B flat history rows, Trackio nested JSON, and generic JSONL. | **Validated for exported files** | Valid wording: “ZeroModel can ingest dependency-light tracker exports.” Do not claim live SDK/API integration yet. |
| End-to-end fixtures can reproduce progress artifacts. | `tests/fixtures/training/` contains deterministic TensorBoard-style, W&B-style, Trackio-style, and generic telemetry fixtures. `test_research_readiness_fixtures.py` verifies best-checkpoint selection, warnings, VPM cell mapping, `.vpm` round-trip, and PNG/SVG rendering. | **Validated for synthetic fixtures** | This is research-readiness evidence, not a scale benchmark or real-world tracker validation. Next proof: sanitized real exports from actual runs. |
| Critic/evidence scores can become risk-first artifacts. | `CriticObservation`, `build_critic_vpm()`, and `observations_from_critic_lines()` convert Writer-style critic results and hallucination/policy/evidence scores into VPMs. Tests cover highest-risk ordering, warnings, Writer line-result conversion, cell mapping, and validation errors. | **Validated for scored critic traces** | Valid wording: “ZeroModel can turn critic/evidence/policy scores into deterministic risk-first artifacts for inspection.” This does not mean ZeroModel detects hallucinations by itself. |
| Task-aware top-left concentration. | `phos_sort_pack`, `pack_artifact`, `guarded_pack_artifact`, and `top_left_concentration`; covered by `test_phos_guarded_pack_measures_concentration`. | **Implemented / thin evidence** | The code measures and guards concentration, but we need benchmark fixtures showing improvement across realistic datasets. |
| Compositional visual logic: AND/OR/NOT/XOR/add/subtract. | `zeromodel.compose` implements shape-checked fuzzy operators; tests cover AND/OR/XOR and comparison. | **Validated for numeric field operations** | Reframe as “explicit fuzzy field composition.” Do not call this symbolic reasoning unless a semantic layer is added and tested. |
| Deterministic reproducible provenance. | Artifacts include source and recipe digests, parents, provenance payload, and identity mismatch validation. Tests cover artifact ID and tamper rejection. | **Validated for artifact identity** | Add tests for parent lineage, multi-artifact provenance graphs, and replay from bundles. |
| Lossless `.vpm` artifact bundles. | `to_bundle()` writes a zip with `manifest.json`; `from_bundle()` validates bundle version and reconstructs `VPMArtifact`. Test covers identity round-trip. | **Validated** | The current bundle is a JSON manifest inside zip, not PNG metadata. |
| PNG as universal self-describing artifact. | `render.png_bytes()` writes a grayscale PNG image from a field. | **Reframe / not validated** | Current PNG is a rendering/transport image only. It is not self-describing and does not embed manifest/provenance. If we want this claim, add PNG metadata chunks plus round-trip tests. |
| PNG survives image pipelines. | No current tests for resize/crop/JPEG/social-media pipelines. | **Not validated** | Avoid this claim. Lossy transforms will likely destroy exact numeric semantics. |
| Hierarchical pyramids / zoomable navigation. | `build_pyramid()` reduces fields into levels with mean/max/sum. Test covers reduced level shapes. | **Implemented / thin evidence** | Supports hierarchy construction, not planet-scale navigation. Need pointer model, child tile IDs, resolver, traversal benchmark. |
| Planet-scale / infinite memory / 40-hop traversal. | No current benchmark or fixture proving the blog’s latency/scale claims. | **Not validated** | Treat as research ambition until benchmarks are in repo. Required: generated corpus, pyramid builder, traversal harness, hardware info, raw results. |
| Milliseconds on tiny hardware / 25KB RAM. | `TopLeftGate` is tiny Python logic, but there is no microcontroller implementation or benchmark. | **Not validated** | Required: C/Rust or MicroPython consumer, target hardware profile, timing and memory report. |
| Edge ↔ cloud symmetry. | Same artifact/field can be rendered, bundled, inspected, and gated. | **Implemented / thin evidence** | Need a concrete edge fixture and cloud viewer example using the same artifact bytes. |
| Multi-metric, multi-view by design. | `ScoreTable` supports multiple metrics; `LayoutRecipe` supports different ordering/column recipes. | **Validated core mechanism** | Add examples showing two recipes over the same source table and asserting source digest is unchanged. |
| Storage-agnostic routing via pointers. | No current pointer/resolver abstraction. | **Not validated** | Add `resolver` interfaces and tests before claiming storage-agnostic routing. |
| Traceable “thought” / 40+ levels. | Current code has provenance, controller signals, learning traces, training progress artifacts, tracker export adapters, end-to-end fixtures, and critic evidence artifacts, but no router frame chain or step graph implementation. | **Reframe / not validated** | Use “traceable artifact lineage,” “learning trace,” “training progress artifact,” “tracker export adapter,” “critic evidence artifact,” or “research fixture” when those are the actual artifacts. Avoid “thought” unless strictly metaphorical. |
| Human-compatible explanations. | Cell/source mapping, SVG/PNG rendering, and critic explanation metadata allow inspection. | **Implemented / thin evidence** | Reframe as “inspectable evidence mapping.” Explanation quality requires user-facing viewer tests and examples. |
| Cheap to adopt. | Package requires only NumPy; README shows short install and simple API. | **Supported, not benchmarked** | Add an end-to-end example from metric rows to gate/render/bundle in under 30 lines. |
| Works with your stack. | `metrics.pack_metrics()` accepts common aliases and builds score tables from metric rows; `zeromodel.adapters` ingests exported tracker logs; `zeromodel.critic` ingests Writer-style critic outputs. | **Implemented for exported telemetry and critic traces** | Add pandas adapters, real fixture exports, sanitized Writer critic outputs, and optional live SDK integrations if broader wording is needed. |
| Great fits: anomaly detection, safety gates, code review traces, search triage. | Current package has primitives and early domain examples for training, learning, and critic traces. | **Implemented / thin evidence** | Create examples for anomaly detection, safety gates, code review traces, and search triage before promoting them as out-of-the-box fits. |
| Viewer you’ll actually use. | Static site exists, but package has no bundled viewer or click-through pointer graph. | **Implemented as website demo only / thin evidence** | Add screenshot tests or a static demo fixture; add pointer graph before claiming Google-Maps-style navigation. |

## Current honest headline

The repo can honestly claim:

> ZeroModel turns scored data into deterministic, inspectable Visual Policy Map artifacts. Those artifacts preserve source mapping, deterministic identity, provenance digests, renderable fields, bundle round-trips, small consumer decisions such as top-left gates without invoking a model at decision time, scored learning traces that distinguish tracking from train/held-out/regression evidence of learning, checkpoint-level training progress artifacts for model telemetry, dependency-light adapters for tracker exports, committed end-to-end fixtures for reproducing the training-progress path, and critic/evidence/policy risk artifacts for inspection.

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

## Recommended validation backlog

1. **PNG metadata round-trip** — embed manifest/provenance in PNG text chunks and prove decode returns the same artifact ID.
2. **Hierarchy traversal benchmark** — generated corpus, child pointers, resolver, traversal harness, raw timing report.
3. **Edge benchmark** — minimal no-model gate in a tiny runtime, with memory and timing measurements.
4. **Multi-view invariant test** — same source table, multiple recipes, identical source digest, different view ordering.
5. **Lineage/provenance replay** — parent artifact chain and deterministic replay from bundles.
6. **Sanitized critic exports** — real Writer critic or RAG verifier outputs checked into tests or docs.
7. **Learning trace domain examples** — agent trace learning, RAG correction learning, Writer evaluator learning.
8. **Sanitized real tracker exports** — real TensorBoard, W&B, and Trackio exports checked into tests or docs.
9. **Examples package** — search triage, safety gate, anomaly toy dataset, code review trace.
10. **Website claim labels** — mark claims as implemented, benchmarked, or roadmap in the static site.

## Bottom line

The cleaned repo now validates the core abstraction, adds a concrete scored-trace definition of visible learning, adds checkpoint-level model-training progress artifacts, can ingest common tracker export files, includes deterministic end-to-end fixtures, and can artifactize critic/evidence/policy scores. It does **not** yet validate the strongest blog-scale performance claims or real-world hallucination detection. The next work should be research design and real-world evidence, not broadening the API.
