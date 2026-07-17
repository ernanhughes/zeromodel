# ZeroModel Visual Addressing Research Status

**Recommended repository path:** `docs/research/ZeroModel-visual-address-research-status.md`  
**Status date:** 17 July 2026  
**Document status:** Research-status record  
**Scope:** Closed-world visual observation addressing, the Phase 1 held-out benchmark, the full pinned DINOv2 result, and the resulting research decision  
**Validation status:** Research  
**Primary result document:** `docs/results/visual-address-phase-one-dinov2-full.md`  
**Earlier design record:** `docs/research/visual-address-phase-one.md`

---

## 1. Purpose

This document records where ZeroModel's visual-address research currently stands.

It is intended to prevent three common failures:

1. treating implemented machinery as validated capability;
2. treating a failed learned baseline as if the wider ZeroModel project failed;
3. continuing to add visual models without revising the research question.

The document separates:

- what has been implemented;
- what has been exhaustively validated;
- what the full benchmark measured;
- what the evidence supports;
- what remains uncertain;
- which research hypotheses should now be stopped, revised, or tested next.

This is not a promotional document. It is an evidence and decision record.

---

## 2. Executive status

ZeroModel has completed the first full scientific cycle for learned visual policy addressing.

The cycle was:

```text
bounded policy
    ↓
exact symbolic addressing
    ↓
deterministic canonical visual addressing
    ↓
provider-neutral visual-address contracts
    ↓
family-held-out benchmark
    ↓
simple and learned baselines
    ↓
full pinned DINOv2 run
    ↓
hypothesis adjudication
```

The principal Phase 1 hypothesis was:

> A pinned frozen visual representation can map held-out benign observations to the correct governed policy address while rejecting distinguishable invalid observations.

The full experiment did not support that hypothesis.

The result is not merely that DINOv2 performed poorly. The more important result is:

> General-purpose global visual representations can preserve broad action-level similarity while losing the exact local distinctions required for governed policy-row identity and safe rejection.

The strongest learned system, DINOv2 all-prototype k-NN, obtained slightly higher benign action accuracy than the normalized-pixel baseline:

```text
D benign action accuracy: 74.78%
B benign action accuracy: 73.44%
difference:                1.34 percentage points
```

However, D was substantially worse on the actual governed-address task:

```text
B exact benign row accuracy: 62.50%
D exact benign row accuracy: 30.36%

B false-acceptance rate:     51.21%
D false-acceptance rate:     73.79%

B accepted-action correctness: 97.92%
D accepted-action correctness: 77.49%
```

The current learned retrieval path therefore fails continuation criteria.

The project has nevertheless discovered a sharper research problem:

> The unresolved boundary is not policy lookup. It is trustworthy compilation of visual evidence into an explicit observed state.

The recommended next hypothesis is a factorized **Visual State Compiler**, not another whole-image nearest-neighbour reader.

---

## 3. Where this sits within ZeroModel

ZeroModel's visual work must be understood as layers with different evidence states.

| Layer | Current status |
|---|---|
| Deterministic VPM artifact construction | Validated |
| Cell-to-source evidence mapping | Validated |
| Closed enumerable policy compilation | Validated |
| Exact state-addressed policy lookup without a model call | Validated |
| Canonical exact visual feature-codeword addressing | Validated in the committed arcade fixture |
| Provider-neutral visual-address contract | Validated for contract mechanics and deterministic conformance |
| Identity-bearing dense representations and manifests | Validated as storage and governance contracts |
| Family-held-out visual benchmark machinery | Validated as benchmark infrastructure |
| Pinned DINOv2 representation path | Implemented and now measured |
| Frozen DINOv2 governed row addressing | Not supported by the full result |
| Safe learned rejection of critical interventions and OOD frames | Refuted at the tested operating point |
| Open-world visual perception | Not validated |
| Real-world deployment value | Not validated |

The distinction is essential:

```text
ZeroModel's compiled policy core works.

The tested learned visual observation provider does not work safely enough.
```

The failure belongs to one observation architecture, not to the entire artifact and policy architecture.

---

## 4. Research progression

### 4.1 Symbolic policy compilation

The bounded arcade shooter provides a complete, enumerable policy surface:

- seven tank positions;
- seven possible target positions plus target absence;
- two cooldown states;
- four actions;
- 112 policy rows.

The exact symbolic runtime state can address the correct row in the compiled policy artifact. The artifact then returns an action and an inspectable decision trace without invoking a learned reasoning model.

This is the strongest established ZeroModel result in this line of work.

### 4.2 Canonical deterministic visual addressing

The next step rendered every bounded state as a deterministic `uint8` frame.

The committed renderer uses:

- frame height: 16 pixels;
- frame width: 28 pixels for the seven-cell fixture;
- integer-only drawing;
- visible tank position;
- visible target position or absence;
- visible cooldown state.

A deterministic feature contract then:

- converts to canonical grayscale;
- applies exact integer box pooling;
- quantizes the result;
- builds a complete visual codebook;
- binds the visual index to the exact policy artifact;
- supports exact feature lookup and calibrated refusal.

This path recovers all 112 canonical frames and remains action-equivalent to the symbolic policy across the committed exhaustive wave fixture.

Its honest boundary is also now clear:

> It validates exact canonical feature-codeword addressing, not accepted tolerance to ordinary non-exact image variation.

### 4.3 Governed visual-address seam

The project then separated representation extraction from policy lookup.

The current contract layer includes concepts equivalent to:

```text
ImageObservation
VisualAddressContract
VisualAddressDecision
VisualAddressProvider
VisualPolicyReader
EncoderManifest
MatrixBlob
VisualAddressManifest
DeploymentBinding
VisualBenchmarkReport
```

This separation allows different observation providers to address the same independently identified policy artifact.

That architectural seam remains valuable even though the first learned provider failed.

### 4.4 Phase 1 held-out benchmark

The benchmark introduced:

- deterministic corruption families;
- disjoint prototype, calibration, and benign-test families;
- distinguishable critical-evidence interventions;
- information-theoretic controls;
- OOD frames;
- explicit false-accept and false-reject denominators;
- per-family counts;
- optional per-observation traces;
- simple and learned baselines.

### 4.5 Full pinned learned run

A full local run was completed with `variants_per_family = 3`.

Run identity:

```text
dataset_digest:
91b1b422482eeeef20eb182162eb2a745f9b50524cc7f94ec95a0aba5f2fa37e

report_digest:
d7d0b4db13c9f96b2ac4583aae25fa2159fb62471fb90110103548317f084035

total observations:
4,378

scored observations:
1,592

validation_status:
research
```

The original generated report and complete environment manifest must still be committed as repository evidence.

---

## 5. Systems evaluated

| ID | System | Role |
|---|---|---|
| **A** | Existing deterministic reader | Exact canonical codeword baseline with refusal |
| **B** | Mean-centred L2-normalized grayscale pixels with medoid retrieval | Simple no-model approximate baseline |
| **C** | Pinned DINOv2-small CLS embedding with one medoid per policy row | Frozen global embedding retrieval |
| **D** | Pinned DINOv2-small CLS embedding with every prototype retained | Higher-capacity frozen k-NN retrieval |
| **G** | Ridge least-squares row classifier over the same DINOv2 vectors | Simple learned row classifier with calibrated rejection |

Pinned encoder identity:

```text
model:
facebook/dinov2-small

revision:
ed25f3a31f01632728cabb09d1542f84ab7b0056

representation:
L2-normalized CLS token from last_hidden_state

preprocessing:
crop-safe square letterboxing followed by the pinned processor
```

The square-letterbox step is important because directly applying the default processor to the wide 16×28 arcade frame could crop policy-relevant side pixels.

---

## 6. Dataset composition

The full run contains:

| Partition | Count | Evaluation role |
|---|---:|---|
| Prototype families | 1,344 | Build the index or fit the probe |
| Calibration families | 1,344 | Set per-row thresholds and margins |
| Held-out benign test families | 1,344 | Expected accept |
| Target-removal information-theoretic controls | 98 | Report separately; no scored disposition |
| Distinguishable critical interventions | 224 | Expected reject |
| OOD observations | 24 | Expected reject |
| **Total** | **4,378** | |
| **Scored evaluation total** | **1,592** | |

### Prototype families

- clean intensity variation;
- lower brightness;
- one-pixel upward translation;
- palette A.

### Calibration families

- held-out contrast;
- one-pixel downward translation;
- palette B;
- low-amplitude integer noise.

### Benign test families

- stronger unseen brightness;
- two-pixel translation;
- palette C;
- a structured patch placed only over source-background pixels.

### Distinguishable critical interventions

- tank removed;
- cooldown indicator removed.

### Information-theoretic control

- target removed when a target exists.

The target-removal control is not a scored rejection case because its pixels can be identical to a valid no-target state. No single-frame reader can distinguish those two hidden histories from the pixels alone.

### OOD families

- blank frame;
- checkerboard;
- impossible two-tank frame.

No corruption family appears in more than one prototype, calibration, or test split.

---

## 7. Full benchmark results

### 7.1 Primary metrics

The console's legacy `action_accuracy` and `row_accuracy` divide by all 1,592 scored cases, including expected rejections. The research interpretation therefore uses explicit benign denominators for row and action recovery.

| System | Benign action accuracy | Exact benign row accuracy | FAR | FRR | Conflicting-action errors |
|---|---:|---:|---:|---:|---:|
| **A** | 0.00% | 0.00% | 0.00% | 100.00% | 0 |
| **B** | 73.44% | 62.50% | 51.21% | 25.00% | 21 |
| **C** | 70.01% | 30.73% | 82.26% | 3.27% | 359 |
| **D** | 74.78% | 30.36% | 73.79% | 3.50% | 292 |
| **G** | 59.45% | 28.79% | 100.00% | 1.79% | 521 |

### 7.2 Acceptance quality

| System | Accepted benign cases | Correct action when accepted | Exact row when accepted |
|---|---:|---:|---:|
| **A** | 0 | — | — |
| **B** | 1,008 | 97.92% | 83.33% |
| **C** | 1,300 | 72.38% | 31.77% |
| **D** | 1,297 | 77.49% | 31.46% |
| **G** | 1,320 | 60.53% | 29.32% |

### 7.3 Correct action from the wrong row

| System | Correct benign actions | Exact benign rows | Correct action from a wrong row |
|---|---:|---:|---:|
| **A** | 0 | 0 | 0 |
| **B** | 987 | 840 | 147 |
| **C** | 941 | 413 | 528 |
| **D** | 1,005 | 408 | 597 |
| **G** | 799 | 387 | 412 |

D's headline action score is therefore not equivalent to policy-address recovery. Of its 1,005 correct benign actions, 597 came from the wrong policy row.

### 7.4 Rejection result

| System | False accepts | Rejection opportunities | FAR | Correct rejections |
|---|---:|---:|---:|---:|
| **A** | 0 | 248 | 0.00% | 248 |
| **B** | 127 | 248 | 51.21% | 121 |
| **C** | 204 | 248 | 82.26% | 44 |
| **D** | 183 | 248 | 73.79% | 65 |
| **G** | 248 | 248 | 100.00% | 0 |

All approximate readers failed the rejection requirement at the measured operating point.

---

## 8. What the experiment discovered

### 8.1 Action prediction and policy addressing are different tasks

The policy contains 112 rows but only four actions.

Many incorrect rows share the same action. This makes action accuracy an incomplete and potentially misleading measure for governed systems.

A correct action from the wrong row loses:

- exact policy identity;
- row-specific evidence;
- replay fidelity;
- exact alternatives;
- policy-version-specific provenance;
- the ability to inspect why that row was addressed.

For ZeroModel's present contract, exact row identity is not incidental metadata. It is part of the claimed mechanism.

This does not prove that exact rows are always required in every application. It establishes that the current experiment must not silently redefine success from exact addressing to action-equivalent classification after seeing the result.

### 8.2 Semantic invariance can oppose governed addressing

DINOv2 is a general-purpose semantic representation.

Such representations are designed to preserve meaning across visual variation. That is useful for recognition, but governed addressing may require preserving small local differences that change:

- state identity;
- evidence presence;
- structural validity;
- permitted action.

The benchmark suggests the following distinction:

```text
semantic recognition:
    "What general kind of scene is this?"

governed visual addressing:
    "Which exact decision state is visibly supported,
     and is every required fact present?"
```

A representation can perform reasonably on the first question and poorly on the second.

### 8.3 Rejection is the dominant unresolved problem

The learned systems did not fail because they never found neighbours.

They failed because similarity systems always produce:

- a nearest candidate;
- a highest class score;
- a margin;
- some apparently plausible address.

None of those quantities establishes that the observation belongs to the valid state surface.

The lower FRR of C, D, and G was purchased mainly by accepting broadly. Their high FAR demonstrates that acceptance coverage and safety must be evaluated separately.

### 8.4 Exact deterministic addressing has a precise boundary

System A rejected every benign perturbation and every invalid frame.

This confirms two things simultaneously:

- exact canonical codeword addressing is strict and safe inside its declared observation contract;
- it is not a tolerant visual recognizer.

The result narrows the claim rather than invalidating it.

### 8.5 The provider seam survived the failed provider

The following ideas remain useful independently of DINOv2:

- separate observation-provider and policy identities;
- explicit score polarity;
- calibrated refusal;
- content-addressed representations;
- prototype-to-policy bindings;
- benchmark-report identity;
- deployment binding;
- provider-neutral delegation into the exact policy.

This is evidence that the architecture can absorb negative provider results without corrupting the policy layer.

---

## 9. Strong conclusions

The following conclusions are supported by the current evidence.

### 9.1 The full frozen-embedding Phase 1 path should not be promoted

C, D, and G do not satisfy the current governed-address and rejection requirements.

### 9.2 Normalized pixels are a stronger addressing baseline than DINOv2 on this fixture

B is not safe enough to deploy, but it provides:

- far better exact-row recovery;
- far fewer conflicting-action errors;
- much higher accepted-decision reliability;
- lower FAR than every learned system.

D's 1.34-point benign action advantage does not compensate for its address and rejection regressions.

### 9.3 The linear probe is eliminated

G provides no useful complexity or performance trade:

- 59.45% benign action accuracy;
- 28.79% exact row accuracy;
- 100% FAR;
- 521 conflicting-action errors.

### 9.4 Current global retrieval does not establish visual generalization

The full result is evidence against the tested hypothesis, not evidence of a successful learned visual reader.

### 9.5 Phase 1 produced a successful research outcome

The project:

- declared kill conditions before the result;
- built the benchmark before tuning;
- measured simple baselines;
- preserved rejection as a first-class metric;
- obtained an unfavourable result;
- allowed the result to stop an attractive direction.

That is a stronger outcome than tuning until an ambiguous success could be claimed.

---

## 10. Conclusions that remain provisional

These are plausible interpretations, not established facts.

### 10.1 Global semantic embeddings may be structurally mismatched to exact policy addressing

One encoder and one global representation were tested. The result does not establish a universal impossibility theorem for learned representations.

### 10.2 A factorized state compiler is probably a better architecture

The evidence points toward explicit factor extraction, but no factorized baseline has yet been implemented or measured.

### 10.3 Patch-level tokens may preserve useful local evidence

Only the DINOv2 CLS token was tested. Patch tokens may behave differently, but this remains unmeasured.

### 10.4 A task-specific representation may recover exact rows better

No supervised contrastive row representation, small CNN, detector, or multi-head factor model was tested.

### 10.5 The measured operating point may not be the best possible point

The run uses the declared empirical lower-quantile calibration with default quantile `0.0`. Complete FAR/FRR operating curves have not yet been committed.

These uncertainties justify careful diagnostic work. They do not justify claiming that the current learned architecture succeeded.

---

## 11. What the experiment does not establish

The result does not prove:

- that every learned visual representation will fail;
- that DINOv2 is generally poor;
- that exact row identity is always the correct product requirement;
- that a different threshold cannot improve the operating trade-off;
- that a small task-specific model cannot solve the fixture;
- that local patch evidence cannot help;
- that temporal evidence cannot resolve ambiguous observations;
- that the current benchmark perfectly represents a real deployment;
- that ZeroModel should become a computer-vision framework;
- that the visual direction has operational value outside the synthetic fixture.

---

## 12. Kill-condition adjudication

The Phase 1 document declared six kill or reduction conditions.

| Condition | Status |
|---|---|
| Normalized pixels or deterministic matching match frozen embeddings in the operating region | **Triggered in the relevant trade-off** |
| Linear probe matches retrieval at lower complexity | **Not triggered; G is worse and is eliminated** |
| Critical interventions are accepted at an unsafe rate | **Triggered** |
| Held-out calibration does not transfer | **Not supported as adequate; likely triggered, pending full curves and per-family analysis** |
| Lightweight governance wrapper reaches parity | **Not evaluated** |
| Prototype count approaches one per observation without compensating benefit | **Strong warning for D** |

### Decision

```text
Do not continue the current global DINOv2 retrieval path as the promoted
ZeroModel visual architecture.
```

This does not require deleting the implementation. Systems B, C, D, and G should remain as reproducible research baselines.

---

## 13. Claim-status transitions

| Claim | New status |
|---|---|
| A bounded policy can be compiled and addressed symbolically | Validated |
| Canonical exact visual feature codewords recover the committed rows | Validated |
| The deterministic reader tolerates the held-out benign families | Refuted for those families |
| Normalized pixels provide safe approximate addressing | Refuted at the tested operating point |
| DINOv2 medoids improve governed policy addressing | Not supported |
| DINOv2 all-prototype k-NN materially beats normalized pixels | Not supported |
| A ridge linear probe matches retrieval at lower complexity | Refuted |
| The tested approximate readers reject distinguishable invalid states safely | Refuted at the tested operating point |
| The visual-address provider seam supports multiple governed providers | Validated as architecture |
| Learned visual addressing is deployment-ready | Not supported |
| ZeroModel provides general image understanding | Not supported |

The claims audit should distinguish:

```text
not implemented
implemented but unmeasured
measured and unsupported
measured and refuted
validated within a bounded claim
```

Those states should not be collapsed into a single “not validated” label.

---

## 14. Current architecture decision

### Retain

- the independently identified policy artifact;
- the exact symbolic policy reader;
- the canonical deterministic visual reader;
- the provider-neutral address contract;
- encoder and preprocessing manifests;
- `MatrixBlob`;
- prototype-to-policy bindings;
- explicit rejection decisions;
- family-held-out benchmark machinery;
- information-theoretic controls;
- systems A, B, C, D, and G as baselines.

### Do not promote

- DINOv2 CLS medoid retrieval;
- DINOv2 all-prototype k-NN;
- the ridge row probe;
- a claim of tolerant learned visual addressing;
- a claim of critical-evidence detection;
- physical control;
- open-world perception;
- Level 2 representation complexity added merely to rescue the current result.

### Fix before treating the result as a complete repository record

- commit the original full JSON report;
- capture the exact local environment manifest;
- add benign-denominator metrics to the primary result schema;
- add Wilson intervals or another declared interval method;
- commit raw per-family results;
- generate paired per-observation comparison data;
- fix no-conflicting-action calibration/runtime semantics;
- scope float-vector representation digests to the exact execution environment;
- update the claims audit.

---

## 15. Revised research question

The original Phase 1 question was:

> Can a whole-image representation directly recover the governed policy row under held-out visual variation?

The revised question should be:

> Can visible policy-critical facts be independently established, rejected when absent or ambiguous, assembled into a typed observed state, and then delegated to the exact compiled ZeroModel policy?

This produces a different architecture:

```text
ImageObservation
    ↓
evidence detectors
    ↓
EvidenceBundle
    ↓
TypedObservedState
    ↓
exact state encoder
    ↓
VPMPolicyLookup
    ↓
action + complete trace
```

The unresolved research boundary becomes:

```text
pixels
    →
trustworthy symbolic evidence
```

ZeroModel already has a strong answer after the state becomes explicit.

---

## 16. Proposed Phase 2: Visual State Compiler

### 16.1 Concept

A Visual State Compiler would not retrieve a complete policy-row image.

It would independently establish fields such as:

```text
tank_present
tank_count
tank_position
target_present
target_position
cooldown_state
frame_structurally_valid
```

Each field should carry:

```text
value
status
confidence or calibrated score
evidence region
detector identity
calibration identity
rejection reason
```

The state assembler would reject when:

- a required field is missing;
- evidence is contradictory;
- more than one tank is detected;
- a coordinate is out of range;
- a field is ambiguous;
- an impossible combination is observed;
- a temporal dependency cannot be resolved from one frame.

### 16.2 Why this is better aligned with ZeroModel

It preserves the project's strongest properties:

- explicit state;
- exact policy lookup;
- inspectable evidence;
- local failure reasons;
- provider identity;
- replay;
- deterministic state assembly;
- refusal rather than forced matching.

It also gives causal diagnostics. When the final policy row is wrong, the system can identify which extracted fact failed.

### 16.3 Initial provider candidates

The first factorized benchmark should include several deliberately simple providers:

1. direct renderer or engine instrumentation;
2. deterministic connected-component and geometric extraction;
3. local template matching;
4. small task-specific classifiers per factor;
5. DINOv2 patch-token probes;
6. a compact multi-head task-specific vision model.

The direct-instrumentation baseline is mandatory. In a bounded installation, direct sensors may be safer and cheaper than any visual pipeline.

---

## 17. Proposed Phase 2 metrics

| Layer | Required metrics |
|---|---|
| Tank presence | precision, recall, abstention |
| Tank count | exact count accuracy |
| Tank position | exact cell accuracy |
| Target presence | precision, recall, abstention |
| Target position | exact cell accuracy |
| Cooldown | exact state accuracy and missing-evidence detection |
| Structural validity | invalid-frame precision and recall |
| State assembly | exact complete-state accuracy |
| Policy address | exact row accuracy |
| Policy action | action accuracy |
| Safety | FAR, FRR, and risk-weighted loss |
| Evidence | region correctness and trace completeness |
| Calibration | reliability and held-out family transfer |
| Efficiency | runtime, memory, stored examples, calibration effort |

The benchmark must preserve separate counts for:

- correct exact state;
- correct action from a wrong state;
- safe rejection;
- unsafe acceptance;
- impossible single-frame cases.

---

## 18. Options from here

### Option A — Close visual research temporarily

Commit the evidence, update claims, preserve the baselines, and return focus to the core ZeroModel artifact programme.

**Use when:** visual perception is not required for the next project milestone.

### Option B — Produce a failure atlas

Analyze per-observation traces and generate:

- row confusion matrices;
- action-equivalent row clusters;
- per-family operating curves;
- score and margin distributions;
- nearest-neighbour atlases;
- B-success/D-failure cases;
- D-success/B-failure cases;
- critical-intervention examples;
- local evidence occlusion maps.

**Use when:** the next architecture should be informed by the exact failure mechanism.

### Option C — Build the factorized Visual State Compiler

Implement deterministic and learned local evidence providers, then compare exact state assembly and safe rejection.

**Use when:** visual addressing remains strategically important.

### Option D — Train a task-specific row representation

Use supervised row identity, contrastive losses, intervention negatives, and abstention calibration.

**Risk:** this may repeat the row-retrieval abstraction error and overfit the renderer.

### Option E — Test patch-level evidence

Use local DINOv2 patch tokens or other spatial features to detect explicit factors.

**Use only as:** one provider inside a factorized architecture.

### Option F — Add temporal evidence

Use object tracks and state transitions for genuinely single-frame-ambiguous cases.

**Use after:** distinguishable single-frame cases are solved.

### Option G — Move to a fixed-camera bounded installation

Compare deterministic visual extraction, a pinned learned extractor, and direct instrumentation in a realistic but constrained environment.

**Use after:** the synthetic Phase 2 component benchmark establishes a credible evidence compiler.

---

## 19. Recommended sequence

### Stage 1 — Evidence closure

One bounded PR should:

- commit the full raw result;
- add the environment manifest;
- add the research result document;
- update the claims audit;
- fix the report's primary denominators;
- add confidence intervals;
- preserve per-family counts;
- record the Phase 1 stop decision.

### Stage 2 — Failure atlas

Produce a diagnostic artifact without changing the representation.

The purpose is to decide whether failures arise from:

- spatial position loss;
- local evidence absence;
- calibration;
- action aliasing;
- preprocessing;
- prototype choice;
- classifier geometry;
- OOD score overlap.

### Stage 3 — Phase 2 design and benchmark

Specify the factor schema, evidence contracts, rejection semantics, baselines, loss function, and kill conditions before implementation.

### Stage 4 — Implement the minimum factorized baseline

Begin with deterministic extraction plus direct instrumentation. Add learned components only where the deterministic provider demonstrably fails.

---

## 20. Phase 2 kill conditions

Stop or radically narrow the Visual State Compiler direction if:

1. direct instrumentation is materially cheaper, safer, and easier to govern;
2. exact factor extraction cannot tolerate modest held-out variation without unsafe acceptance;
3. factorized extraction does not materially improve exact row recovery over normalized pixels;
4. critical-evidence absence remains poorly calibrated;
5. the evidence and governance layer adds more complexity than a conventional detector plus structured log;
6. the synthetic fixture does not predict performance in a fixed-camera bounded scene;
7. the learned component requires near-exhaustive examples for every state without a compensating benefit.

---

## 21. Open research questions

### Problem definition

- Is exact policy-row recovery truly required in the intended product?
- Would an explicit many-observations-to-one-symbolic-state contract be sufficient?
- Is visual input strategically important, or is it primarily a demonstration?

### Benchmark validity

- Is family-held-out variation enough, or is renderer-held-out variation required?
- Should policy rows or state combinations also be held out?
- What real operating distribution should weight errors?
- What false-acceptance target is required before deployment can be discussed?

### Representation

- Does the CLS token discard policy-critical locality?
- Can patch tokens preserve object existence and position?
- Would a tiny task-specific model outperform a large semantic encoder?
- Can factor-level supervision prevent action-equivalent row aliasing?

### Calibration

- Should thresholds be per row, per factor, or globally risk-optimized?
- How many independent calibration examples are required per row?
- What does calibration transfer mean across environment changes?
- How should no-conflicting-action rows be represented?

### Governance

- Does the complete manifest and binding chain provide more audit value than a conventional model digest plus append-only structured log?
- Which identities are necessary for deployment?
- Which identities are research-only complexity?
- What should be signed or attested?

### Deployment

- What is the first realistic bounded environment?
- When is direct instrumentation the correct answer?
- Which single-frame ambiguities require temporal evidence?
- What safe fallback is required?

---

## 22. Evidence inventory

Current relevant repository files include:

```text
docs/research/visual-address-phase-one.md
docs/claims-audit.md

examples/arcade_visual_sign_reader.py
examples/arcade_visual_address_benchmark.py
examples/arcade_shooter_policy.py

zeromodel/visual.py
zeromodel/vision.py
zeromodel/visual_address.py
zeromodel/visual_address_manifest.py
zeromodel/visual_benchmark.py
zeromodel/visual_corruptions.py
zeromodel/visual_dataset.py
zeromodel/visual_encoder.py
zeromodel/visual_experiment.py
zeromodel/visual_precomputed.py
zeromodel/visual_retrieval.py
zeromodel/matrix_blob.py
zeromodel/deployment_binding.py

tests/test_visual_sign_reader.py
tests/test_visual_address.py
tests/test_visual_benchmark.py
tests/test_visual_retrieval.py
tests/test_arcade_visual_address_benchmark.py

.github/scripts/run_visual_address_smoke.py
.github/workflows/visual-address-benchmark.yml
```

Result records to add:

```text
docs/results/visual-address-phase-one-dinov2-full.md
docs/results/visual-address-phase-one-dinov2-full.json
docs/research/ZeroModel-visual-address-research-status.md
```

---

## 23. Bottom line

The project is no longer searching blindly for a visual model.

It has established:

```text
compiled bounded policy                     works
exact symbolic addressing                   works
canonical exact visual codeword addressing  works
provider-neutral visual governance          works as architecture
global DINOv2 row retrieval                 fails current continuation criteria
safe observation-to-symbol compilation      remains open
```

The scientifically responsible next move is:

> Close Phase 1 as a measured negative result, create a failure atlas, and test a factorized Visual State Compiler only after its hypothesis, baselines, safety target, and kill conditions are declared.

That is genuine research progress. The failed provider has clarified the problem that still needs to be solved.
