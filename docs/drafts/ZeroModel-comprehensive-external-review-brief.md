# ZeroModel Comprehensive External Review Brief

**Required repository path:** `docs/drafts/ZeroModel-comprehensive-external-review-brief.md`  
**Prepared:** 17 July 2026  
**Review status:** Request for independent critical review  
**Repository:** `ernanhughes/zeromodel`  
**Review ref:** **[INSERT THE EXACT COMMIT SHA REVIEWED]**  
**Primary area:** Governed visual observation addressing and the proposed next research direction  
**Required posture:** Adversarial, evidence-led, code-specific, and willing to recommend stopping the direction

---

# 1. Your role

You are being asked to act as an independent technical and research reviewer of ZeroModel.

Do not behave as a collaborator trying to preserve the authors' current direction.

Your task is to determine:

- what has actually been built;
- what has actually been validated;
- whether the research question is correctly formulated;
- whether the experiment supports the authors' interpretation;
- whether the architecture is unnecessarily complicated;
- whether important baselines or failure modes are missing;
- whether the proposed next direction follows from the evidence;
- whether the entire visual-address direction should be narrowed, replaced, or stopped.

The authors currently believe they have discovered a meaningful distinction between:

```text
semantic visual similarity
```

and:

```text
governed policy-address fidelity
```

They also currently believe the next promising direction is a factorized **Visual State Compiler**.

Those are hypotheses to review, not conclusions you are required to accept.

The most useful review may show that:

- the benchmark asks the wrong question;
- exact row identity is not the right requirement;
- the negative result is caused by calibration rather than representation;
- the learned path was tested unfairly;
- the fixture is too synthetic to support any broader inference;
- direct instrumentation makes the visual direction unnecessary;
- the governance machinery adds little beyond conventional logging;
- the factorized proposal is merely ordinary computer vision with additional names;
- another simpler research programme would be better;
- ZeroModel's strongest contribution lies elsewhere.

Be direct.

---

# 2. Review objectives

Your review should answer five high-level questions.

## 2.1 What is ZeroModel now?

Reconstruct the implemented system from the code rather than relying on the project's language.

State what is:

- a data structure;
- a deterministic algorithm;
- a policy artifact;
- a runtime consumer;
- a visual representation;
- a calibration artifact;
- a benchmark harness;
- a governance or lineage contract;
- a research hypothesis;
- a validated capability;
- a metaphor.

## 2.2 What did the full visual experiment actually establish?

Evaluate both the raw result and the interpretation.

Determine whether the evidence justifies saying:

- the DINOv2 path failed;
- normalized pixels are the stronger baseline;
- action accuracy is misleading;
- exact row recovery is the correct primary metric;
- rejection is the central problem;
- global semantic embeddings are mismatched to the task.

## 2.3 What is missing or wrong?

Search for:

- implementation defects;
- invalid assumptions;
- leakage;
- weak baselines;
- inappropriate metrics;
- underpowered comparisons;
- calibration errors;
- post-hoc reinterpretation;
- governance complexity without value;
- untested security properties;
- unsupported positioning or novelty claims.

## 2.4 What should happen next?

Do not default to the proposed Visual State Compiler.

Compare at least:

- stopping visual work;
- direct instrumentation;
- deterministic factor extraction;
- conventional object detection;
- task-specific classification;
- supervised contrastive row learning;
- patch-token evidence;
- temporal state estimation;
- a fixed-camera bounded installation;
- removing exact row identity from the requirement;
- focusing ZeroModel on policy artifacts rather than perception.

## 2.5 What is the smallest decisive next experiment?

Recommend one bounded experiment or PR that produces the most information.

It should have:

- a falsifiable hypothesis;
- mandatory baselines;
- a declared loss or target;
- raw artifacts;
- kill conditions;
- explicit non-goals.

---

# 3. Project in one sentence

The strongest current interpretation is:

> ZeroModel compiles bounded scored policies into deterministic, identity-bearing, inspectable artifacts and allows an exact runtime state—or a separately governed observation provider—to address a policy row and recover an action with traceable evidence.

This sentence is itself reviewable.

The project has used stronger public metaphors and ambitions, including visual intelligence, infinite memory, and images as a medium of intelligence. The current repository claims audit substantially narrows those claims.

A reviewer should distinguish the defensible finite-policy artifact contribution from broader programme language.

---

# 4. Current evidence ladder

## 4.1 Validated core

The repository currently contains evidence for:

- deterministic `ScoreTable`, `LayoutRecipe`, and `VPMArtifact` construction;
- deterministic artifact identity;
- cell-to-source mapping;
- multiple deterministic views over one score table;
- bounded state-addressed policy lookup;
- exact action selection without invoking a learned model at policy-decision time;
- exact policy trace and artifact identity;
- Lua export of a bounded policy;
- finite row-property checking;
- linked verification and repair lineage;
- canonical deterministic visual feature-codeword addressing in one bounded arcade fixture;
- provider-neutral visual-address contract mechanics;
- identity-bearing tensor, encoder, calibration, prototype, and deployment records;
- family-held-out visual benchmark infrastructure.

## 4.2 Implemented but not validated as a successful capability

The repository contains implementations of:

- normalized-pixel approximate addressing;
- pinned DINOv2-small CLS extraction;
- DINOv2 medoid retrieval;
- DINOv2 all-prototype k-NN;
- a ridge linear row probe;
- held-out corruption families;
- critical-intervention and OOD evaluation;
- per-family and per-observation traces.

The full run now measures those implementations. The learned systems did not satisfy the current continuation criteria.

## 4.3 Not established

The repository does not establish:

- open-world perception;
- learned critical-evidence detection;
- safe visual deployment;
- robustness across real cameras;
- robustness across renderers;
- temporal perception;
- a fixed-camera real-world result;
- a task-specific learned visual state compiler;
- improved human inspection;
- governance superiority over a lightweight conventional stack;
- safety certification;
- general visual reasoning.

---

# 5. Historical progression

The visual-address direction evolved through several stages.

## 5.1 Exact symbolic policy

A complete bounded arcade state is encoded as a policy row.

The policy contains:

- seven tank positions;
- seven target positions plus target absence;
- two cooldown states;
- four actions;
- 112 policy rows.

The exact runtime state selects a row, and the policy artifact selects an action.

## 5.2 Exact canonical visual sign reader

Every state is rendered as an integer frame:

```text
height: 16
width: 28
dtype: uint8
```

The frame visibly contains:

- tank position;
- target position or absence;
- cooldown state.

The canonical reader extracts a deterministic quantized feature codeword and addresses a separate visual index bound to the exact policy artifact.

It succeeds on all canonical rows and the committed exhaustive trajectory fixture.

It does not currently demonstrate accepted tolerance to the held-out benign perturbation families.

## 5.3 Provider-neutral visual addressing

The project created a seam through which multiple visual providers can return a governed decision:

```text
ImageObservation
    ↓
VisualAddressProvider
    ↓
VisualAddressDecision
    ↓
VisualPolicyReader / VPMPolicyLookup
    ↓
policy action
```

The address provider and policy artifact remain separately identified.

## 5.4 Learned Phase 1 benchmark

The benchmark compares:

- A: exact deterministic reader;
- B: normalized-pixel medoid retrieval;
- C: frozen DINOv2 medoid retrieval;
- D: frozen DINOv2 all-prototype k-NN;
- G: rejection-equipped ridge linear probe.

The declared Phase 1 research question was whether a pinned learned representation could improve held-out benign recovery while retaining safe rejection.

---

# 6. Current code architecture

Review the current repository rather than relying only on this summary.

## 6.1 Policy and original visual codebook

```text
examples/arcade_shooter_policy.py
    bounded state/action fixture
    row-ID encoding
    policy artifact compilation

examples/arcade_visual_sign_reader.py
    canonical integer renderer
    deterministic visual feature specification
    complete frame enumeration
    canonical visual index compilation
    visual-policy equivalence fixture

zeromodel/visual.py
    VisualFeatureSpec
    VisualIndexCalibration
    VisualIndexBuild
    VisualDecision
    extract_visual_features
    build_visual_index
    VisualSignReader
```

Questions to ask:

- Is the canonical renderer too deliberately aligned with the feature extractor?
- Is this a meaningful visual result or an encoded state protocol?
- Does that distinction matter if the claim is bounded and explicit?
- Is the separate visual index valuable compared with directly decoding the renderer?

## 6.2 Governed address contracts

```text
zeromodel/visual_address.py
zeromodel/visual_address_manifest.py
zeromodel/matrix_blob.py
zeromodel/deployment_binding.py
zeromodel/vision.py
```

Key concepts include:

- immutable observations;
- provider contracts;
- score semantics;
- observation and representation specification digests;
- address artifact identity;
- calibration identity;
- policy identity;
- source scope;
- replay contract;
- accepted and rejected decisions;
- prototype-to-policy bindings;
- deployment status.

Questions to ask:

- Which identities are essential?
- Which are research scaffolding?
- Does this materially outperform a model digest, config file, and append-only JSON event log?
- Is `exact_decision` replay realistic for float representations across environments?
- Is the deployment binding an actual security control or only a consistency check?

## 6.3 Learned representations

```text
zeromodel/visual_encoder.py
    EncoderManifest
    FrozenVisualEncoder
    HuggingFaceDinoV2Encoder

zeromodel/visual_precomputed.py
    precomputed representation providers

zeromodel/visual_retrieval.py
    NormalizedPixelEncoder
    VectorCalibration
    build_vector_address
    VectorAddressIndex
    FrozenVectorAddressProvider
    LinearProbeBuild
    LinearProbeIndex
    build_linear_probe
```

The current DINOv2 path uses:

```text
model:
facebook/dinov2-small

revision:
ed25f3a31f01632728cabb09d1542f84ab7b0056

representation:
L2-normalized CLS token

preprocessing:
identity-bearing square letterbox followed by the pinned processor
```

Questions to ask:

- Is a global CLS token a fair test of the proposed visual-sign task?
- Does resizing a 16×28 synthetic frame to the encoder's input destroy small local evidence?
- Would patch tokens, a small CNN, or explicit detectors be a more appropriate baseline?
- Does the DINOv2 result support a general conclusion about semantic representations, or only this preprocessing and pooling choice?

## 6.4 Dataset and benchmark

```text
zeromodel/visual_corruptions.py
zeromodel/visual_dataset.py
zeromodel/visual_experiment.py
zeromodel/visual_benchmark.py

examples/arcade_visual_address_benchmark.py
.github/scripts/run_visual_address_smoke.py
.github/workflows/visual-address-benchmark.yml
```

The evaluator:

- separates expected accept, expected reject, and impossibility-control cases;
- only counts row and action correctness on expected-accept observations;
- counts accepted conflicting actions;
- reports false accepts over distinguishable rejection opportunities;
- reports false rejects over benign opportunities;
- retains per-family counts;
- can retain per-observation traces.

Questions to ask:

- Are the split families genuinely independent?
- Is family-held-out testing enough when every frame comes from the same renderer and policy surface?
- Is there seed leakage or parameter overlap?
- Is three variants per family enough?
- Should renderer, state combinations, or policy rows be held out?
- Are the OOD fixtures representative?
- Is the critical intervention set sufficiently broad?
- Is target removal correctly excluded as an impossibility control?
- Are there other hidden impossibility cases?

---

# 7. Important implementation details to inspect

## 7.1 Calibration

`VectorCalibration` currently stores:

- per-row acceptance thresholds;
- per-row ambiguity margins;
- one global `calibration_count`;
- a calibration quantile;
- metadata and digest.

The current default quantile is `0.0`, the minimum observed calibration value.

For retrieval, calibration computes:

```text
correct score
    minus
best conflicting-action score
```

For a row with no conflicting-action candidate, calibration uses a synthetic conflict score of `-1.0`.

At runtime, if no conflicting-action candidate exists, the code falls back to the second-ranked prototype even if it has the same action.

This appears to create a calibration/runtime semantic mismatch.

Review whether:

- no-conflict rows exist in this fixture;
- the mismatch affected the result;
- ambiguity should be optional when no conflicting action exists;
- calibration should store per-row sample counts;
- quantile `0.0` is too permissive or too brittle;
- threshold and margin should be selected from an explicit loss curve.

## 7.2 Benchmark metric schema

`VisualBenchmarkMetrics.action_accuracy` and `.row_accuracy` divide by all scored observations:

```text
benign expected accepts
+
distinguishable expected rejects
```

The evaluator separately places `benign_action_accuracy` and `benign_row_accuracy` in result notes.

Review whether:

- the primary schema is misleading;
- the benign metrics should be first-class;
- correct rejection should have its own score rather than sharing a denominator;
- accepted-decision precision should be first-class;
- risk-weighted loss is required;
- a predeclared FAR target is mandatory;
- confidence intervals and paired tests should be included.

## 7.3 Float representation identity

The representation digest hashes the exact big-endian float32 vector plus the representation specification digest.

This identifies the exact vector produced in one execution environment.

Review whether the project incorrectly implies that it is reproducible across:

- CPU and GPU;
- BLAS implementations;
- PyTorch versions;
- compiler settings;
- processor implementations;
- hardware architectures.

Consider whether:

- quantized vectors;
- tolerance-aware identity;
- environment-scoped identity;
- golden vectors;
- or a separately identified consumer plan

would be more honest.

## 7.4 Summary serialization defect

The first CI smoke workflow completed the benchmark and wrote its report, but the final console-summary serialization failed because a nested immutable mapping remained a `mappingproxy`.

The full local run shown to the authors produced a console summary successfully, but reviewers should verify:

- which script and commit produced it;
- whether the CI summary defect remains on the reviewed ref;
- whether serialization tests cover nested frozen metadata;
- whether generated reports are stable and round-trippable.

This is not the main research result, but it is evidence that the reporting path requires stronger end-to-end tests.

---

# 8. Benchmark dataset

The full run uses `variants_per_family = 3`.

| Partition | Observations |
|---|---:|
| Prototype | 1,344 |
| Calibration | 1,344 |
| Held-out benign test | 1,344 |
| Information-theoretic controls | 98 |
| Distinguishable critical interventions | 224 |
| OOD | 24 |
| **Total** | **4,378** |
| **Scored** | **1,592** |

## 8.1 Prototype families

- clean intensity variation;
- lower brightness;
- one-pixel upward translation;
- palette A.

## 8.2 Calibration families

- held-out contrast;
- one-pixel downward translation;
- palette B;
- low-amplitude noise.

## 8.3 Benign test families

- stronger unseen brightness;
- two-pixel vertical translation;
- palette C;
- noncritical structured background patch.

## 8.4 Distinguishable rejection cases

- tank removed;
- cooldown indicator removed;
- blank frame;
- checkerboard;
- impossible two-tank frame.

## 8.5 Information-theoretic control

Removing a visible target can create a frame identical to a valid no-target state.

Those 98 cases are reported but excluded from FAR and FRR denominators.

Review this treatment carefully. The authors believe it is correct because the pixels contain no information capable of distinguishing the two hidden states.

---

# 9. Full result

Run identity:

```text
dataset_digest:
91b1b422482eeeef20eb182162eb2a745f9b50524cc7f94ec95a0aba5f2fa37e

report_digest:
d7d0b4db13c9f96b2ac4583aae25fa2159fb62471fb90110103548317f084035

observations:
4,378

scored observations:
1,592

benign opportunities:
1,344

distinguishable rejection opportunities:
248

validation_status:
research
```

## 9.1 Raw console metrics

| System | Whole-evaluation action accuracy | Whole-evaluation row accuracy | FAR | FRR |
|---|---:|---:|---:|---:|
| **A** | 0.00% | 0.00% | 0.00% | 100.00% |
| **B** | 62.00% | 52.76% | 51.21% | 25.00% |
| **C** | 59.11% | 25.94% | 82.26% | 3.27% |
| **D** | 63.13% | 25.63% | 73.79% | 3.50% |
| **G** | 50.19% | 24.31% | 100.00% | 1.79% |

## 9.2 Derived benign metrics

| System | Benign action accuracy | Exact benign row accuracy | Conflicting-action errors |
|---|---:|---:|---:|
| **A** | 0.00% | 0.00% | 0 |
| **B** | 73.44% | 62.50% | 21 |
| **C** | 70.01% | 30.73% | 359 |
| **D** | 74.78% | 30.36% | 292 |
| **G** | 59.45% | 28.79% | 521 |

## 9.3 Accepted benign reliability

| System | Accepted benign | Correct action when accepted | Exact row when accepted |
|---|---:|---:|---:|
| **B** | 1,008 | 97.92% | 83.33% |
| **C** | 1,300 | 72.38% | 31.77% |
| **D** | 1,297 | 77.49% | 31.46% |
| **G** | 1,320 | 60.53% | 29.32% |

## 9.4 Action-equivalent wrong-row outcomes

| System | Correct actions | Exact rows | Correct action from wrong row |
|---|---:|---:|---:|
| **B** | 987 | 840 | 147 |
| **C** | 941 | 413 | 528 |
| **D** | 1,005 | 408 | 597 |
| **G** | 799 | 387 | 412 |

## 9.5 Rejection outcomes

| System | False accepts | Opportunities | FAR |
|---|---:|---:|---:|
| **A** | 0 | 248 | 0.00% |
| **B** | 127 | 248 | 51.21% |
| **C** | 204 | 248 | 82.26% |
| **D** | 183 | 248 | 73.79% |
| **G** | 248 | 248 | 100.00% |

---

# 10. Authors' current interpretation

The authors currently interpret the result as follows.

## 10.1 System A

A validates exact canonical addressing but not perturbation tolerance.

## 10.2 System B

B is the strongest overall addressing baseline:

- best exact-row recovery;
- fewest conflicting-action errors among approximate systems;
- most reliable accepted actions;
- lowest approximate-system FAR.

It is still unsafe at a 51.21% FAR.

## 10.3 System C

C beats G but loses to B on the relevant fidelity and rejection trade-off.

## 10.4 System D

D has the highest benign action point estimate, exceeding B by 1.34 points.

The authors do not regard this as a meaningful win because D:

- loses 32.14 points of exact-row accuracy;
- adds 22.58 points of FAR;
- reduces accepted-action correctness by 20.43 points;
- produces 271 more conflicting-action errors;
- retains every prototype.

## 10.5 System G

G is eliminated because it combines weak benign recovery with a 100% FAR.

## 10.6 Research interpretation

The authors believe:

1. action accuracy can hide wrong-address behaviour;
2. global semantic invariance is poorly aligned with exact governed-state identity;
3. similarity alone is insufficient for rejection;
4. the next problem is visual evidence compilation rather than whole-row retrieval.

Review all four claims critically.

---

# 11. Reasons the current interpretation may be wrong

A useful external review should actively test these alternatives.

## 11.1 Exact row identity may be an unnecessarily strict target

If several rows are action-equivalent and no downstream consumer uses row-specific evidence, exact row recovery may not matter operationally.

Questions:

- Is exact row identity a product requirement or an artifact-driven preference?
- Does row identity improve safety, replay, or inspection enough to justify the stricter task?
- Could the correct abstraction be an action-equivalence class?
- Could multiple observations legitimately address one symbolic state or action?
- Would a hierarchical address—factor state, action class, exact row—be more appropriate?

## 11.2 The comparison may unfairly favour normalized pixels

The fixture is tiny, geometric, and directly rendered.

Normalized pixels preserve exact geometry and may be almost optimal for this synthetic task.

Questions:

- Is this a meaningful test of DINOv2?
- Does the result say anything beyond “semantic encoders are poor on tiny synthetic sprites”?
- Should the image be rendered at a native visual scale rather than enlarged from 16×28?
- Should a conventional small CNN, HOG, template matcher, or object detector be the mandatory learned baseline?

## 11.3 The learned representation choice may be inappropriate

Only the global CLS token was tested.

Questions:

- Would patch tokens preserve local object existence and coordinates?
- Would intermediate layers work better?
- Would DINOv2 register tokens or pooled spatial bins help?
- Would a task-specific projection be a fairer test?
- Was the model's preprocessing designed for natural images, not integer arcade glyphs?

## 11.4 The calibration may be the main failure

The experiment reports one calibrated operating point.

Questions:

- What do complete FAR/FRR curves show?
- Is quantile `0.0` defensible?
- How sensitive are results to threshold and margin selection?
- Can any operating point achieve acceptable FAR while preserving useful row accuracy?
- Were per-row calibration sample counts sufficient?
- Is the conflicting-action margin the right rejection statistic?
- Should OOD detection be separate from row discrimination?

## 11.5 The safety language may be under-specified

The authors describe FAR values as unsafe, but no deployment-specific acceptable FAR was declared.

Questions:

- What loss function should govern the trade?
- Are tank removal, cooldown removal, blank, checkerboard, and two-tank cases equally costly?
- Should false accepts be severity-weighted?
- Is 51% clearly unacceptable for research but not yet formally “unsafe” without a use case?
- Should the document use “fails the benchmark's rejection expectation” rather than “unsafe”?

## 11.6 The benchmark may be too small or too synthetic

Questions:

- Do three variants per family support stable conclusions?
- Do the 1,344 benign observations provide independent information when they derive from 112 states and deterministic transforms?
- Should uncertainty account for clustering by state and family?
- Are Wilson intervals over observations misleading because samples are correlated?
- Should inference use state-level or family-level bootstrap resampling?
- Is a paired comparison required?

## 11.7 The benchmark may not test generalization in the relevant sense

All states and all rows appear in prototype and calibration splits.

Only corruption families are held out.

Questions:

- Is this invariance testing rather than state generalization?
- Should some state combinations be held out?
- Would held-out rows make sense if the goal is exact finite policy addressing?
- Should a second renderer or camera style be held out?
- Should the benchmark include renderer drift, compression, scale, blur, crop, and sensor noise?

## 11.8 The governance layer may be overbuilt

Questions:

- What practical incident question can the current contracts answer that a conventional model registry plus structured event log cannot?
- How much code and review burden does the identity chain add?
- Are all digests persisted and independently verifiable?
- Is deployment binding enforceable outside one process?
- Does a content digest provide authenticity?
- Is the distinction between artifact identity and consumer-plan identity consistently applied?

## 11.9 The proposed factorized compiler may be ordinary perception renamed

Questions:

- Is `Visual State Compiler` a useful architectural distinction or merely object detection plus state assembly?
- What is genuinely ZeroModel-specific?
- Does the name obscure established prior art?
- Would a conventional detector feeding a typed state schema accomplish the same result?
- Is the contribution the evidence contract rather than the perception algorithm?

## 11.10 Direct instrumentation may dominate the entire direction

The arcade engine already has exact state.

In many bounded installations, explicit sensors or software signals may be available.

Questions:

- Why use vision at all?
- Is the visual path only valuable when instrumentation is unavailable?
- Is visual redundancy useful as an independent verifier?
- What deployment makes observation addressing preferable to direct telemetry?
- Should the next benchmark compare against direct instrumentation cost, reliability, and governance?

---

# 12. Known evidence and engineering gaps

Reviewers should verify and expand this list.

## 12.1 Result preservation

- The full original JSON report is not yet included in this brief.
- The full local environment manifest is incomplete.
- The exact command and Git commit must be recorded.
- The exact model cache state and package versions must be recorded.
- Raw per-family results should be committed.
- Per-observation traces should be retained for paired analysis.

## 12.2 Statistical analysis

- no paired B-versus-D significance test;
- no state-clustered or family-clustered uncertainty;
- no operating curves;
- no predeclared target FAR;
- no risk-weighted objective;
- no multiple-comparison discussion;
- no calibration sensitivity analysis.

## 12.3 Baselines

Missing or not yet reported:

- direct symbolic instrumentation;
- deterministic factor extraction;
- perceptual hash;
- HOG or other classical local descriptor;
- small task-specific CNN;
- multi-head factor classifier;
- supervised contrastive row encoder;
- patch-token retrieval;
- explicit OOD detector;
- one-class or conformal rejection;
- standard k-NN/classifier implementation from a conventional library;
- lightweight model registry plus JSON-log governance baseline.

## 12.4 Dataset

- one synthetic renderer;
- one policy fixture;
- one source resolution;
- limited corruption families;
- three variants per family;
- no temporal sequence benchmark;
- no real camera;
- no held-out renderer;
- no environment drift;
- no state-prior weighting;
- no label-noise test;
- no adversarial near-codeword generation.

## 12.5 Code and contracts

- benchmark's primary action and row accuracy denominators are easy to misread;
- no-conflict calibration/runtime mismatch may exist;
- global calibration count hides per-row counts;
- float-vector digest reproducibility is environment-scoped;
- report serialization previously failed on nested immutable metadata;
- report and manifest persistence may be fragmented across several file types;
- `deployment_status="research"` blocks claims but does not itself enforce safe deployment;
- no signatures or attestations;
- no cross-process persistence service;
- no declared memory/runtime benchmark for learned providers.

## 12.6 Product and research purpose

- no named external deployment;
- no declared acceptable error cost;
- no user study demonstrating value of exact row provenance;
- no comparison of visual addressing with direct sensors;
- unclear whether visual addressing is central to ZeroModel or a research branch.

---

# 13. Proposed next direction under review

The authors currently propose replacing whole-image row retrieval with a factorized Visual State Compiler.

Proposed flow:

```text
ImageObservation
    ↓
local evidence providers
    ↓
EvidenceBundle
    ↓
TypedObservedState
    ↓
exact state encoding
    ↓
VPMPolicyLookup
    ↓
action + trace
```

Possible fields for the arcade fixture:

```text
tank_present
tank_count
tank_position
target_present
target_position
cooldown_state
frame_validity
```

Possible field record:

```text
field_id
value
status
score
calibration_id
provider_id
evidence_region
rejection_reason
```

Possible provider baselines:

- direct engine state;
- deterministic connected components and geometry;
- local templates;
- patch-token probes;
- small factor classifiers;
- compact multi-head model.

The proposed advantage is that missing evidence and invalid combinations become explicit rather than being inferred from one global similarity score.

Review whether this direction is:

- logically implied by the result;
- merely one plausible option;
- unnecessarily elaborate;
- a standard perception stack with useful governance;
- likely to overfit the fixture;
- worth pursuing within ZeroModel.

---

# 14. Questions for external reviewers

## 14.1 Problem definition

1. What is the actual problem the code solves today?
2. Is “observation-addressed policy” a coherent and useful category?
3. Is exact policy-row identity a necessary requirement?
4. Should the target be exact state, action-equivalence class, or action?
5. When is visual addressing preferable to direct telemetry?
6. Is the arcade fixture a valid hydrogen-atom environment or an engineered encoding trick?
7. Does the project need a visual component at all?

## 14.2 Experimental validity

8. Is the family-held-out split free of material leakage?
9. Are prototype, calibration, and test families sufficiently distinct?
10. Is three variants per family enough?
11. Are observations independent enough for the reported intervals?
12. Should uncertainty be computed over states, families, or individual frames?
13. Should the benchmark hold out renderers or state combinations?
14. Are the critical interventions valid expected-reject cases?
15. Is target removal correctly treated as information-theoretically unscorable?
16. What other impossibility controls are missing?
17. Are blank, checkerboard, and two-tank frames meaningful OOD tests?
18. Does preprocessing create an avoidable information bottleneck?
19. Is the source resolution too small for the learned encoder?

## 14.3 Metrics and decision theory

20. Which metric should be primary?
21. Is exact-row accuracy more important than action accuracy?
22. Should correct action from a wrong row count as partial success?
23. Should acceptance precision be first-class?
24. What false-acceptance target should be declared?
25. What deployment loss function is appropriate?
26. Should critical interventions be severity-weighted?
27. Are whole-evaluation action and row accuracy fields misleading?
28. What paired statistical analysis is required?
29. What operating curves are required?
30. How should calibration uncertainty be represented?

## 14.4 Representation

31. Is DINOv2 CLS an appropriate baseline?
32. What representation would be the strongest conventional baseline?
33. Should patch tokens be tested before abandoning the encoder?
34. Would a tiny supervised model be more informative?
35. Would explicit factor supervision solve the wrong-row/right-action problem?
36. Is global semantic invariance genuinely opposed to this task?
37. Does the result generalize beyond tiny synthetic sprites?
38. What result would falsify the authors' semantic-invariance interpretation?

## 14.5 Calibration and rejection

39. Is empirical lower quantile `0.0` defensible?
40. Should thresholds be per row, per action, or global?
41. Should ambiguity compare against conflicting actions or all other rows?
42. How should rows with no conflicting action candidate behave?
43. Should OOD detection be a separate component?
44. Would conformal prediction help?
45. Are similarity and margin sufficient rejection statistics?
46. Can any threshold curve salvage B, C, or D?
47. What calibration data volume is required?
48. How should environment drift trigger recalibration?

## 14.6 Architecture and governance

49. Is the policy/index/encoder/calibration/manifest/binding separation correct?
50. Which identifiers are redundant?
51. Does the architecture provide audit value proportional to its complexity?
52. What is the minimal conventional governance baseline?
53. Does exact-decision replay make sense for floating-point providers?
54. Should representation identity be quantized?
55. Should accepted and rejected decisions be different types?
56. Should evidence regions be mandatory for learned acceptance?
57. Is the deployment binding a useful contract without signatures?
58. What persistence and attestation layer is missing?
59. Does the provider seam remain valuable even if visual research stops?

## 14.7 Proposed Visual State Compiler

60. Is factorization the correct hypothesis revision?
61. Which factors should be explicit?
62. Should state assembly be deterministic?
63. Is the proposal simply object detection plus a typed schema?
64. What is the smallest implementation that could falsify it?
65. Which baseline should be implemented first?
66. Should direct instrumentation be considered the gold standard?
67. Should DINOv2 patch tokens be included?
68. What evidence-region metric is required?
69. How should single-frame impossibility be represented?
70. When should temporal state be introduced?
71. What kill condition should stop the factorized direction?

## 14.8 Novelty and positioning

72. What prior work best describes the mechanism?
73. Is the visual matcher technically novel?
74. Is the artifact governance layer the actual contribution?
75. Is “Visual State Compiler” useful terminology or unnecessary renaming?
76. What is the smallest publishable claim?
77. Does the visual branch strengthen or distract from ZeroModel?
78. Which public claims should be removed immediately?
79. What would make the work interesting to a technical reviewer?
80. What result would justify a paper rather than a repository note?

## 14.9 Next experiment

81. What is the single highest-information next experiment?
82. What files and APIs should it change?
83. What baselines are mandatory?
84. What target and loss must be declared?
85. What result continues the direction?
86. What result stops it?
87. Should the next step be analysis only rather than new code?
88. Should the result be reproduced on CI before any new architecture work?

---

# 15. Required reviewer output

Use the structure below.

## A. Overall verdict

Choose one primary verdict:

```text
strong and well-founded direction
promising but incorrectly framed
valid bounded research result
useful infrastructure, weak research thesis
overbuilt conventional mechanism
benchmark is not yet trustworthy
visual direction should be paused
visual direction should be stopped
```

Explain the verdict in no more than eight paragraphs.

## B. Reconstructed system

Describe what the repository actually implements in plain technical language.

Separate:

- policy artifact;
- observation provider;
- representation;
- calibration;
- runtime decision;
- trace;
- benchmark;
- deployment claim.

## C. Most damaging critique

State the single criticism that, if correct, most changes the project's direction.

Include:

```text
Claim or assumption:
Failure mode:
Evidence:
Why it matters:
How to test it:
Decision if confirmed:
```

## D. Severity-ranked findings

For every finding use:

```text
Severity: blocker | major | moderate | minor | optional
Area:
Finding:
Repository evidence:
Why it matters:
Smallest corrective change:
Required test:
Claims impact:
```

Do not dilute serious findings with a long list of stylistic suggestions.

## E. Evaluation of the full result

Answer explicitly:

- Did A fail or define its expected boundary?
- Is B the strongest practical baseline?
- Did C fail?
- Did D meaningfully beat B?
- Is G eliminated?
- Is exact-row accuracy the right measure?
- Are the FAR values sufficient to stop the current path?
- Is the current interpretation post hoc?

## F. Alternative interpretations

Provide at least three plausible interpretations of the result.

Rank them by likelihood.

For each, state what additional evidence would distinguish it from the authors' interpretation.

## G. Missing baselines

List the mandatory baselines that should have existed before or must exist next.

Separate:

- algorithm baseline;
- perception baseline;
- calibration baseline;
- governance baseline;
- deployment baseline.

## H. Recommended next experiment

Specify one bounded experiment.

Include:

```text
Hypothesis:
Why this experiment:
Systems:
Dataset:
Splits:
Metrics:
Predeclared target:
Raw artifacts:
Statistical analysis:
Kill conditions:
Explicit non-goals:
Expected implementation files:
```

## I. Recommended next PR

Describe one PR only.

Include:

- branch purpose;
- files to add;
- files to change;
- tests;
- report artifacts;
- claims-audit changes;
- what must not be included.

## J. Claim rewrite

Provide:

1. the strongest claim the current repository can honestly make;
2. the strongest visual-address claim;
3. claims that should be removed;
4. claims that may be tested next.

## K. Research opportunity

State whether the result contains a genuinely interesting research observation.

Possible answers include:

- action-equivalent wrong-address behaviour is publishable;
- semantic invariance versus governed identity is interesting;
- rejection is the main contribution;
- this is established nearest-neighbour/OOD behaviour;
- the result is too fixture-specific;
- the artifact-governance layer is the real novelty;
- no substantial novelty remains.

Explain the reasoning and cite relevant prior-art categories.

---

# 16. Information available on request

Do not invent missing details.

The reviewer may request any of the following before finalizing the review:

- exact Git commit SHA;
- original full benchmark JSON;
- compact derived summary JSON;
- complete environment manifest;
- exact local command;
- Python, NumPy, PyTorch, Transformers, and platform versions;
- encoder manifest;
- loaded-weights digest;
- preprocessing digest;
- per-family counts;
- complete per-observation traces;
- B-versus-D paired outcome table;
- nearest-neighbour examples;
- confusion matrix;
- calibration thresholds per row;
- score and margin distributions;
- workflow logs;
- relevant pull-request patches;
- test files;
- claims-audit diff;
- prior visual-sign-reader design brief;
- public blog text;
- intended deployment scenarios.

When a missing item materially affects the verdict, say:

```text
Review blocked pending:
Why it is required:
What decision it could change:
```

Do not fill gaps with confident assumptions.

---

# 17. Suggested repository reading order

1. `docs/claims-audit.md`
2. `docs/research/visual-address-phase-one.md`
3. `docs/results/visual-address-phase-one-dinov2-full.md`
4. `examples/arcade_shooter_policy.py`
5. `examples/arcade_visual_sign_reader.py`
6. `zeromodel/visual.py`
7. `zeromodel/visual_address.py`
8. `zeromodel/visual_address_manifest.py`
9. `zeromodel/matrix_blob.py`
10. `zeromodel/visual_encoder.py`
11. `zeromodel/visual_retrieval.py`
12. `zeromodel/visual_experiment.py`
13. `zeromodel/visual_benchmark.py`
14. `examples/arcade_visual_address_benchmark.py`
15. `tests/test_visual_sign_reader.py`
16. `tests/test_visual_retrieval.py`
17. `tests/test_arcade_visual_address_benchmark.py`
18. `.github/scripts/run_visual_address_smoke.py`
19. `.github/workflows/visual-address-benchmark.yml`

Review implementation and tests before accepting the prose interpretation.

---

# 18. Pre-review facts versus hypotheses

## Facts currently evidenced

- the bounded policy has 112 rows and four actions;
- canonical deterministic visual addressing succeeds on the committed exact fixture;
- the held-out benchmark machinery executes;
- the full local DINOv2 run produced the recorded counts;
- B has substantially higher exact-row accuracy than C, D, and G;
- D has a 1.34-point higher benign action point estimate than B;
- B, C, D, and G have high FAR at the measured operating point;
- G accepts every distinguishable rejection opportunity;
- hundreds of learned-system correct actions come from wrong rows.

## Current interpretations

- exact row identity should be primary;
- global semantic embeddings are structurally mismatched;
- rejection is the central problem;
- D does not meaningfully beat B;
- the learned Phase 1 path should stop;
- factorized visual state compilation is the best next direction.

## Open hypotheses

- patch-level learned evidence can preserve local facts;
- deterministic factor extraction will outperform global retrieval;
- a task-specific representation can recover exact states safely;
- factorized evidence improves governance;
- a fixed-camera real-world environment will retain the observed pattern;
- direct instrumentation may dominate vision.

Keep these categories separate in the review.

---

# 19. Decision standard

The review should not ask whether the work is interesting in isolation.

It should ask whether the next unit of engineering effort is justified.

A continuation recommendation should require:

- a precise unresolved hypothesis;
- evidence that the next experiment distinguishes competing explanations;
- a conventional baseline;
- an operationally meaningful target;
- a credible rejection method;
- bounded implementation cost;
- an explicit stop condition.

A stop or pause recommendation is valid and useful.

---

# 20. Final instruction to the reviewer

Assume the authors are capable of implementing almost any proposed extension.

Therefore do not recommend complexity merely because it is technically possible.

Prefer the answer that most rapidly reveals whether ZeroModel has:

- a distinct research contribution;
- a useful bounded engineering pattern;
- a governance layer worth retaining;
- or an attractive metaphor wrapped around conventional mechanisms.

The desired outcome is not encouragement.

The desired outcome is a more accurate map of reality.
