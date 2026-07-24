# ZeroModel Perception Runtime — Comprehensive Design

**Status:** Proposed architecture  
**Package:** `zeromodel-perception`  
**Namespace:** `zeromodel.perception`  
**Target branch:** `main`  
**Design branch:** `design/perception-runtime`  

## 1. Purpose

The Perception runtime turns arbitrary recorded visual interactions into inspectable, deterministic, provenance-ready visual policy artifacts.

Its minimum input is a sequence of paired observations and actions:

```text
image_0 -> action_0
image_1 -> action_1
image_2 -> action_2
...
```

When the next visual state is available, the preferred record is:

```text
image_t + action_t -> image_t+1
```

The runtime must support four connected capabilities:

1. **Representation** — encode arbitrary source images and arbitrary action/result values as deterministic PNG/VPM artifacts.
2. **Discovery** — determine which spatially addressable source fields contribute to action or result fields.
3. **Conformance testing** — compare observed evidence with optional human-supplied expectations and preserve mismatches as first-class findings.
4. **Inference** — predict the most likely action/result for a previously unseen image, with confidence, nearest evidence, alternatives, and rejection.

Space Invaders is only a controlled validation domain. No production Perception API may contain domain concepts such as tank, alien, bullet, player, enemy, shoot, road, car, tree, crowd, or person.

## 2. Central principle

A Visual Policy Map is an addressable PNG representation. The source and target sides of the learning problem must both be constructible from arbitrary data.

```text
arbitrary source image -> source VPM PNG
arbitrary action/result -> target VPM PNG
```

The runtime learns or computes a mapping:

```text
source VPM -> target VPM
```

and, where required, a transition mapping:

```text
source VPM + action VPM -> next-state VPM
```

The key epistemic rule is:

> Information believed to determine a result should leave a measurable spatial signature in the representation of that result.

Agreement is explanatory evidence. Disagreement is not merely failure; it is a discovery signal that may indicate an incorrect model, incorrect human expectation, damaged representation, stale source data, target misalignment, hidden state, unexpected visual evidence, or an incomplete task specification.

## 3. Relationship to existing packages

The current repository keeps the core artifact kernel conservative and places new behavior in consumer packages. Perception must follow that rule.

### 3.1 `zeromodel.core`

Owns deterministic VPM artifact primitives, layout, score tables, content identity, and bounded policy lookup.

Perception may consume core artifacts but must not widen core contracts for learned behavior.

### 3.2 `zeromodel.observation`

Owns generic observation DTOs and visual address contracts.

Perception should reuse observation-owned image and address abstractions where they are semantically sufficient. New DTOs that describe learned mappings, datasets, evidence, predictions, or training outputs belong to Perception unless they are later proven to be general observation contracts.

### 3.3 `zeromodel.vision`

Owns deterministic, bounded, closed-world visual addressing and codebook-based reading.

Perception must not be added to `vision`. The existing vision package explicitly does not claim general computer vision, semantic perception, learned embeddings, or approximate learned providers. Perception is the separate package for those concerns.

Perception may optionally use `zeromodel.vision` as one provider family, but it must not require it for raw PNG-to-VPM construction.

### 3.4 `zeromodel.video`

Owns temporal video policy and the current video action-set domain.

Video may supply recorded interaction streams to Perception. Perception must remain usable without video and without any arcade domain.

A later integration adapter may map video action-set observation DTOs into generic Perception interaction DTOs. The Perception package must not import the video domain unless an explicit dependency is approved. The preferred initial direction is for video examples or integration code to depend on Perception, not for Perception to depend on video.

### 3.5 `zeromodel.artifacts`

Owns content-addressed artifact storage and reference identity.

Perception should expose immutable DTOs that can reference source PNGs, target PNGs, evidence PNGs, model artifacts, reports, and manifests. Durable storage integration should use artifact references rather than embedding large binary values in application DTOs.

### 3.6 `zeromodel.persistence.sqlalchemy`

Persistence must remain explicit and DTO-only across Store boundaries. Perception should first define protocols and an in-memory implementation. SQL persistence should be added later as a separate vertical slice only when the aggregate boundaries are stable.

### 3.7 `zeromodel.trust`

Trust may later sign promoted perception models, manifests, evidence reports, or inference receipts. Perception must not depend on trust in its first slice.

## 4. Proposed package boundary

Add a tenth publishable package:

```toml
[packages.perception]
distribution = "zeromodel-perception"
namespace = "zeromodel.perception"
source_root = "packages/perception/src"
depends_on = ["core", "observation"]
publishable = true
owned_prefixes = ["zeromodel.perception"]
```

Recommended initial dependencies:

```text
numpy>=1.23
Pillow>=10
zeromodel==<repo-version>
zeromodel-observation==<repo-version>
```

`Pillow` is justified because this package explicitly owns generic PNG reading, writing, normalization, and deterministic image materialization. Heavy frameworks such as Torch, TensorFlow, Transformers, scikit-learn, OpenCV, or remote model clients must not be production dependencies in the first slice.

The initial implementation should use NumPy-based algorithms and explicit provider protocols. Optional learned backends can be separate extras or later provider packages.

## 5. Scope

### 5.1 In scope

- Arbitrary PNG or image-array observations.
- Arbitrary discrete actions encoded as deterministic target PNGs.
- Parameterized actions encoded as deterministic multi-field target PNGs.
- Optional before/action/after transitions.
- Deterministic source and target visual encoding.
- Dataset validation and temporal alignment.
- Whole-image and field-based similarity.
- Evidence-weighted nearest-neighbor inference.
- Sparse or simple NumPy translators.
- Source-to-target field dependency maps.
- Keep-only, remove-only, replacement, and counterfactual tests.
- Optional component, region, aggregate, and relation annotations.
- Expected-evidence contracts.
- Unexpected-evidence discovery and clustering.
- Confidence, alternatives, rejection, and out-of-distribution findings.
- Immutable manifests and reproducible model identity.
- In-memory stores and DTO-only protocols.
- Research adapters for later algorithms.

### 5.2 Explicitly out of scope for the first production slice

- General object recognition.
- Automatic semantic naming of image regions.
- Claims of causal proof from correlation alone.
- Large neural models.
- GPU requirements.
- Remote foundation-model calls.
- End-to-end game playing.
- Reinforcement learning.
- Reward optimization.
- Environment-specific game concepts.
- Unbounded online learning.
- Database persistence before aggregate stability.
- Silent repair of corrupted or mismatched observations.
- A guarantee that a single image contains all state required for an action.

## 6. Core data model

### 6.1 Image identity

Every materialized image must have a deterministic identity derived from exact normalized pixel content and encoding metadata.

```python
@dataclass(frozen=True)
class PerceptionImageDTO:
    image_id: str
    width: int
    height: int
    channels: int
    dtype: str
    color_space: str
    content_digest: str
    png_artifact_ref: str | None
    inline_array: tuple[int, ...] | None
```

Large images should be referenced through artifact storage. Inline values are acceptable only for bounded tests and tiny fixtures.

### 6.2 Action identity

Actions must be domain-neutral.

```python
@dataclass(frozen=True)
class ActionValueDTO:
    action_id: str
    action_type: str
    discrete_label: str | None
    parameters: tuple[ActionParameterDTO, ...]
```

Examples:

```text
keyboard key: LEFT
controller axis: x=-0.7
mouse: dx=12, dy=-3, button=1
composite: MOVE_RIGHT + FIRE
```

The package must not assume one-hot discrete labels, although the first slice may support them first.

### 6.3 Interaction observation

```python
@dataclass(frozen=True)
class RecordedInteractionDTO:
    interaction_id: str
    sequence_id: str
    step_index: int
    observed_at_ns: int | None
    source_image_id: str
    action: ActionValueDTO
    next_image_id: str | None
    source_provenance: ProvenanceDescriptorDTO
```

The pairing is authoritative. The runtime must never infer that adjacent files are temporally aligned merely from filenames.

### 6.4 Source VPM

```python
@dataclass(frozen=True)
class SourceVPMDTO:
    source_vpm_id: str
    source_image_id: str
    encoder_spec_id: str
    width: int
    height: int
    channels: int
    content_digest: str
    png_artifact_ref: str
```

The first encoder should preserve the image almost as-is, subject only to explicit deterministic normalization.

### 6.5 Target VPM

```python
@dataclass(frozen=True)
class TargetVPMDTO:
    target_vpm_id: str
    action_id: str
    encoder_spec_id: str
    field_schema_id: str
    width: int
    height: int
    channels: int
    content_digest: str
    png_artifact_ref: str
```

The target VPM is a PNG, not an implicit Python class label. A decoder converts the predicted target image back into action candidates.

### 6.6 Transition target

```python
@dataclass(frozen=True)
class TransitionTargetDTO:
    transition_target_id: str
    source_vpm_id: str
    action_vpm_id: str
    next_state_vpm_id: str
```

Policy prediction and transition prediction are separate tasks and should not be conflated.

### 6.7 Field address

```python
@dataclass(frozen=True)
class VPMFieldAddressDTO:
    vpm_id: str
    channel: int
    x0: int
    y0: int
    x1: int
    y1: int
    level: int
```

A field may be a pixel, tile, region, channel, pooled block, hierarchy node, or derived field. Address semantics must be explicit in the associated field schema.

### 6.8 Source-to-target dependency

```python
@dataclass(frozen=True)
class SourceTargetDependencyDTO:
    source_fields: tuple[VPMFieldAddressDTO, ...]
    target_fields: tuple[VPMFieldAddressDTO, ...]
    score: float
    score_semantics: str
    direction: str
    stability: float | None
    support_count: int
```

The score semantics must state whether the value represents correlation, mutual information, regression weight, removal effect, keep effect, neighbor agreement, counterfactual effect, or another defined quantity.

### 6.9 Evidence VPM

```python
@dataclass(frozen=True)
class EvidenceVPMDTO:
    evidence_vpm_id: str
    source_vpm_id: str
    target_schema_id: str
    method_id: str
    content_digest: str
    png_artifact_ref: str
    dependencies: tuple[SourceTargetDependencyDTO, ...]
```

The Evidence VPM must be machine-readable and visually inspectable. It must not be only a rendered heatmap detached from exact values.

### 6.10 Prediction

```python
@dataclass(frozen=True)
class PerceptionPredictionDTO:
    prediction_id: str
    model_id: str
    source_vpm_id: str
    predicted_target_vpm_id: str
    action_candidates: tuple[ActionCandidateDTO, ...]
    selected_action: ActionValueDTO | None
    confidence: float
    confidence_semantics: str
    status: str
    evidence_vpm_id: str
    neighbor_evidence: tuple[NeighborEvidenceDTO, ...]
    findings: tuple[PerceptionFindingDTO, ...]
```

Recommended statuses:

```text
accepted
low_confidence
rejected_out_of_distribution
rejected_ambiguous
rejected_missing_evidence
unsupported_action_schema
```

## 7. Visual encoding

### 7.1 Source image encoder

The first source encoder should be deliberately boring and reversible.

Required steps:

1. Validate dimensions, channels, dtype, and finite values where relevant.
2. Convert the declared input color space to a declared canonical color space.
3. Apply an explicit resize policy only if configured.
4. Apply an explicit quantization policy only if configured.
5. Preserve stable orientation and coordinates.
6. Serialize to deterministic PNG bytes.
7. Compute content-derived identity.

No implicit crop, augmentation, histogram correction, denoising, object detection, or learned embedding is allowed.

```python
@dataclass(frozen=True)
class SourceImageEncoderSpecDTO:
    version: str
    target_width: int | None
    target_height: int | None
    resize_mode: str
    color_space: str
    channels: tuple[str, ...]
    quantization_levels: int
    alpha_policy: str
```

### 7.2 Target action encoder

The action encoder must deterministically generate a PNG schema from an action vocabulary or declared action schema.

For a bounded discrete vocabulary:

```text
one row per action
one stable cell or region per action
intensity = membership or score
```

For parameterized actions:

```text
action family fields
parameter fields
validity mask fields
optional uncertainty fields
```

The mapping from action to pixels must be canonical and preserved in a `TargetFieldSchemaDTO`.

### 7.3 Decoder

The decoder converts a predicted target VPM into ranked action candidates. It must not assume that the highest single pixel is always the winner. Decoding rules are schema-owned and deterministic.

## 8. Dataset assembly and validation

A Perception dataset is not merely a directory of images. It is an immutable manifest of exact pairings.

```python
@dataclass(frozen=True)
class PerceptionDatasetManifestDTO:
    dataset_id: str
    source_encoder_spec_id: str
    target_encoder_spec_id: str
    action_schema_id: str
    interactions: tuple[str, ...]
    split_assignments: tuple[SplitAssignmentDTO, ...]
    manifest_digest: str
```

Validation must detect:

- duplicate interaction identities;
- conflicting actions for byte-identical source images;
- missing source images;
- missing action encodings;
- non-monotonic sequence indices;
- duplicate timestamps where prohibited;
- source/next-state dimension drift;
- action schema drift;
- target encoder drift;
- source images assigned across prohibited splits;
- near-duplicate leakage between train and evaluation sets;
- temporal leakage where adjacent frames cross splits;
- inconsistent preprocessing metadata.

Conflicting actions for visually identical states are not automatically errors. They may indicate hidden state, multiple valid policies, human inconsistency, stochasticity, delayed actions, or bad alignment. The manifest validator should emit findings rather than silently collapse conflicts.

## 9. Learning and inference modes

The package should implement a progression from simple, inspectable methods to optional more capable providers.

### 9.1 Mode A: whole-VPM nearest neighbor

Baseline:

```text
unknown source VPM
    -> distance to known source VPMs
    -> weighted action vote
    -> predicted target VPM
```

Supported initial distances:

- normalized L1;
- normalized L2;
- Hamming distance after quantization;
- tile-wise distance;
- masked distance.

This baseline proves the end-to-end contract without claiming learned semantics.

### 9.2 Mode B: evidence-weighted nearest neighbor

Field weights are learned or estimated from the training set. Similarity becomes:

```text
high weight for action-discriminative fields
low weight for irrelevant fields
separate reporting for unexpected fields
```

This is the first genuinely useful Perception policy.

### 9.3 Mode C: sparse linear translator

Flatten or tile source fields into bounded features and learn a sparse mapping to target fields.

Initial production implementation should use a deterministic NumPy solver with explicit regularization and bounded dimensions. If the repository does not already contain a suitable solver, an initial ridge regression translator is preferable to introducing a new dependency. L1 sparsity may remain research until implemented deterministically and tested.

Outputs must include the exact source-to-target coefficient matrix and feature schema.

### 9.4 Mode D: decision tree provider

A small tree provider may later expose threshold rules. It should be optional and may require a separate dependency extra.

### 9.5 Mode E: learned embedding or neural provider

This remains out of the first production slice. Provider protocols should allow it later without changing DTO contracts.

## 10. Model aggregate

```python
@dataclass(frozen=True)
class PerceptionModelDTO:
    model_id: str
    model_family: str
    dataset_id: str
    source_encoder_spec_id: str
    target_encoder_spec_id: str
    field_schema_id: str
    training_spec_id: str
    learned_parameters_ref: str
    evidence_summary_id: str
    evaluation_id: str
    model_digest: str
```

Model identity must change when any of the following changes:

- dataset manifest;
- split assignments;
- source encoding;
- target encoding;
- field schema;
- algorithm family;
- hyperparameters;
- random seed;
- learned parameters;
- calibration data;
- action decoder.

## 11. Confidence and rejection

A prediction must not be accepted simply because an action has the largest score.

Initial confidence signals:

- nearest-neighbor distance;
- margin between first and second action candidates;
- agreement among nearest neighbors;
- distance from the training distribution;
- selected-evidence coverage;
- expected-evidence availability;
- model-family agreement when multiple providers are evaluated;
- target reconstruction quality.

Calibration must be explicit and must use a dataset split separate from final evaluation.

Rejection is a core capability, not an error path.

## 12. Optional semantic annotations

The generic runtime operates without annotations. Annotations add hypotheses and diagnostic power.

### 12.1 Region annotation

```python
@dataclass(frozen=True)
class PerceptionRegionAnnotationDTO:
    annotation_id: str
    source_image_id: str
    label: str | None
    role: str | None
    fields: tuple[VPMFieldAddressDTO, ...]
    properties: tuple[AnnotationPropertyDTO, ...]
    provenance: ProvenanceDescriptorDTO
```

A label is optional. An anonymous region remains valid.

### 12.2 Aggregate annotation

For crowds, forests, traffic, textures, or distributed phenomena:

```python
@dataclass(frozen=True)
class AggregateAnnotationDTO:
    annotation_id: str
    aggregate_type: str
    support_fields: tuple[VPMFieldAddressDTO, ...]
    measured_properties: tuple[AnnotationPropertyDTO, ...]
```

Possible properties include count, density, occupied area, repetition, orientation distribution, motion estimate, texture statistics, or spatial dispersion.

### 12.3 Relation annotation

```python
@dataclass(frozen=True)
class RelationAnnotationDTO:
    relation_id: str
    relation_type: str
    member_annotation_ids: tuple[str, ...]
    derived_fields: tuple[VPMFieldAddressDTO, ...]
    value: float | str | None
```

The runtime must support the possibility that the relation between regions matters more than either region independently.

## 13. Evidence-conformance contracts

```python
@dataclass(frozen=True)
class EvidenceExpectationDTO:
    expectation_id: str
    source_annotation_ids: tuple[str, ...]
    expected_target_fields: tuple[VPMFieldAddressDTO, ...]
    required_relation_ids: tuple[str, ...]
    minimum_registration: float | None
    maximum_unexplained_registration: float | None
    forbidden_annotation_ids: tuple[str, ...]
```

Conformance compares expected and observed source-to-target relationships.

Possible results:

```text
confirmed
confirmed_with_unexpected_evidence
missing_expected_evidence
forbidden_evidence_present
wrong_target_placement
unstable_evidence
inconclusive
```

The package must not collapse these into a single boolean.

## 14. Evidence discrepancy and discovery

The most valuable artifact may be the mismatch between expected and observed evidence.

Required visual outputs:

```text
Observed Evidence VPM
Expected Evidence VPM
Difference VPM
Unexplained Evidence VPM
```

The difference encoding must distinguish:

- expected and observed;
- expected but absent;
- observed but unexpected;
- observed in the wrong target field;
- unstable across observations;
- insufficiently supported;
- unresolved due to ambiguity.

Unexpected evidence must remain addressable even when no semantic label exists.

```python
@dataclass(frozen=True)
class UnexplainedEvidenceDTO:
    unexplained_id: str
    source_fields: tuple[VPMFieldAddressDTO, ...]
    affected_target_fields: tuple[VPMFieldAddressDTO, ...]
    contribution_score: float
    recurrence_count: int
    stability: float
    intervention_effect: float | None
    prototype_ref: str | None
    suggested_labels: tuple[SuggestedLabelDTO, ...]
```

Suggested labels are hypotheses. They must never replace the measured evidence identity.

## 15. Intervention tests

Correlation or predictive importance is not automatically causality.

The runtime should support deterministic interventions:

1. **Keep-only** — retain selected fields and neutralize the remainder.
2. **Remove-only** — neutralize selected fields and retain the remainder.
3. **Valid replacement** — replace a selected region with another valid observed region.
4. **Counterfactual swap** — replace a component or relation while preserving the rest.
5. **Random control** — intervene on an equal-sized random field set.
6. **Inverted ranking control** — intervene on the least important fields.
7. **Relation-preserving mutation** — alter appearance while preserving a declared relation.
8. **Relation-changing mutation** — alter the relation while preserving member appearances where possible.

Each intervention must record intended and actual changes.

```python
@dataclass(frozen=True)
class PerceptionInterventionDTO:
    intervention_id: str
    source_vpm_id: str
    method: str
    intended_fields: tuple[VPMFieldAddressDTO, ...]
    changed_fields: tuple[VPMFieldAddressDTO, ...]
    preserved_fields: tuple[VPMFieldAddressDTO, ...]
    replacement_source_ids: tuple[str, ...]
    collateral_change_score: float
    validity_status: str
    output_prediction_id: str
```

## 16. Temporal perception

Single-frame policy inference is only valid when the frame contains sufficient state.

The package must support configurable temporal windows:

```text
frame_t -> action_t
frame_t-k ... frame_t -> action_t
frame_t + action_t -> frame_t+1
```

A temporal source VPM may be represented as:

- channel stacking;
- tiled frame sequence;
- frame-difference channels;
- motion summary fields;
- a deterministic visual montage.

The temporal encoding must still materialize as an addressable PNG/VPM.

When identical current frames map to conflicting actions but preceding frames disambiguate them, the runtime should report that single-frame state is incomplete rather than treating the conflicts as random label noise.

## 17. Service architecture

Recommended internal layering:

```text
DTOs
  -> pure encoders and evaluators
  -> Engines
  -> Services
  -> Store protocols
  -> in-memory Stores
  -> optional SQL Stores later
```

### 17.1 Engines

- `SourceVPMEncoderEngine`
- `TargetVPMEncoderEngine`
- `DatasetAssemblyEngine`
- `FieldExtractionEngine`
- `SimilarityEngine`
- `PerceptionTrainingEngine`
- `PerceptionInferenceEngine`
- `EvidenceExtractionEngine`
- `EvidenceConformanceEngine`
- `InterventionEngine`
- `UnexpectedEvidenceEngine`
- `TransitionInferenceEngine`

Engines are deterministic application logic over DTOs and arrays. They do not own persistence.

### 17.2 Services

- `PerceptionDatasetService`
- `PerceptionTrainingService`
- `PerceptionInferenceService`
- `PerceptionEvidenceService`
- `PerceptionValidationService`

Services coordinate Stores and Engines. DTOs are the only objects crossing Store boundaries.

### 17.3 Store protocols

Initial protocols:

```python
class PerceptionDatasetStore(Protocol): ...
class PerceptionModelStore(Protocol): ...
class PerceptionEvidenceStore(Protocol): ...
class PerceptionPredictionStore(Protocol): ...
```

The first production runtime should default to in-memory Stores. Persistence is an explicit later stage.

## 18. Proposed source tree

```text
packages/perception/
  pyproject.toml
  README.md
  src/zeromodel/perception/
    __init__.py
    dto.py
    image.py
    action.py
    fields.py
    dataset.py
    encoding.py
    similarity.py
    neighbors.py
    translation.py
    training.py
    inference.py
    evidence.py
    expectations.py
    interventions.py
    discovery.py
    evaluation.py
    engines.py
    services.py
    stores.py
    in_memory.py
    exceptions.py
  tests/
    test_image_encoding.py
    test_action_encoding.py
    test_dataset_manifest.py
    test_nearest_neighbor_inference.py
    test_rejection.py
    test_evidence_fields.py
    test_keep_remove_interventions.py
    test_expectation_conformance.py
    test_unexpected_evidence.py
    test_temporal_windows.py
    test_public_api.py
```

This may be split into smaller files only as code pressure appears. The first implementation should avoid premature RMDTO fragmentation.

## 19. Public API sketch

```python
from zeromodel.perception import (
    ActionSchema,
    PerceptionDatasetBuilder,
    PerceptionRuntime,
    SourceImageEncoderSpec,
    TargetActionEncoderSpec,
)

runtime = PerceptionRuntime.in_memory(
    source_encoder=SourceImageEncoderSpec.identity_png(),
    target_encoder=TargetActionEncoderSpec.discrete(
        actions=("LEFT", "RIGHT", "FIRE", "NOOP"),
    ),
)

dataset = runtime.datasets.build(recorded_interactions)
model = runtime.training.fit_nearest_neighbor(dataset)

prediction = runtime.inference.predict(
    model_id=model.model_id,
    image=unknown_image,
)

assert prediction.status in {
    "accepted",
    "low_confidence",
    "rejected_out_of_distribution",
    "rejected_ambiguous",
}
```

Evidence extraction:

```python
evidence = runtime.evidence.explain_prediction(prediction.prediction_id)
```

Expectation testing:

```python
result = runtime.validation.check_expectation(
    model_id=model.model_id,
    image=unknown_image,
    expectation=expectation,
)
```

## 20. Evaluation

### 20.1 Prediction metrics

- top-1 action accuracy;
- top-k action recall;
- target VPM reconstruction error;
- calibration error;
- rejection accuracy;
- out-of-distribution rejection;
- nearest-neighbor agreement;
- action margin distribution.

### 20.2 Evidence metrics

- keep-only sufficiency;
- remove-only necessity;
- random-control separation;
- evidence stability across nearby observations;
- evidence completeness relative to full input;
- expected evidence coverage;
- forbidden evidence contribution;
- unexplained contribution;
- target-field placement accuracy;
- counterfactual direction correctness.

### 20.3 Transition metrics

- next-state VPM reconstruction;
- action-conditioned improvement over no-action baseline;
- localized change prediction;
- temporal alignment sensitivity;
- consistency across repeated transitions.

### 20.4 Failure taxonomy

Every evaluation should classify failures rather than returning only aggregate accuracy:

```text
representation_failure
alignment_failure
action_schema_failure
insufficient_visible_state
ambiguous_policy
model_underfit
model_shortcut
unexpected_dependency
expectation_failure
transition_failure
out_of_distribution
```

## 21. Testing strategy

### 21.1 Fast tests

Fast tests must be bounded, deterministic, NumPy/Pillow-only, and finish within the repository fast-suite budget.

Required fixtures:

1. Tiny synthetic images where one pixel determines one action.
2. Images where a region, not one pixel, determines an action.
3. Images where relative placement determines the action.
4. Images with a large irrelevant background.
5. Images with a spurious shortcut region.
6. Identical images with conflicting actions.
7. Temporal pairs where a previous frame resolves the conflict.
8. Out-of-distribution unknown images.
9. Parameterized actions.
10. Before/action/after transitions.

### 21.2 Integration tests

Integration tests may cover artifact storage, video adapters, SQL persistence, and larger image sets. They should be authored but not run without explicit authorization under repository policy.

### 21.3 Research tests

Space Invaders and other games belong primarily in `research/` until a bounded claim is promoted. The production package should be proven first with synthetic, domain-neutral fixtures.

## 22. Security and integrity

- Reject malformed or decompression-bomb images before materialization.
- Bound width, height, channels, and decoded byte size.
- Do not execute metadata or embedded scripts.
- Record exact decoder library/version where reproducibility requires it.
- Do not silently transpose orientation metadata.
- Preserve source digest and normalized-image digest separately.
- Do not accept model artifacts whose schema or digest does not match the manifest.
- Keep prediction and evidence artifacts immutable.

## 23. Research lineage to leverage

The package architecture should borrow mechanisms, not claims, from several research families:

- concept bottleneck models for explicit intermediate evidence;
- TCAV-style concept/result hypothesis tests;
- ACE-style unknown concept discovery;
- concept completeness for evidence sufficiency;
- ROAR/KAR-style remove/keep evaluation;
- right-for-the-right-reasons constraints;
- shortcut-learning diagnostics;
- counterfactual concept intervention;
- nearest-neighbor and prototype reasoning;
- deep-learning test adequacy and surprise detection.

ZeroModel's distinctive contribution is the unification of source, target, evidence, expected evidence, unexplained evidence, and discrepancy as deterministic, addressable, provenance-backed visual artifacts.

## 24. Delivery plan

The safest way to build this against `main` is a sequence of small vertical slices. Do not attempt the full design in one PR.

### Stage P0 — Architecture and package registration

Deliverables:

- this design document;
- `package-boundaries.toml` entry;
- root pytest/mypy source paths;
- package `pyproject.toml` and README;
- empty namespace with version/public API contract;
- package-boundary tests.

No scientific behavior.

### Stage P1 — Deterministic representation slice

```text
arbitrary image -> SourceVPMDTO
arbitrary discrete action -> TargetVPMDTO
```

Deliverables:

- image and action DTOs;
- deterministic PNG encoders;
- target field schema;
- content identity;
- round-trip tests;
- bounded security checks.

### Stage P2 — Dataset ledger slice

```text
image + action [+ next image] -> immutable dataset manifest
```

Deliverables:

- `RecordedInteractionDTO`;
- manifest builder;
- split assignment;
- conflict and leakage findings;
- in-memory dataset Store;
- focused tests.

### Stage P3 — Baseline inference slice

```text
known image/action pairs + unknown image -> predicted action
```

Deliverables:

- whole-VPM nearest-neighbor model;
- model identity;
- ranked candidates;
- confidence and rejection;
- neighbor evidence;
- synthetic validation.

This is the first complete user-visible capability.

### Stage P4 — Evidence-weighted inference slice

Deliverables:

- field partition schema;
- simple field relevance estimation;
- weighted similarity;
- Evidence VPM;
- keep-only/remove-only controls;
- comparison with whole-image baseline.

### Stage P5 — Sparse translator slice

Deliverables:

- source-to-target matrix translator;
- deterministic fitting;
- explicit coefficient artifact;
- target VPM reconstruction;
- completeness and calibration metrics.

### Stage P6 — Annotation and conformance slice

Deliverables:

- region, aggregate, and relation annotations;
- expected evidence contracts;
- observed/expected/difference VPMs;
- conformance statuses;
- synthetic right-result/wrong-evidence tests.

### Stage P7 — Unexpected evidence discovery slice

Deliverables:

- unexplained-field extraction;
- spatial grouping;
- cross-observation clustering;
- prototypes;
- intervention validation;
- hypothesis labels as optional metadata.

### Stage P8 — Temporal and transition slice

Deliverables:

- frame-window source encodings;
- action-conditioned next-state prediction;
- hidden-state findings;
- temporal alignment diagnostics.

### Stage P9 — Persistence and artifact integration

Only after DTOs stabilize:

- artifact references;
- SQLAlchemy Store implementations;
- relational ownership rules;
- reconstruction tests;
- durable model/evidence/prediction aggregates.

### Stage P10 — Space Invaders validation adapter

Use the existing video action-set domain as one validation provider.

The adapter may annotate tanks, aliens, bullets, and relations, but those concepts remain outside the Perception production package.

## 25. First implementation decision

The first implementation should target one narrow statement:

> Given a bounded set of arbitrary PNG images paired with bounded discrete actions, ZeroModel Perception can deterministically construct both visual representations and return a ranked, rejectable action prediction for a previously unseen PNG, together with the exact known examples and image fields supporting that prediction.

This statement is broad enough to prove the architecture and narrow enough to test without neural networks, semantic labels, SQL, video, or game-specific code.

## 26. Acceptance criteria for the first complete vertical slice

1. A caller can provide arrays or PNG bytes plus action values.
2. Source and target VPM PNGs are deterministic and content-addressed.
3. The same manifest and specs produce the same model identity.
4. An unknown image returns ranked actions.
5. The runtime can reject an image outside calibrated support.
6. The prediction includes nearest supporting observations.
7. The prediction includes an Evidence VPM or exact field weights.
8. Large irrelevant regions can be down-weighted in the evidence-weighted stage.
9. A deliberately planted shortcut can be exposed as influential evidence.
10. No production code contains Space Invaders domain concepts.
11. DTOs are the only values crossing Store boundaries.
12. Focused tests pass under the fast tier.

## 27. Open decisions

These decisions should be resolved during the relevant stages, not prematurely:

- exact target PNG layout for composite and continuous actions;
- initial field partition strategy;
- whether Pillow belongs in the base dependency or a PNG extra;
- ridge versus L1 sparse translation in production;
- confidence calibration method;
- exact artifact-store integration boundary;
- whether Perception owns its own generic provenance DTO or reuses an existing one;
- whether temporal windows are stacked channels or tiled frames;
- when learned embedding providers become a separate distribution;
- whether SQL persistence belongs under the existing SQLAlchemy package or a future Perception persistence package.

## 28. Architectural invariants

1. Any image domain is valid; no domain ontology is required.
2. Source and target representations are both deterministic visual artifacts.
3. Raw image semantics are optional annotations, not prerequisites.
4. Predictions remain inspectable and rejectable.
5. Evidence is a first-class artifact, not a decorative heatmap.
6. Expected evidence is a hypothesis, not ground truth.
7. Unexpected evidence is preserved, not normalized away.
8. Predictive association is not mislabeled as causality.
9. Interventions record collateral changes.
10. Training, calibration, and evaluation identities are immutable.
11. No hidden path may bypass an explicit evidence bottleneck when a bottleneck model is claimed.
12. Production claims must be supported by bounded tests or promoted evidence.
13. Scientific experiments remain in `research/` until promoted.
14. Persistence remains explicit and DTO-only.
15. Core remains conservative.

## 29. Summary

`zeromodel-perception` is the package that closes the loop between arbitrary visual observation and visual policy execution:

```text
observe
  -> encode source PNG/VPM
  -> encode action/result PNG/VPM
  -> learn visual relationship
  -> expose evidence
  -> test expected evidence
  -> discover unexpected evidence
  -> predict an action for an unseen image
  -> reject when support is insufficient
  -> record the new observation
```

Its purpose is not merely to classify images. It is to make the relationship between visual information and action/results addressable, testable, versionable, and inspectable within the ZeroModel architecture.
