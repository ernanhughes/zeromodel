# ZeroModel Visual Policy Addressing
## Unified Deterministic and Embedding Architecture — Comprehensive Design and External-Review Brief

**Document status:** Architecture proposal with one implemented baseline  
**Repository:** `ernanhughes/zeromodel`  
**Current implemented baseline:** `VisualSignReader` / visual-index contract v2  
**Primary purpose:** External technical review before the next implementation stage  
**Scope:** Visual observation → governed VPM policy address → action or rejection  
**Non-scope:** A universal computer-vision agent, arbitrary game-winning system, or autonomous physical controller

---

# 1. Purpose of this document

This document defines the next-stage architecture for **visual policy addressing in ZeroModel**.

The system should allow an image, screenshot, camera frame, or short video observation to identify the nearest valid state or region inside a Visual Policy Map (VPM), after which the existing policy artifact selects the action.

The intended abstraction is:

```text
observation
    ↓
visual address provider
    ↓
accepted VPM row or policy region
    ↓
VPMPolicyLookup
    ↓
action + evidence + artifact trace
```

The architecture deliberately supports more than one way to produce the visual address:

```text
Path A — deterministic visual addressing
    no model, no training, tightly controlled domain

Path B — frozen embedding addressing
    pretrained encoder, no user training

Path C — automatically adapted embedding addressing
    pretrained encoder + lightweight fitting performed by ZeroModel

Path D — expert-supplied perception
    custom trained encoder or projection, fully identity-pinned
```

The product principle is:

> **Use the least intelligent visual-addressing mechanism that survives the domain’s validation requirements.**

This preserves the cheap, deterministic path for users who cannot or should not train models, while allowing learned visual embeddings when real variation makes a fixed codebook inadequate.

---

# 2. Executive summary

ZeroModel already contains an implemented deterministic visual reader for a bounded arcade world.

That implementation:

- converts canonical images into fixed integer feature vectors;
- stores those vectors in a visual-index artifact;
- binds the visual index to an exact policy artifact using an `addresses` relation;
- resolves an observation to a policy row or returns a rejection;
- preserves feature, calibration, index, policy, and decision identities;
- exhaustively recovers all 112 canonical arcade states;
- reproduces the symbolic policy across 2,401 four-target waves;
- validates a reachable ambiguity rejection;
- does not mutate caller-owned image arrays;
- explicitly limits its claim to exact quantized feature-codeword addressing in the committed fixture.

This is a useful **Level 0 baseline**, but it does not provide broad visual generalization.

The proposed architecture extends the same artifact and policy contract with a new family of **embedding-address artifacts**:

```text
image
    ↓
pinned visual encoder
    ↓
global embedding + optional local patch embeddings
    ↓
prototype or region index
    ↓
calibrated acceptance / ambiguity / evidence checks
    ↓
VPM row or region
    ↓
policy action
```

The user should not normally need to train a model. The preferred progression is:

```text
1. Try deterministic compilation.
2. If validation fails, try a frozen pretrained encoder.
3. If that fails, automatically fit a small projection or prototypes.
4. Permit a custom trained encoder only for advanced domains.
```

The architecture therefore separates:

- **perception**, which may require intelligence;
- **policy**, which remains a durable identified artifact;
- **address calibration**, which defines what observations are safe to map;
- **execution**, which performs the selected action;
- **verification**, which confirms the expected result.

ZeroModel’s distinctive contribution is not the invention of image embeddings or nearest-neighbour retrieval.

It is the governed handoff:

> **A visual representation becomes an identity-bearing, calibrated and inspectable address into an independently identified policy artifact.**

---

# 3. Status labels

Every claim in this document uses one of four statuses.

| Status | Meaning |
|---|---|
| **Implemented** | Present and tested in the current repository. |
| **Next experiment** | Proposed immediate research or implementation work. |
| **Research option** | Plausible future design requiring evidence before adoption. |
| **Boundary** | Explicitly not claimed by the current system. |

This distinction is essential. The document must not allow proposed embedding capabilities to be mistaken for already implemented features.

---

# 4. The problem being solved

## 4.1 Symbolic policy addressing

The current finite policy reader expects a stable symbolic row identifier:

```text
tank=3|target=5|cooldown=0
    ↓
VPMPolicyLookup
    ↓
RIGHT
```

This works when the runtime system already exposes the exact state.

Many real systems instead provide observations:

- an image from a fixed camera;
- a screenshot of a user interface;
- a frame from a game;
- a remote-desktop stream;
- a short video clip;
- a sensor visualization;
- an equipment panel.

The visual-addressing problem is:

> **Given an observation, identify the policy row or policy region that it safely addresses without first requiring a human-written symbolic parser.**

## 4.2 What “without a semantic middleman” means

It does not mean that raw pixels magically become policy actions.

Some transformation is always required.

The distinction is between:

### Human-semantic parsing

```text
image
    ↓
car count = 1
car position = bridge centre
direction = approaching
    ↓
policy row
```

and:

### Latent visual addressing

```text
image or clip
    ↓
visual representation
    ↓
nearest safe policy region
```

The second route may still use a pretrained encoder, patch features, prototypes and calibration. It avoids requiring a bespoke object/state parser for every application.

## 4.3 Why this is useful

A stable visual-policy region can be reused at runtime without invoking a reasoning model.

Intelligence is used when needed to:

- select or train an encoder;
- discover useful visual structure;
- define or adapt prototypes;
- recover from novel interfaces;
- propose new policy mappings.

Once reviewed and calibrated, the stable result becomes a durable artifact.

This follows the larger ZeroModel principle:

> **When a decision is genuinely new, invoke intelligence. When the policy is stable, build a sign.**

---

# 5. Core architecture

## 5.1 Unified runtime pipeline

```text
Raw observation
    ↓
Observation canonicalizer
    ↓
VisualAddressProvider
    ↓
VisualAddressDecision
    ├── accepted row / region
    └── rejection with evidence
    ↓
VPMPolicyLookup
    ↓
PolicyDecision
    ↓
Action executor
    ↓
Postcondition verifier
```

The address provider may be deterministic or learned. The policy reader should not need to know which one was used.

## 5.2 Shared interface

```python
from typing import Protocol

class VisualAddressProvider(Protocol):
    def read(self, observation: "VisualObservation") -> "VisualAddressDecision":
        ...
```

All providers return the same conceptual decision:

```python
@dataclass(frozen=True)
class VisualAddressDecision:
    accepted: bool
    reason: str

    observation_digest: str
    representation_digest: str

    provider_kind: str
    provider_version: str
    address_artifact_id: str
    policy_artifact_id: str

    nearest_row_id: str | None
    nearest_score: float | None
    second_row_id: str | None
    second_score: float | None
    ambiguity_measure: float | None

    local_evidence_score: float | None
    visible_evidence_fraction: float | None
    critical_evidence_present: bool | None

    matched_row_id: str | None
    exact_match: bool
    accepted_by: tuple[str, ...]

    trace: dict[str, object]
```

The address decision should remain separate from the policy decision.

Composition:

```python
address = provider.read(observation)

if not address.accepted:
    return RejectedVisualPolicyDecision.from_address(address)

policy_decision = policy_reader.read(address.matched_row_id)
return VisualPolicyDecision.combine(address, policy_decision)
```

This separation prevents perception evidence from being conflated with action values.

---

# 6. Architecture ladder

## 6.1 Level 0 — exact deterministic codebook

**Status: Implemented**

```text
canonical image
    ↓
fixed integer grayscale / pooling / quantization
    ↓
exact feature codeword
    ↓
VPM row
```

Best fit:

- generated interfaces;
- stable displays;
- finite game screens;
- equipment panels with fixed rendering;
- fixed camera with negligible environmental variation;
- known menus and alerts.

Strengths:

- no model;
- no training;
- tiny dependency footprint;
- strong reproducibility;
- simple artifact identity;
- exact replay;
- easy audit.

Limitations:

- little or no tolerance beyond quantization;
- every valid state must be enumerable;
- the renderer must visibly encode all policy-relevant state;
- no learned invariance to lighting, translation, occlusion or appearance.

## 6.2 Level 1 — deterministic neighbourhoods

**Status: Next experiment**

```text
image
    ↓
fixed deterministic descriptor
    ↓
nearest known codeword
    ↓
per-state calibrated acceptance basin
```

Potential enhancements:

- integer squared distances;
- per-row acceptance radii;
- ratio-based ambiguity;
- classical local features;
- perceptual hashes;
- multiscale deterministic descriptors;
- perturbation-calibrated thresholds.

This remains a “dumb” architecture in the productive sense:

- no training;
- no external model;
- predictable;
- suitable for modest hardware.

It should only be adopted if it demonstrates meaningful non-exact tolerance without unsafe false acceptance.

## 6.3 Level 2 — frozen pretrained embeddings

**Status: Next experiment**

```text
image
    ↓
frozen pretrained visual encoder
    ↓
embedding and patch tokens
    ↓
stored prototypes
    ↓
calibrated VPM region
```

No user training is required.

The user supplies:

- example images;
- a VPM policy artifact;
- optional row labels or demonstrations;
- a validation set or generated variants.

ZeroModel performs:

- embedding extraction;
- prototype selection;
- clustering if necessary;
- threshold calibration;
- OOD tests;
- artifact construction.

Candidate encoders include DINOv2, DINOv3 or another pinned vision foundation model. The architecture must remain encoder-agnostic.

## 6.4 Level 3 — automatic lightweight adaptation

**Status: Research option**

```text
frozen encoder output
    ↓
small projection or aggregation head
    ↓
policy-aware embedding
```

The user does not run an ML training workflow manually.

A ZeroModel compile command may fit:

- a linear projection;
- a shallow MLP;
- a NetVLAD-style aggregator;
- prototype clusters;
- class-conditional covariance;
- a metric-learning head.

The resulting weights, dataset, configuration and calibration are all identified and frozen.

This is best treated as **compilation with fitting**, not as an opaque autonomous training system.

## 6.5 Level 4 — expert custom perception

**Status: Research option**

Advanced users may supply:

- a custom encoder;
- a fine-tuned foundation model;
- a video encoder;
- a domain-specific detector;
- a segmentation model;
- a multimodal sensor fusion model.

ZeroModel’s responsibility becomes:

- verify the supplied model identity;
- verify preprocessing identity;
- calibrate the representation against the policy;
- preserve artifact lineage;
- reject unsupported observations;
- expose evidence and drift.

---

# 7. Artifact model

The architecture should use separate artifacts for separate responsibilities.

## 7.1 Policy artifact

**Status: Implemented**

Contains:

- stable row or region IDs;
- action values;
- evidence metrics;
- source mapping;
- policy provenance;
- artifact identity.

It must not contain encoder weights or raw image datasets.

## 7.2 Visual address artifact

Contains either deterministic descriptors or learned prototypes.

Common metadata:

```json
{
  "kind": "visual_address_index",
  "address_kind": "deterministic_codebook | embedding_prototype | embedding_region",
  "address_version": "...",
  "addresses_policy_artifact_id": "...",
  "representation_dimension": 256,
  "distance_or_similarity": "cosine",
  "normalization": "l2",
  "calibration_digest": "...",
  "observation_spec_digest": "...",
  "provider_spec_digest": "..."
}
```

Common provenance:

```json
{
  "kind": "visual_address_index",
  "parents": [
    {
      "artifact_id": "<policy artifact id>",
      "relation": "addresses"
    }
  ]
}
```

## 7.3 Encoder manifest artifact

Required for learned paths.

```json
{
  "kind": "visual_encoder_manifest",
  "architecture": "dinov3_vits16",
  "weights_digest": "sha256:...",
  "code_version": "...",
  "framework": "pytorch",
  "framework_version": "...",
  "preprocessing_digest": "...",
  "output_contract": {
    "global_dimension": 384,
    "patch_dimension": 384,
    "normalization": "l2"
  },
  "license_record": "...",
  "source_record": "..."
}
```

The manifest identifies the encoder; it need not embed the weight bytes inside a `.vpm` file.

## 7.4 Projection or aggregator artifact

Required only when fitted.

```json
{
  "kind": "visual_projection",
  "architecture": "linear | mlp | netvlad",
  "weights_digest": "sha256:...",
  "input_dimension": 384,
  "output_dimension": 256,
  "training_manifest_digest": "...",
  "optimizer_spec_digest": "...",
  "seed": 1234,
  "parent_encoder_manifest_id": "..."
}
```

## 7.5 Calibration artifact

Calibration should be independently inspectable rather than hidden in one metadata dictionary.

It records:

- development dataset identity;
- calibration split identity;
- OOD split identity;
- global and per-region thresholds;
- ambiguity rule;
- local evidence rule;
- operating point;
- false-acceptance estimate;
- false-rejection estimate;
- subgroup results;
- corruption and occlusion curves;
- kill-condition result.

## 7.6 Validation report artifact

A report should link the complete chain:

```text
validation report
    validates → visual address artifact
    validates → policy artifact
    validates → encoder manifest
    validates → calibration artifact
```

This allows the deployed decision trace to cite exactly what was validated.

## 7.7 Dataset manifest

The dataset itself may live outside the repository, but the manifest must be identity-bearing.

It should include:

- observation IDs;
- labels or policy-row bindings;
- corruption family;
- split assignment;
- source category;
- transformation parameters;
- content digests;
- licensing and privacy metadata where relevant.

---

# 8. Observation contract

## 8.1 Image observation

```python
@dataclass(frozen=True)
class ImageObservation:
    pixels: object
    timestamp: str | None
    source_id: str
    sequence_id: str | None
    metadata: dict[str, object]
```

Canonicalization must define:

- colour channel order;
- orientation;
- crop;
- resize;
- aspect-ratio policy;
- value range;
- normalization;
- alpha handling;
- camera or screen rectification;
- digest semantics.

## 8.2 Video observation

```python
@dataclass(frozen=True)
class VideoObservation:
    frames: tuple[ImageObservation, ...]
    start_timestamp: str
    end_timestamp: str
    sample_rate_hz: float
```

Video becomes necessary when the action depends on motion or history:

- approaching versus departing;
- opening versus closing;
- damage direction;
- interface transition;
- object entering versus leaving.

## 8.3 Multisensor observation

```python
@dataclass(frozen=True)
class MultisensorObservation:
    image_or_video: ImageObservation | VideoObservation
    scalar_signals: dict[str, float | int | bool | str]
    previous_policy_state: str | None
```

For safety-relevant infrastructure, visual evidence should usually supplement rather than replace physical sensors.

---

# 9. Deterministic path

## 9.1 Current implemented contract

The repository currently implements:

- integer BT.601 grayscale conversion;
- exact integer box pooling;
- uniform integer quantization;
- a separate visual-index artifact;
- an `addresses` policy parent;
- compile-time separation audit;
- exact-feature fast path;
- nearest-candidate evidence;
- distance rejection;
- mathematically effective absolute-gap ambiguity rejection;
- caller-memory ownership;
- complete decision trace.

The contract is explicitly bounded. It does not claim open-world perception or learned generalization.

## 9.2 Appropriate future deterministic enhancements

A deterministic v3 experiment may compare:

1. current pooled grayscale;
2. colour histograms;
3. edge-orientation histograms;
4. multiscale pooled descriptors;
5. perceptual hashing;
6. local binary patterns;
7. classical keypoint descriptors;
8. combinations of deterministic features.

Every candidate must be treated as a declared feature contract with its own identity.

## 9.3 Per-row calibration

A global radius is controlled by the most confusable pair.

A per-row radius may be derived from:

- nearest valid neighbour;
- within-state perturbation distribution;
- distance to nearest conflicting-action region;
- explicit false-acceptance target.

A safer policy-aware radius is:

```text
maximum accepted radius for row r
    constrained by
minimum distance to any row requiring a conflicting action
```

This avoids giving a state a large radius merely because its nearest same-action neighbour is far away.

## 9.4 Deterministic kill condition

Do not keep increasing feature complexity indefinitely.

Move to the frozen-embedding path when:

- legitimate variation overlaps conflicting policy states;
- small lighting or viewpoint changes cause unacceptable rejection;
- robustness requires brittle domain-specific feature engineering;
- the deterministic descriptor becomes more complex than a pinned encoder;
- false acceptance cannot be controlled.

---

# 10. Frozen embedding path

## 10.1 Representation

A visual encoder may produce:

```text
global embedding:
    one vector representing the image

patch embeddings:
    one vector for each spatial image patch
```

The global vector supports fast candidate retrieval.

Patch vectors support local verification and occlusion reasoning.

## 10.2 Why a frozen encoder is the default learned path

A frozen encoder:

- requires no domain training;
- is easy to pin by digest;
- can be reused across applications;
- limits the trainable surface;
- allows prototypes to be rebuilt cheaply;
- is easier to govern than an end-to-end trained policy.

DINOv2 and DINOv3 are relevant because they produce general-purpose global and dense visual features. DINOv3’s official model card exposes class and patch tokens and recommends frozen-feature use before fine-tuning.

Encoder choice remains an experiment, not a locked dependency.

## 10.3 Prototype construction

For each VPM row or policy region, collect embeddings from valid examples.

Possible prototype schemes:

### Single centroid

```text
all row embeddings
    ↓
normalized mean
    ↓
one prototype
```

Simple, but may average incompatible visual modes.

### Multiple medoids

```text
row embeddings
    ↓
cluster
    ↓
representative real observations
```

Safer when one state has several appearance modes.

### Region density model

```text
row embeddings
    ↓
mean + covariance or non-parametric neighbourhood
```

Potentially more accurate, but harder to interpret and calibrate.

The first experiment should compare centroid and medoid prototypes.

## 10.4 Similarity

For normalized vectors:

```text
cosine similarity = prototype · query
cosine distance = 1 - similarity
```

The trace must record:

- nearest prototype;
- owning row;
- nearest similarity;
- second-nearest conflicting-row similarity;
- ambiguity margin;
- threshold used;
- prototype ID;
- encoder identity.

## 10.5 Policy-region versus exact-row retrieval

Two evaluation targets must be kept separate.

### Exact-row retrieval

The system identifies the original simulator or workflow state.

Useful for diagnostics and exhaustive comparison.

### Action-equivalent region retrieval

The system may confuse exact rows while still choosing the same safe action.

Useful for deployment when exact state distinctions are unnecessary.

The artifact must declare which contract it provides:

```text
address_semantics:
    exact_row
    action_equivalent_region
    hierarchical_region_then_row
```

---

# 11. Local and global evidence

## 11.1 Why global embeddings are insufficient

A global embedding may remain similar even when the exact decision-critical object is hidden.

Example:

```text
bridge scene with a car
bridge scene with the car region removed
```

The road, water, barriers and background may dominate the global descriptor.

The runtime should not accept “bridge clear” merely because the visible background resembles clear examples.

## 11.2 Two-stage retrieval

```text
query global embedding
    ↓
retrieve top candidate regions
    ↓
compare local patch evidence
    ↓
accept or reject
```

The local verification may measure:

- fraction of expected patches matched;
- spatially consistent patch correspondences;
- presence of critical patch groups;
- patch entropy;
- amount of unmasked visual evidence;
- consistency across several recent frames.

## 11.3 Critical evidence

Critical evidence should not be inferred solely from raw attention maps.

Possible sources:

1. annotated critical regions;
2. learned patch importance from held-out interventions;
3. policy-conflict analysis;
4. occlusion sensitivity tests;
5. counterfactual masking;
6. physical sensor cross-checks.

A region is decision-critical when removing it changes the safe policy action or destroys the evidence needed to distinguish conflicting actions.

## 11.4 Occlusion contract

The system must not claim that any image remains understandable after arbitrary 50% removal.

The valid claim is:

> **The address should remain stable under removal of non-critical evidence within the validated corruption envelope, and it should reject when critical evidence is missing.**

Two separate tests are required:

### Non-critical occlusion

Expected:

- same policy region;
- accepted;
- local evidence remains above threshold.

### Critical occlusion

Expected:

- rejection;
- reason `insufficient_decision_evidence`;
- no action or only a declared fail-safe action.

## 11.5 Aggregation methods

Potential aggregation methods include:

- class token;
- mean patch pooling;
- generalized mean pooling;
- NetVLAD-style aggregation;
- attention-weighted pooling;
- robust trimmed pooling;
- multiple local prototype banks.

NetVLAD is relevant because it aggregates local descriptors into a fixed-length representation and was originally designed for place recognition under appearance change.

DELG is relevant because it explicitly combines global and local image features.

These are candidates for comparison, not mandatory architecture.

---

# 12. Policy-aware embedding

## 12.1 Motivation

A generic image embedding preserves visual similarity.

A policy-aware embedding should preserve distinctions that matter to action.

Example:

```text
two visually similar images
    but
one requires STOP and one requires GO
```

These must be strongly separated.

Conversely:

```text
two visually different images
    but
both require the same safe action
```

These may occupy the same broad policy region.

## 12.2 Training relationships

The fitting process may use four relationship tiers:

```text
same exact row:
    strongest attraction

different row, same policy region:
    moderate attraction or weak separation

different row, different action:
    strong separation

critical conflicting actions:
    strongest separation
```

Criticality may come from:

- policy decision margin;
- Q-value consequence;
- declared safety relationship;
- verification properties;
- manually reviewed action-pair severity.

## 12.3 Recommended first adaptation

Do not begin with a large MLP and NetVLAD stack.

Start with:

```text
frozen encoder
    ↓
linear projection
    ↓
L2 normalization
    ↓
multiple prototypes per row
```

This provides a clean baseline for determining whether policy-aware adaptation adds value.

Only add trainable aggregation when the frozen features plus linear projection fail.

## 12.4 Training artifacts

A fitted projection must record:

- encoder identity;
- initialization;
- trainable parameter shapes;
- random seed;
- optimizer;
- learning rate;
- epochs;
- early-stopping rule;
- training split;
- validation split;
- negative sampling;
- loss definition;
- action relationship matrix;
- resulting weights digest.

---

# 13. Temporal addressing

## 13.1 Why single images may be insufficient

Some policy facts are not observable from one frame:

- approaching versus departing;
- object velocity;
- opening versus closing;
- attack direction;
- previous interface step;
- whether a warning is new or stale.

The correct address may therefore depend on:

```text
recent frames + previous accepted policy state
```

## 13.2 Temporal architecture

```text
short frame sequence
    ↓
video encoder or frame embeddings
    ↓
temporal aggregator
    ↓
temporal visual address
    ↓
VPM transition policy
```

Possible implementation levels:

1. concatenate recent frozen image embeddings;
2. exponentially weighted history;
3. small recurrent or temporal projection;
4. frozen V-JEPA-style video encoder;
5. custom domain video model.

Again, use the cheapest validated level.

## 13.3 VPM transition evidence

A temporal decision should record:

- previous accepted row;
- current candidate row;
- expected transition;
- transition likelihood or distance;
- whether the transition is allowed;
- whether the observation implies a jump;
- time since previous state.

A visually plausible state that violates the workflow or physical transition model should be rejected.

---

# 14. Compiler workflow

The intended user experience is not “train a computer-vision model.”

It is:

```bash
zeromodel visual compile \
  --policy policy.vpm \
  --examples examples/manifest.json \
  --mode auto \
  --output visual-address.vpm
```

## 14.1 `--mode deterministic`

Runs:

- feature contract;
- exact separation;
- perturbation generation;
- calibration;
- validation report.

Fails if the deterministic path does not meet targets.

## 14.2 `--mode frozen`

Runs:

- pinned encoder;
- global and patch extraction;
- prototype construction;
- calibration;
- held-out validation;
- artifact generation.

No training.

## 14.3 `--mode adapt`

Runs:

- frozen encoder;
- lightweight projection fitting;
- prototype construction;
- calibration;
- held-out evaluation;
- comparison with frozen mode.

The adapted artifact is emitted only if it improves the declared objective without violating false-acceptance limits.

## 14.4 `--mode auto`

Runs the architecture ladder in order.

Example decision:

```json
{
  "selected_mode": "frozen_embedding",
  "rejected_modes": [
    {
      "mode": "deterministic",
      "reason": "held_out_translation_false_rejection_rate_exceeded"
    }
  ],
  "selection_evidence": {
    "action_accuracy": 0.984,
    "false_acceptance_rate": 0.004,
    "false_rejection_rate": 0.021
  }
}
```

The selected mode and failed alternatives should become part of the build report.

---

# 15. Runtime reader

## 15.1 Unified reader

```python
class VisualPolicyReader:
    def __init__(
        self,
        address_provider: VisualAddressProvider,
        policy_lookup: VPMPolicyLookup,
    ) -> None:
        self.address_provider = address_provider
        self.policy_lookup = policy_lookup

    def read(self, observation: VisualObservation) -> VisualPolicyDecision:
        address = self.address_provider.read(observation)
        if not address.accepted or address.matched_row_id is None:
            return VisualPolicyDecision.rejected(address)

        policy = self.policy_lookup.read(address.matched_row_id)
        return VisualPolicyDecision.accepted(address, policy)
```

## 15.2 Rejection reasons

Common reasons:

```text
unsupported_observation_shape
observation_preprocessing_failed
no_visual_region_within_threshold
ambiguous_visual_address
insufficient_decision_evidence
critical_region_missing
unexpected_temporal_transition
encoder_identity_mismatch
address_policy_identity_mismatch
calibration_not_valid_for_source
source_drift_detected
```

## 15.3 Fail-safe actions

A rejection does not always mean “do nothing.”

The policy may declare a rejection action:

```text
web workflow:
    stop and request review

industrial machine:
    safe stop

bridge:
    lock opening mechanism

game:
    invoke exploratory policy

monitoring:
    record incident and alert
```

The fail-safe action belongs to a separate, explicit safety policy. It must not be invented by the address provider.

---

# 16. Drift and recalibration

## 16.1 Drift sources

- camera movement;
- replacement camera;
- changed UI theme;
- lighting season;
- new vehicle types;
- altered sprite rendering;
- encoder library change;
- preprocessing change;
- compression pipeline;
- domain population change.

## 16.2 Drift signals

- rising nearest distance;
- shrinking nearest/second margin;
- increasing rejection;
- changed patch-evidence distribution;
- transition inconsistencies;
- operator overrides;
- new clusters outside known regions.

## 16.3 Recalibration

A recalibration should create a new artifact, not mutate an existing one.

```text
old address artifact
    superseded_by → new address artifact

new calibration
    calibrated_from → new dataset manifest

validation report
    validates → new artifact
```

A policy artifact may remain unchanged while its observation-address artifact evolves.

---

# 17. Security and safety

## 17.1 Threats

- adversarial perturbation;
- replayed image;
- camera substitution;
- display spoofing;
- prototype poisoning;
- mislabeled examples;
- encoder-weight substitution;
- crafted near-prototype images;
- hidden critical object;
- stale frame;
- unexpected crop;
- model supply-chain compromise.

## 17.2 Required controls

- weight and preprocessing digests;
- artifact-policy binding;
- source identity;
- timestamp freshness;
- replay detection where needed;
- signed deployment manifests for critical systems;
- held-out corruption tests;
- explicit OOD set;
- critical-evidence checks;
- sensor fusion for physical control;
- rate-limited or approval-gated actions;
- complete rejection trace.

## 17.3 Safety boundary

For physical infrastructure:

> **Visual embedding similarity must not be treated as sufficient proof of safety.**

The system should combine visual evidence with:

- physical occupancy sensors;
- mechanism state;
- interlocks;
- emergency stop;
- previous transition state;
- operator approval where required.

---

# 18. First experiment: Arcade Visual Address Benchmark

The current arcade environment remains the best calibration fixture because it provides complete symbolic ground truth.

## 18.1 Dataset

For each of 112 policy rows, generate at least 100 visual variants:

```text
112 × 100 = 11,200 observations
```

Variation families:

- clean renderer variants;
- brightness and contrast;
- colour palette;
- sprite shape;
- one- and two-pixel translation;
- scale;
- background texture;
- noise;
- JPEG-style degradation;
- random masks;
- structured non-critical masks;
- structured critical masks;
- partial crop.

## 18.2 Split design

Do not use a simple random image split.

Hold out complete variation families.

Example:

```text
train/calibrate:
    red and blue sprites
    brightness changes
    random masks

held-out validation:
    green sprites
    new target shape
    structured occlusion
```

This measures generalization rather than memorization.

## 18.3 Systems compared

### System A

Current deterministic visual reader.

### System B

Enhanced deterministic descriptor with per-row calibration.

### System C

Frozen encoder global prototype.

### System D

Frozen encoder global retrieval plus local patch verification.

### System E

Frozen encoder plus linear policy-aware projection.

### System F

Optional NetVLAD-style aggregation plus projection.

## 18.4 Metrics

- exact row retrieval;
- action-equivalent retrieval;
- top-k row retrieval;
- acceptance rate;
- false acceptance;
- false rejection;
- conflicting-action error;
- OOD rejection;
- critical-mask rejection;
- non-critical-mask stability;
- nearest/second margin;
- calibration error;
- runtime;
- memory;
- artifact size;
- prototype count;
- cross-version reproducibility.

## 18.5 Required corruption curves

Evaluate at:

```text
0%
10%
25%
40%
50%
60%
75%
```

For each corruption level, report:

- exact row accuracy;
- action accuracy;
- false acceptance;
- rejection;
- critical versus non-critical masking.

## 18.6 Success criterion

The system must demonstrate both:

1. correct action under substantial **non-critical** corruption;
2. rejection under removal of **critical** evidence.

A high average accuracy that accepts critically corrupted images is a failure.

## 18.7 Kill conditions

Terminate or radically narrow the embedding direction if:

1. a direct classifier matches or exceeds action accuracy, rejection and inspectability while the VPM layer adds no policy-governance value;
2. local evidence cannot prevent unsafe acceptance under critical masking;
3. calibration does not transfer to held-out corruption families;
4. prototypes require nearly one exemplar per observation, reducing the system to a large memory lookup;
5. the frozen encoder’s license, access, runtime or hardware makes the claimed user experience unrealistic;
6. the deterministic path outperforms the learned path for the target domain.

---

# 19. Role of Bertin and matrix pattern analysis

Bertin-style ordering is not the runtime matcher.

The runtime uses high-dimensional representations.

Bertin analysis can make the visual-policy geometry inspectable.

```text
prototype similarity matrix
    ↓
pattern detector / seriation
    ↓
ordered VPM inspection view
```

Questions it may expose:

- Do same-action regions form coherent blocks?
- Are conflicting actions visually entangled?
- Is there a smooth progression between operational states?
- Which rows form the closest conflicting pair?
- Which prototypes are isolated?
- Does a corruption family form its own unwanted cluster?
- Does adaptation improve policy separation?
- Are there visually similar states with incompatible actions?

A discovered ordering should be frozen into a linked view artifact, not treated as proof of semantic correctness.

---

# 20. Ten practical application classes

These are examples of where the architecture might be useful. They are not all validated use cases.

| # | Application | Likely starting level | Addressed policy |
|---|---|---|---|
| 1 | **Known web workflow state recognition** | DOM/deterministic, visual fallback | Which workflow step and action is valid |
| 2 | **Legacy or remote-desktop automation** | Frozen screenshot embeddings | Which known screen or dialog is visible |
| 3 | **Kiosk and check-in flows** | Deterministic signs | Continue, request input, reject unexpected screen |
| 4 | **Industrial control-panel monitoring** | Deterministic or frozen embedding | Normal, warning, alarm, safe-stop state |
| 5 | **Parking-bay or loading-zone occupancy** | Frozen embedding + local evidence | Clear, occupied, obstructed, uncertain |
| 6 | **Bridge or crossing operational state** | Temporal embedding + physical sensors | Clear, approaching, occupied, exiting, blocked |
| 7 | **Production-line station state** | Deterministic or frozen embedding | Ready, processing, jammed, missing part |
| 8 | **Warehouse aisle or robot-zone supervision** | Temporal embedding | Clear, human present, obstruction, uncertain |
| 9 | **Game menus, tutorials and bounded arenas** | Hybrid deterministic/embedding | Menu state, tactical region, action primitive |
| 10 | **Visual regression and deployment verification** | Screenshot/DOM embeddings | Expected state, changed state, release-blocking anomaly |

## 20.1 Why web workflows are illustrative but not the whole project

A browser often exposes a DOM or accessibility tree, so visual embeddings should not replace exact structure unnecessarily.

The web example demonstrates the broader architecture:

```text
observe
    → address
    → decide
    → act
    → verify
```

But the visual-policy contribution is most distinctive where exact symbolic structure is absent, unreliable or inaccessible.

## 20.2 Why Quake is not an immediate claim

A complete Quake-playing system additionally requires:

- screen localization;
- video understanding;
- memory;
- policy learning;
- continuous controller output;
- feedback;
- exploration;
- map and opponent adaptation.

Visual addressing is one component, not the complete agent.

---

# 21. Claims ladder

## Claim V1 — implemented

> ZeroModel can compile a deterministic visual index for the committed closed arcade world and recover exact quantized feature codewords, policy rows and actions from canonical observations.

## Claim V2 — implemented

> The deterministic reader can reject declared distant and ambiguous fixtures while preserving complete artifact and decision evidence.

## Claim V3 — not yet supported

> Deterministic visual addressing tolerates meaningful non-exact visual variation.

Requires perturbation benchmarks.

## Claim V4 — next experiment

> A frozen pretrained visual encoder can address bounded policy regions from held-out visual variants without user training.

Requires the arcade embedding benchmark.

## Claim V5 — not yet supported

> Local and global evidence can preserve correct action under substantial non-critical occlusion while rejecting critical occlusion.

Requires explicit intervention tests.

## Claim V6 — not yet supported

> Policy-aware metric fitting improves action separation over frozen generic embeddings.

Requires a baseline-controlled held-out experiment.

## Claim V7 — not supported

> ZeroModel provides general image understanding or universal visual control.

## Claim V8 — not supported

> An arbitrary image with half of its pixels removed remains understandable.

## Claim V9 — not supported

> A visual-only system is sufficient for safety-critical infrastructure control.

---

# 22. Recommended implementation sequence

## PR A — experiment contracts

No heavy model dependency yet.

Add:

- dataset manifest schema;
- corruption specification;
- benchmark result schema;
- address-provider protocol;
- common visual-address decision;
- baseline evaluation harness;
- claims-audit entry.

## PR B — frozen embedding baseline

Add:

- optional vision extra;
- encoder manifest;
- frozen encoder adapter;
- global prototype artifact;
- cosine matcher;
- held-out evaluation;
- exact policy binding;
- no user training.

## PR C — local evidence

Add:

- patch-token contract;
- local prototype store;
- local agreement metric;
- critical-mask evaluation;
- `insufficient_decision_evidence` rejection.

## PR D — automatic lightweight adaptation

Add only after frozen baselines:

- linear projection;
- policy-aware objective;
- training manifest;
- weights artifact;
- comparison report;
- selection gate.

## PR E — temporal observation

Add:

- clip contract;
- previous-state evidence;
- temporal transition validation;
- a motion-dependent fixture.

---

# 23. Important corrections to supplied proposals

The supplied design drafts contain valuable ideas, but they should not be implemented verbatim.

## 23.1 Do not call model inference “zero inference”

A frozen encoder still performs model inference at runtime.

The valid claim is:

> No reasoning model or policy model is invoked after visual encoding.

## 23.2 Do not assume DINOv3 is a simple dependency

The official DINOv3 repository provides strong dense features, but its weights and custom licence require review, and its training stack is substantial.

The architecture must support alternative encoders.

## 23.3 Do not store large prototype banks in arbitrary metadata

Embeddings, patch banks and calibration tables should be structured artifact values or referenced content-addressed resources.

Metadata should describe the schema and identities.

## 23.4 Do not average all patch grids blindly

Spatially corresponding patch means may blur multiple poses and transformations.

Use:

- medoids;
- clustering;
- spatially aware matching;
- key patch selection;
- or learned aggregation.

## 23.5 Do not set target metrics before measuring task difficulty

Figures such as “95% at 50% occlusion” are hypotheses, not justified universal thresholds.

The benchmark should publish full operating curves.

## 23.6 Do not treat attention as proof of evidence

Attention maps are not automatically calibrated explanations or critical-region detectors.

Critical evidence requires intervention-based validation.

## 23.7 Keep policy and address artifacts separate

Embedding prototypes should not be injected into the policy artifact’s metadata.

The separate-artifact architecture remains the stronger design.

---

# 24. External-review instructions

Please evaluate the architecture as a proposed ZeroModel direction, not as an already completed product.

Distinguish findings as:

```text
blocker
major
minor
optional enhancement
research question
```

For each finding, provide:

```text
Severity:
Area:
Finding:
Why it matters:
Smallest correction:
Required experiment or test:
Claims impact:
```

Do not reward the design for being ambitious. Look for ways it could become an unnecessary wrapper around standard embedding retrieval.

---

# 25. Questions for reviewers

## Architecture

1. Is the shared `VisualAddressProvider` abstraction the correct seam?
2. Should the address decision and policy decision remain separate?
3. Is `addresses` the correct policy relationship?
4. Should encoder, projection, prototype and calibration records be separate artifacts?
5. Is a VPM artifact an appropriate container for dense embedding prototypes?
6. Should embedding storage use an external content-addressed matrix referenced by the VPM?
7. Are exact rows and action-equivalent regions being distinguished clearly enough?
8. Is the architecture actually ZeroModel-specific, or merely nearest-prototype classification with provenance?

## Product ladder

9. Is “least intelligent mechanism that passes validation” a sound product principle?
10. Is the deterministic path worth preserving?
11. Can frozen embeddings genuinely deliver a no-training user experience?
12. What should cause `auto` mode to move from deterministic to frozen?
13. What should cause it to move from frozen to adapted?
14. Should automatic adaptation be in ZeroModel core or an optional research package?
15. Is expert-supplied custom perception too broad for the project?

## Representation

16. Which frozen encoders should be benchmarked first?
17. Is a global class token sufficient for the first baseline?
18. Should mean patch pooling precede NetVLAD?
19. Are medoids preferable to centroids?
20. How many prototypes per row should be permitted?
21. Should prototypes represent exact rows, action regions or both?
22. What representation is most robust to benign partial occlusion?
23. Does policy-aware metric learning risk hiding visually meaningful distinctions?

## Calibration

24. How should per-prototype acceptance thresholds be estimated?
25. Should calibration optimize false acceptance globally or per conflicting action?
26. Is nearest/second-nearest margin sufficient?
27. Should OOD rejection include an explicit background or unknown distribution?
28. How should uncertainty be combined across global and local evidence?
29. Should threshold selection be conformal, percentile-based, EVT-based or simpler?
30. How should calibration validity be scoped to a camera, renderer or source?
31. What minimum validation set is needed before emitting a deployable artifact?

## Occlusion

32. How should the system distinguish non-critical from critical masking?
33. Is patch matching enough when objects move spatially?
34. Should local matching preserve geometry?
35. What intervention test best identifies decision-critical patches?
36. How should the reader react when the global match is strong but local evidence is missing?
37. Is “insufficient decision evidence” a separate rejection class from OOD?
38. Should recent frames be used to infer missing current evidence?

## Temporal systems

39. At what point should the architecture move from images to video?
40. Is previous accepted VPM state sufficient as lightweight memory?
41. Should temporal transition rules be inside the policy artifact or a separate transition artifact?
42. Which simple motion-dependent fixture should precede a bridge example?
43. Is V-JEPA-style encoding relevant, or would stacked image embeddings suffice?

## Safety and governance

44. Are artifact digests enough, or are signatures required?
45. How should encoder licence and weight access be recorded?
46. What attacks are most likely against prototype addressing?
47. Should safety-critical actions require independent sensor agreement?
48. How should human overrides become calibration evidence?
49. What drift signals should automatically disable an address artifact?
50. What data or traces must be redacted?

## Evidence and novelty

51. What baseline would most threaten the need for this architecture?
52. If a linear classifier over frozen embeddings works, what does VPM add?
53. What inspection study would demonstrate value beyond accuracy?
54. Can Bertin ordering reveal policy conflicts that ordinary confusion matrices miss?
55. What result would justify calling the learned space a “Visual Policy Map”?
56. What result should cause the project to stop using that term?
57. Is the arcade benchmark too synthetic to decide the architecture?
58. Which fixed-camera domain is the cheapest honest next test?
59. What is the smallest publishable claim?
60. What are the three strongest kill conditions?

---

# 26. Requested reviewer response

## A. Overall verdict

Choose one:

```text
strong architecture
promising but overbuilt
useful deterministic and frozen-prototype framework
standard retrieval wrapped in unnecessary artifacts
wrong direction for ZeroModel
```

Explain the verdict.

## B. Severity-ranked findings

Use the required finding format.

## C. Architecture simplification

Describe the smallest system that preserves the real value.

## D. Baselines

Name the mandatory conventional baselines.

## E. Next experiment

Specify one bounded experiment:

- dataset;
- models;
- metrics;
- success criterion;
- kill conditions;
- exact claim enabled by success.

## F. Recommended next PR

Specify:

- files;
- API surface;
- dependencies;
- tests;
- documentation;
- claims-audit change;
- explicit non-goals.

---

# 27. Reference research

These works motivate candidate components; none alone validates the ZeroModel architecture.

1. **DINOv2: Learning Robust Visual Features without Supervision**  
   https://arxiv.org/abs/2304.07193

2. **DINOv3**  
   https://arxiv.org/abs/2508.10104  
   Official implementation: https://github.com/facebookresearch/dinov3

3. **NetVLAD: CNN Architecture for Weakly Supervised Place Recognition**  
   https://openaccess.thecvf.com/content_cvpr_2016/html/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.html

4. **Unifying Deep Local and Global Features for Image Search (DELG)**  
   https://arxiv.org/abs/2001.05027

5. **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)**  
   https://arxiv.org/abs/2301.08243

6. **Masked Autoencoders Are Scalable Vision Learners**  
   https://arxiv.org/abs/2111.06377

7. **V-JEPA official implementation**  
   https://github.com/facebookresearch/jepa

---

# 28. Final design position

The visual architecture should not force every user into model training.

It should provide a validated progression:

```text
deterministic codebook
    ↓ when evidence requires
deterministic tolerant descriptor
    ↓ when evidence requires
frozen pretrained embedding
    ↓ when evidence requires
automatic lightweight adaptation
    ↓ when evidence requires
expert custom perception
```

All levels should terminate in the same governed interface:

```text
observation
    ↓
identity-bearing visual address
    ↓
accepted VPM row or region
    ↓
independently identified policy
    ↓
action or rejection
```

The current deterministic reader remains valuable as the cheapest and most auditable level.

The next serious question is not whether embeddings can retrieve similar images. That is established.

The question is:

> **Can ZeroModel turn a visual representation into a calibrated, policy-aware and independently governed address that remains stable under benign variation, rejects missing critical evidence, and provides operational value beyond a conventional classifier?**

That is the experiment the next stage must answer.
