# ZeroModel 1.0.12: Temporal Visual Evidence and the Video Policy Reader

You are beginning the next major ZeroModel research and implementation stage.

The target release is:

```text
ZeroModel 1.0.12
```

The repository may still identify itself as `1.0.11` when this chat begins. Treat `1.0.12` as the target release. Do not change the package version until the release definition of done has actually been satisfied.

This chat is responsible for designing, implementing, measuring and documenting the first genuine ZeroModel **video policy-reading path**.

This must be actual temporal visual processing, not merely:

* an animated VPM;
* a GIF made from independent images;
* a loop that applies a still-image classifier without temporal state;
* a cached previous answer presented as video understanding;
* a demo that has no frozen benchmark, rejection contract or evidence record.

The goal is to determine whether local and temporal visual evidence can turn a sequence of world-produced frames into a governed exact policy address.

A valid negative result remains an acceptable research outcome.

---

# 1. Primary mission

Build a bounded video front end that can:

1. read an actual video file or deterministic frame stream;
2. preserve frame identity, timestamps and source provenance;
3. generate local evidence for each frame;
4. associate evidence across time;
5. detect consistent and inconsistent state transitions;
6. retain exact policy-row identity;
7. expose same-action wrong-row ambiguity;
8. reject unsupported, contradictory or out-of-domain observations;
9. delegate accepted exact rows to the existing deterministic VPM policy;
10. produce a complete reconstructable temporal trace.

The architectural question is:

> Can structured local evidence plus temporal consistency produce useful governed visual-policy coverage where whole-image similarity failed?

The release question is narrower:

> Can ZeroModel 1.0.12 provide a real, deterministic, testable and evidence-producing video policy-reader interface, even if the first approximate temporal provider ends in Outcome B or C?

---

# 2. Authoritative project links

## Repository and documentation

* Repository:
  https://github.com/ernanhughes/zeromodel

* Documentation site:
  https://ernanhughes.github.io/zeromodel/

* Repository README:
  https://github.com/ernanhughes/zeromodel/blob/main/README.md

* Claims audit:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/claims-audit.md

* Release documentation:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/release.md

## Research memory

* Visual research logbook:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/research/visual-research-logbook.md

* Visual programme status cut:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/research/visual-programme-status-cut-2026-07-18.md

* Visual representation identity ADR:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/adr/visual-representation-identity.md

* Two-week source audit, if present:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/research/visual-logbook-source-audit-2026-07-04-to-2026-07-18.md

## Canonical visual research documents

* Exact visual sign reader:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/research/visual-sign-reader.md

* Visual-address Phase Zero:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/research/visual-address-phase-zero.md

* Visual-address Phase One:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/research/visual-address-phase-one.md

* Visual-address research status:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/research/visual-address-research-status.md

* Phase One review adjudication:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/research/visual-address-review-adjudication.md

* System B v2 adjudication:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/research/visual-address-system-b-v2-adjudication.md

* Registered local-baseline showdown:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/research/visual-local-baseline-showdown.md

* Visual-AI status after registration:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/research/visual-ai-research-status-after-registration.md

* Fixed-camera benchmark direction:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/research/fixed-camera-status-panel-benchmark.md

* Governance-parity direction:
  https://github.com/ernanhughes/zeromodel/blob/main/docs/research/visual-governance-parity.md

## Frozen evidence

* Phase One global benchmark:
  https://github.com/ernanhughes/zeromodel/tree/main/docs/results/visual-address-phase-one-v1

* System B v2:
  https://github.com/ernanhughes/zeromodel/tree/main/docs/results/visual-address-system-b-v2

* Registered local baseline R1:
  https://github.com/ernanhughes/zeromodel/tree/main/docs/results/visual-local-baseline-showdown-v1

* R1 independent post-analysis:
  https://github.com/ernanhughes/zeromodel/tree/main/docs/results/visual-local-baseline-showdown-v1-postanalysis

## Important pull requests

* Exact visual sign reader, PR #27:
  https://github.com/ernanhughes/zeromodel/pull/27

* System B calibration and evidence repair, PR #32:
  https://github.com/ernanhughes/zeromodel/pull/32

* Registered-pixel baseline, PR #34:
  https://github.com/ernanhughes/zeromodel/pull/34

---

# 3. Public website and articles

## Main website

* Programmer.ie:
  https://programmer.ie/

* ZeroModel article index:
  https://programmer.ie/tags/zeromodel/

## Foundational ZeroModel article

* **ZeroModel: Visual AI You Can Scrutinize**
  https://programmer.ie/post/zeromodel/

This is historically important but contains broad early claims. The current repository claims audit is authoritative when public prose and measured repository evidence differ.

## Related conceptual articles

* **A Complete Visual Reasoning Stack: From Conversations to Epistemic Fields**
  https://programmer.ie/post/visual_reasoning_stack/

* **Phōs: Visualizing How AI Learns and How to Build It Yourself**
  https://programmer.ie/post/phos/

* **The Space Between Models Has Holes: Mapping the AI Gap**
  https://programmer.ie/post/gap/

* **Search–Solve–Prove: Building a Place for Thoughts to Develop**
  https://programmer.ie/post/ssp/

* **Intelligence Through Execution: The Executable Cognitive Kernel**
  https://programmer.ie/post/eck/

* **A Memory Gate for AI: Policy-Bounded Acceptance in the Executable Cognitive Kernel**
  https://programmer.ie/post/verify/

These articles provide the wider intellectual background:

* visual representations;
* policy-bounded execution;
* verification before commitment;
* visible evidence;
* deterministic policy artifacts;
* separation of stochastic inference from governed action.

Do not treat earlier public claims as stronger evidence than current tests, reports and the claims audit.

Do not invent a URL for Blog 2 if it has not yet been published. Inspect the repository for its draft or enhancement plan instead.

---

# 4. Source hierarchy

When sources disagree, use this order:

```text
1. Current code and tests on main
2. Claims audit
3. Frozen evidence bundles and their manifests
4. Research adjudications
5. Research logbook and current status cut
6. Specifications and design notes
7. Pull-request discussions
8. Public blog posts
9. Chat summaries
```

A public claim does not override a measured negative result.

A source file does not establish that an experiment was run.

A passing unit test does not establish empirical usefulness.

A committed result directory does not establish validity unless its identities and bundle manifest verify.

---

# 5. Required startup procedure

Begin from the latest remote `main`.

Run:

```bash
git fetch origin
git switch main
git pull --ff-only
git status --short
git rev-parse HEAD
git log -n 20 --oneline --decorate
python -m pytest -q
```

Then inspect:

```text
pyproject.toml
README.md
docs/claims-audit.md
docs/research/visual-research-logbook.md
docs/research/visual-programme-status-cut-2026-07-18.md
docs/adr/visual-representation-identity.md
docs/results/visual-local-baseline-showdown-v1/
docs/results/visual-local-baseline-showdown-v1-postanalysis/
examples/arcade_visual_local_evidence_benchmark.py
examples/arcade_visual_registered_calibration_v2.py
zeromodel/visual_registration.py
zeromodel/visual_local_baselines.py
zeromodel/visual_address.py
zeromodel/visual_experiment.py
```

Discover the actual present repository structure. Do not assume all suggested file names remain unchanged.

Before implementing anything, produce a concise inherited-state report containing:

```text
Current main SHA
Current package version
Current test result
Measured systems
Frozen evidence identities
Confirmed mechanisms
Retired approaches
Preparation-only code
Unresolved Stage 2 question
Proposed 1.0.12 implementation boundary
```

Do not ask the user to restate the project history already recorded in the repository.

---

# 6. Inherited scientific position

Treat the following as inherited unless current repository evidence has superseded it.

## 6.1 The policy core is stable

ZeroModel already has a bounded deterministic policy core:

* immutable VPM artifacts;
* exact row and metric identities;
* deterministic policy lookup;
* policy diagnostics;
* verification artifacts;
* repair lineage;
* provider-neutral visual addressing;
* Lua policy consumption;
* evidence bundles;
* a five-state claims ladder.

The video work belongs in front of this core.

Do not redesign the policy artifact merely because observation reading remains uncertain.

## 6.2 Made images and found images are different

A VPM is a **made image**:

* authored by ZeroModel;
* intentionally arranged;
* coordinate-addressable;
* deterministic;
* already part of the policy representation.

A camera or rendered observation is a **found image**:

* produced by the world;
* translated, occluded or corrupted;
* temporally incomplete;
* epistemically uncertain;
* evidence about state rather than state itself.

The video reader must not treat found images as though they were authored VPM coordinates.

The correct relationship is:

```text
found video
    → bounded evidence
    → candidate state or claim set
    → governed acceptance
    → exact VPM policy lookup
```

## 6.3 Exact row is not the same as correct action

Previous systems often selected the correct action from the wrong policy row.

Preserve separately:

```text
exact-row correctness
action correctness
```

A correct action does not establish that the system understood the exact state.

Wrong-row same-action results remain errors because:

* row identity carries provenance;
* visually similar rows may later require different actions;
* present action equivalence does not prove state equivalence;
* transition validity depends on exact state;
* conflicting-action neighbours must remain visible.

## 6.4 Ranking is not acceptance

System B and R1 contained strong pre-rejection ranking information while failing to produce useful governed accepted coverage.

A video reader must therefore expose at least:

```text
best row
best same-action competing row
best conflicting-action competing row
raw action
raw exact-row confidence
accepted row or rejection
rejection reasons
```

Never hide raw prediction behind rejection.

Never present raw ranking accuracy as governed accuracy.

## 6.5 Global DINOv2 CLS retrieval was not promoted

The measured global DINOv2 systems did not improve governed exact policy addressing enough to justify promotion.

This is a bounded negative result for:

* the committed fixture;
* global CLS representation;
* recorded preprocessing;
* recorded calibration;
* measured operating conditions.

It does not establish that every learned local or temporal representation will fail.

## 6.6 Registration confirmed a mechanism

Bounded deterministic registration improved:

```text
exact-row top-1:
75% → 87.5%

action top-1:
96.875% → 98.4375%
```

It completely repaired the declared held-out two-pixel translation family at raw top-1:

```text
224 / 336 → 336 / 336
```

Those shifts were unseen instances inside the declared search envelope.

This did not establish arbitrary translation invariance.

## 6.7 Governed acceptance remained unresolved

R1 still accepted:

```text
0 / 1,344 benign final observations
```

at its selected zero-observed-distinguishable-false-accept operating point.

The committed independent threshold–margin post-analysis found that no useful operating point had been hidden by the original coupled-quantile search.

Therefore Stage 2 should not reopen that frozen conclusion without new evidence.

## 6.8 Fresh v3 work is preparatory

The repository contains a fresh local-evidence benchmark fixture and independent registered-calibration path.

The fixture includes or may include:

* held-out translation;
* translation plus photometric changes;
* local occlusion;
* critical-region corruption;
* same-action wrong-row controls;
* conflicting-action near-neighbours;
* compositional invalid states;
* information-theoretic controls;
* beyond-bounds translations.

Code and tests do not imply that the fixture has already produced a verified evidence package.

Inspect its current status before using it.

---

# 7. What “video working” means

ZeroModel 1.0.12 should support a real bounded video path with three layers.

## Layer A: video transport

The system can read:

* an actual video file;
* a deterministic lossless clip;
* or an iterable frame stream.

It preserves:

* source identity;
* frame index;
* presentation timestamp;
* decoding order;
* dimensions;
* channel format;
* frame digest;
* clip digest;
* declared frame rate;
* decode warnings.

## Layer B: temporal visual evidence

The system generates:

* frame-local evidence;
* local-region matches;
* region motion;
* temporal identity;
* state candidates;
* transition evidence;
* contradictions;
* out-of-domain signals.

## Layer C: governed policy execution

Only an accepted exact policy row may reach deterministic policy lookup.

The final decision must retain:

* video source identity;
* frame identity;
* evidence identity;
* selected row;
* policy artifact identity;
* action cell;
* acceptance or rejection status;
* temporal lineage.

A simple loop that reads each frame independently satisfies only Layer A.

Version 1.0.12 must implement at least a bounded, measured path through all three layers.

---

# 8. Proposed architecture

Use this as the conceptual target, adjusting names to current repository conventions:

```text
Video file or frame stream
    ↓
VideoFrameSource
    ↓
Frame normalization and identity
    ↓
Frame-local evidence provider
    ↓
Local region support and contradiction
    ↓
Temporal association
    ↓
Transition-consistent candidate state set
    ↓
Temporal acceptance gates
    ↓
Accepted exact policy row
    ↓
Existing VPMPolicyLookup
    ↓
Action plus complete video evidence trace
```

The policy core remains independently identified.

The video reader must not embed policy action choice inside an opaque temporal model.

---

# 9. Suggested public interfaces

Discover and reuse current repository contracts first.

Potential types include:

```python
@dataclass(frozen=True)
class VideoFrame:
    clip_id: str
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    pixels: np.ndarray
    source_digest: str
    frame_digest: str


@dataclass(frozen=True)
class VideoClipManifest:
    version: str
    clip_id: str
    source_kind: str
    frame_count: int
    width: int
    height: int
    channels: int
    nominal_fps: float | None
    frame_digests: tuple[str, ...]
    payload_digest: str


class VideoFrameSource(Protocol):
    def manifest(self) -> VideoClipManifest:
        ...

    def frames(self) -> Iterable[VideoFrame]:
        ...


@dataclass(frozen=True)
class LocalRegionEvidence:
    region_id: str
    expected_box: tuple[int, int, int, int]
    matched_box: tuple[int, int, int, int] | None
    score: float | None
    dx: int | None
    dy: int | None
    overlap_fraction: float
    critical: bool
    status: str


@dataclass(frozen=True)
class TemporalEvidence:
    frame_id: str
    row_id: str
    local_support: float
    contradiction_score: float
    spatial_consistency: float
    transition_consistency: float
    temporal_persistence: float
    conflicting_action_gap: float
    evidence_window: tuple[str, ...]


@dataclass(frozen=True)
class VideoAddressDecision:
    clip_id: str
    frame_id: str
    raw_row_id: str | None
    raw_action_id: str | None
    accepted_row_id: str | None
    accepted_action_id: str | None
    accepted: bool
    rejection_reasons: tuple[str, ...]
    policy_artifact_id: str
    provider_id: str
    evidence_id: str
```

Do not create redundant types when existing contracts can be extended safely.

Keep arrays immutable or defensively copied according to existing repository ownership rules.

---

# 10. Core ingestion versus optional adapters

Keep the core dependency-light.

The core video API should accept:

```text
Iterable[np.ndarray]
Iterable[VideoFrame]
lossless frame directory
deterministic NPZ or repository-native clip bundle
```

Add a separate optional video dependency group only when necessary.

Inspect whether the best minimal adapter is:

* OpenCV;
* PyAV;
* ImageIO plus FFmpeg;
* another existing repository-compatible library.

Do not add multiple video libraries without a measured reason.

Suggested optional dependency shape:

```toml
video = [
    "...minimal selected adapter..."
]
```

The canonical research dataset should not depend on lossy codec bytes being bit-identical across machines.

Use:

```text
lossless frame payloads
+
deterministic frame manifests
```

as canonical benchmark truth.

An MP4, WebM or GIF may be included as a human-viewable derivative, but its codec output should not become the canonical dataset identity unless deterministic encoding has been demonstrated.

---

# 11. Minimum working 1.0.12 baseline

Before approximate temporal inference, implement a deterministic positive baseline.

Create a synthetic arcade clip from canonical frames for a known state sequence.

The video path should:

1. decode or iterate every frame;
2. recover each canonical exact row through the existing exact reader;
3. delegate to the same VPM policy;
4. reproduce the symbolic action sequence;
5. emit a complete temporal trace;
6. reconstruct the sequence from serialized evidence;
7. fail on reordered or tampered frames when the contract requires ordered replay.

This proves that:

> ZeroModel can consume a bounded video sequence and preserve exact observation-to-policy lineage.

It does not prove tolerance to approximate world observations.

The exact canonical video baseline and the approximate temporal provider must remain separate systems.

---

# 12. Stage 2 frame-local evidence

The first approximate provider should remain deterministic.

Start with:

```text
translation-equivariant local template correlation
```

rather than another global embedding.

For each candidate policy row, produce multiple independently scored regions.

Potential regions in the arcade fixture include:

* target band;
* tank band;
* cooldown indicator;
* central playfield;
* action-critical local patch;
* stable background anchors.

For every region record:

```text
region identity
expected location
matched location
score
valid overlap
displacement
critical status
support or contradiction
competing row support
```

Use normalized or zero-mean normalized correlation where appropriate.

Support:

* valid-pixel masks;
* per-region weights;
* native resolution;
* bounded integer displacement;
* no wrapping;
* no interpolation in the first deterministic system;
* deterministic tie-breaking.

Do not disguise one whole-image comparison as a collection of local evidence.

---

# 13. Required semantic caution

Never serialize:

```text
evidence absent
```

merely because a region was not matched in one frame.

A single frame cannot distinguish:

* true absence;
* occlusion;
* corruption;
* low contrast;
* displacement beyond the search region;
* an unseen appearance mode;
* decode damage.

Use wording such as:

```text
expected region unmatched against known modes
```

Temporal evidence may later strengthen an absence hypothesis, but it must not silently convert an unmatched observation into certainty.

---

# 14. Temporal association

Video adds information unavailable in a still frame.

Use that information explicitly.

The first temporal layer should calculate at least:

```text
frame-to-frame region displacement
track persistence
row-candidate persistence
action-candidate persistence
transition validity
candidate-set contraction or expansion
temporary unmatched regions
reappearance
identity switches
motion outside declared bounds
```

Do not begin with an opaque sequence transformer.

Start with a deterministic temporal evidence layer whose decisions can be reconstructed.

A suitable first structure may be:

```text
window of N frame-local evidence records
    ↓
candidate rows per frame
    ↓
declared state-transition graph
    ↓
temporally consistent candidate paths
    ↓
support and contradiction per path
    ↓
accepted exact row or rejection
```

---

# 15. Transition specification

Create a versioned transition contract for the bounded arcade policy.

Potential representation:

```python
@dataclass(frozen=True)
class PolicyTransitionSpec:
    version: str
    allowed_row_transitions: Mapping[str, tuple[str, ...]]
    maximum_frame_gap: int
    maximum_position_delta: int
    action_conditioned: bool
```

The transition specification must derive from declared environment dynamics, not from final-evaluation observations.

Track separately:

```text
possible transition
impossible transition
unknown because of dropped frames
unknown because the observation was rejected
```

Do not treat an unobserved intermediate frame as proof of an impossible transition.

---

# 16. No silent carry-forward

The system must not convert:

```text
I could not read this frame
```

into:

```text
the previous state is still true
```

without an explicit bounded persistence rule.

Any carried state must record:

* source frame;
* elapsed frames;
* elapsed time;
* allowed persistence horizon;
* transition assumptions;
* confidence degradation;
* whether the current frame independently supports it.

When the horizon is exceeded, reject.

---

# 17. Candidate-state representation

Avoid forcing every uncertain frame into one row too early.

A temporal frame may produce:

```text
one exact supported row
several compatible rows
several same-action rows
conflicting-action candidates
no supported rows
```

Represent this explicitly.

Potential structure:

```python
@dataclass(frozen=True)
class TemporalStateClaimSet:
    frame_id: str
    candidate_row_ids: tuple[str, ...]
    compatible_action_ids: tuple[str, ...]
    conflicting_action_ids: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    contradictory_evidence_ids: tuple[str, ...]
    status: str
```

Default governed execution should require an accepted exact row.

A research-only same-action consensus result may be reported, but must never be described as exact-state understanding.

---

# 18. Acceptance gates

Do not collapse every criterion into one confidence number.

Calibrate separate gates for:

```text
minimum local support
minimum critical-region match coverage
maximum local contradiction
maximum displacement disagreement
minimum conflicting-action gap
minimum temporal persistence
minimum transition consistency
maximum identity-switch rate
maximum unmatched duration
declared motion envelope
```

Independent gates must not be searched only along one shared quantile curve.

Cache evidence once, then evaluate candidate gate combinations over cached calibration evidence.

Record why each frame or sequence was rejected.

Suggested rejection reasons:

```text
decode_error
frame_shape_invalid
frame_unreadable
insufficient_local_support
expected_region_unmatched
critical_region_unmatched
spatially_inconsistent_matches
conflicting_action_evidence
same_action_row_ambiguity
transition_impossible
transition_unknown_due_to_gap
temporal_identity_unstable
persistence_horizon_exceeded
outside_declared_motion_bound
outside_declared_visual_domain
calibration_gate_failed
```

---

# 19. Fresh temporal benchmark

Build video clips from the fresh v3 fixture rather than evaluating only isolated frames.

Do not tune using the frozen final split.

Use explicit:

```text
prototype
benign calibration
rejection calibration
final evaluation
```

roles.

Create temporally meaningful families.

## Positive or expected-accept families

* canonical state trajectories;
* bounded smooth movement;
* held-out within-envelope translations;
* camera jitter;
* brightness and contrast changes;
* mild compression;
* temporary local occlusion;
* temporary target disappearance with valid reappearance;
* dropped frames within a declared recoverable gap;
* variable frame-rate sampling;
* legal action-conditioned transitions.

## Expected-reject families

* beyond-bounds translation;
* impossible state jumps;
* conflicting-action near-neighbour sequences;
* same-action wrong-row sequences;
* compositional invalid states;
* duplicated critical object;
* reordered frames;
* stale repeated frames;
* timestamp reversal;
* excessive dropped-frame gap;
* identity swap;
* critical-region corruption;
* contradictory local motion;
* persistent unmatched critical region;
* decode or shape corruption.

## Information-theoretic controls

Preserve cases where essential evidence has actually been removed.

Do not count indistinguishable impossible controls as distinguishable false-accept opportunities.

---

# 20. Beyond-bounds motion control

The previous registration system used a bounded displacement envelope.

The new temporal benchmark must test motion outside that envelope.

For example:

```text
within bound:
±1, ±2 or declared local search range

outside bound:
±4, ±5 or values beyond the frozen provider range
```

The question is not whether the provider can magically recover unreachable motion.

The question is:

> Does the provider fail safely when the observation exceeds its declared motion model?

A safe result should reject or mark the state unresolved rather than mis-register confidently to a reachable but wrong row.

---

# 21. Comparison systems

Evaluate at least these systems on the same temporal fixture.

## V0: exact canonical video reader

Exact frame codewords plus deterministic temporal trace.

Purpose:

```text
positive transport and lineage baseline
```

## V1: independent-frame registered reader

Apply the strongest existing frame-only registered provider independently.

Purpose:

```text
measure what temporal evidence must improve over
```

## V2: local-correlation frame reader

Multiple local evidence regions, but no temporal filtering.

Purpose:

```text
isolate the value of locality
```

## V3: local-correlation temporal reader

Local evidence plus transition and persistence constraints.

Purpose:

```text
isolate the value of time
```

Do not compare a new temporal system only against historical headline numbers from a different fixture.

---

# 22. Primary Stage 2 research question

The decisive comparison is:

> Does V3 produce greater governed accepted exact-row coverage than V2 while preserving zero observed distinguishable false accepts and zero accepted conflicting-action errors on the declared fresh final clips?

Secondary questions:

1. Does temporal consistency reduce same-action wrong-row confusion?
2. Does it reject impossible transitions?
3. Does it recover after temporary occlusion?
4. Does it fail safely outside the motion envelope?
5. Does it reduce or increase time-to-decision?
6. Does temporal persistence introduce stale-state false acceptance?
7. Can every accepted frame be reconstructed from explicit evidence?

---

# 23. Metrics

Report frame-level and sequence-level metrics separately.

## Frame-level

```text
raw exact-row top-1 accuracy
raw action top-1 accuracy
accepted exact-row precision
accepted action precision
accepted coverage
false accepts
false rejects
accepted conflicting-action errors
same-action wrong-row errors
```

## Sequence-level

```text
complete-sequence exact-row correctness
complete-sequence action correctness
accepted sequence coverage
time to first stable acceptance
time to recover after occlusion
identity-switch count
row-switch count
action-switch count
impossible-transition accept count
stale-state carry-forward count
outside-bound false-accept count
```

## Temporal quality

```text
track persistence
unmatched duration
candidate-path count
candidate-set size
transition consistency
motion residual
temporal contradiction
```

## Runtime

```text
decode time
frame evidence time
temporal update time
policy lookup time
end-to-end latency
frames per second
peak memory where practical
```

## Statistical reporting

Report:

* observed rates;
* counts and denominators;
* Wilson intervals where applicable;
* risk–coverage curves;
* results by row, action, family and clip.

When no observations are accepted:

```text
precision = null
confidence interval = null
```

Do not serialize undefined precision as `0.0`.

Use:

> zero observed distinguishable false accepts on the declared fixture

Never use:

> zero false-accept risk

---

# 24. Outcome ladder

## Outcome A

```text
zero observed distinguishable false accepts
zero accepted conflicting-action errors
final benign exact-row coverage >= 50%
```

Interpretation:

```text
the bounded temporal local-evidence reader has a useful governed operating point
```

Next action:

```text
fixed-camera bounded validation
```

## Outcome B

```text
zero observed distinguishable false accepts
zero accepted conflicting-action errors
10% <= final benign exact-row coverage < 50%
```

Interpretation:

```text
temporal evidence adds bounded value but remains incomplete
```

Next action:

```text
deterministic geometry or stronger local evidence
```

## Outcome C

```text
no feasible transferred operating point
or final benign exact-row coverage < 10%
```

Interpretation:

```text
the measured temporal local-evidence architecture remains insufficient
```

Next action:

```text
deterministic geometry extraction before learned video systems
```

## Invalid

Use invalid when:

* final data entered selection;
* clip identities do not reproduce;
* source videos cannot be reconstructed;
* canonical frame manifests disagree;
* policy identities do not verify;
* impossible controls are misclassified;
* sequence metrics cannot be reconstructed;
* evidence files fail hashing;
* the code state is ambiguous;
* the benchmark depends on nondeterministic codec output;
* required negative families are missing.

A valid Outcome C is preferable to an invalid Outcome A.

---

# 25. Evidence package

Suggested directory:

```text
docs/results/visual-video-stage-two-v1/
```

Required artifacts should include, adapting to repository conventions:

```text
README.md
protocol.json
environment.json
argv.json
command.txt
runtime.json
run-manifest.json

video-dataset-manifest.json
clip-manifest.json
frame-manifest.json
transition-spec.json
policy-reference.json
frozen-comparator-references.json

local-template-spec.json
local-template-manifest.json
video-provider-config.json
candidate-grid.json
selected-calibration.json

frame-traces.jsonl
sequence-traces.jsonl
final-report.json
final-summary.json
adjudication.json

paired-v1-v2-comparison.json
paired-v2-v3-comparison.json
risk-coverage-curve.json

local-region-atlas.json
temporal-transition-atlas.json
identity-switch-atlas.json
conflicting-action-atlas.json
same-action-row-confusion-atlas.json
occlusion-recovery-atlas.json
outside-bound-atlas.json
residual-error-atlas.json

bundle-manifest.json
```

Human-viewable video derivatives may be included under:

```text
docs/assets/visual-video-stage-two-v1/
```

Do not place large derived videos into the canonical bundle when they make evidence verification impractical. Record their source identities and generation command.

---

# 26. Trace contract

Every frame trace must retain enough information to reconstruct the decision.

At minimum:

```text
clip ID
frame ID
frame index
timestamp
frame digest
expected disposition
expected row
expected action
family
transformation parameters

raw predicted row
raw predicted action
accepted row
accepted action
accepted or rejected
rejection reasons

local region evidence
best same-action competitor
best conflicting-action competitor
local support
contradiction
spatial consistency

previous accepted state
candidate transition
transition status
temporal persistence
unmatched duration
candidate path count

policy artifact ID
provider ID
calibration ID
evidence ID
```

Sequence traces must record:

```text
ordered frame IDs
accepted state sequence
raw state sequence
action sequence
identity switches
rejection spans
occlusion spans
recovery points
impossible transitions
sequence outcome
```

---

# 27. Test requirements

Create focused tests for all new contracts.

## Video transport

* deterministic lossless frame loading;
* real video adapter smoke test;
* frame ordering;
* timestamp monotonicity;
* duplicate timestamps;
* decode errors;
* frame shape changes;
* RGB/grayscale handling;
* source and frame digest stability;
* caller-memory ownership.

## Temporal contracts

* legal transition;
* impossible transition;
* dropped frame;
* duplicated frame;
* reordered frame;
* temporal gap;
* stale carry-forward prevention;
* persistence horizon;
* candidate path reconstruction;
* deterministic tie-breaking.

## Local evidence

* exact local match;
* bounded translation;
* outside-bound translation;
* masked pixels;
* weighted regions;
* critical region unmatched;
* contradictory region;
* spatially inconsistent regions;
* same-action competitor;
* conflicting-action competitor.

## Semantic safety

* unmatched region is not serialized as certain absence;
* rejected frame retains raw prediction;
* correct action with wrong row remains an exact-row error;
* temporal consensus does not silently become exact-row certainty;
* impossible transitions are rejected;
* uncertain gaps do not become state persistence automatically.

## Calibration isolation

Use spy providers to prove that no final clip or frame was:

* decoded for selection;
* correlated;
* tracked;
* summarized;
* used for gate selection;
* used for template extraction.

## Evidence integrity

Independently reconstruct:

* frame metrics;
* sequence metrics;
* false-accept counts;
* false-reject counts;
* conflicting-action accepts;
* family totals;
* identity-switch totals;
* all headline results.

Verify every bundle hash.

## Full repository

Run focused tests first, then:

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
```

---

# 28. Suggested implementation locations

Inspect current conventions before creating new modules.

Potential modules:

```text
zeromodel/video.py
zeromodel/video_source.py
zeromodel/video_address.py
zeromodel/visual_local_correlation.py
zeromodel/visual_local_evidence.py
zeromodel/temporal_evidence.py
zeromodel/policy_transitions.py
```

Potential examples:

```text
examples/arcade_visual_video_baseline.py
examples/arcade_visual_video_showdown.py
examples/render_arcade_video_fixture.py
```

Potential tests:

```text
tests/test_video_source.py
tests/test_video_address.py
tests/test_visual_local_correlation.py
tests/test_temporal_evidence.py
tests/test_policy_transitions.py
tests/test_arcade_visual_video_baseline.py
tests/test_arcade_visual_video_showdown.py
tests/test_visual_video_result_records.py
```

Avoid a single monolithic module.

Keep transport, visual evidence, temporal logic and policy execution separable.

---

# 29. Research papers: immediate deterministic foundation

Read these before freezing the Stage 2 local-evidence protocol.

## Normalized weighted cross-correlation

**Gastón A. Ayubi, Bartlomiej Kowalski and Alfredo Dubra — “Normalized Weighted Cross Correlation for Multi-Channel Image Registration”**

* Full text:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12448653/

* DOI:
  https://doi.org/10.1364/OPTCON.525065

Why it matters:

* masks;
* irregular boundaries;
* sparse valid pixels;
* per-pixel and per-channel weights;
* zero-mean normalized correlation;
* fast Fourier formulations.

Potential ZeroModel use:

```text
critical-region weighting
valid-overlap masks
RGB channel weighting
occlusion-aware local matching
```

## Cross-correlation counterexample

**Serap A. Savari — “A Counterexample in Cross-Correlation Template Matching”**

* Paper:
  https://arxiv.org/abs/2410.19085

Why it matters:

Cross-correlation can align poorly under sampling, quantization and noise, especially for piecewise-constant structures.

Potential ZeroModel use:

* design adversarial controls;
* avoid treating maximum correlation as proof;
* compare correlation with differences or segmented structure;
* test sampled one-pixel state distinctions.

## Fast exhaustive NCC

**Wei and Lai — “Fast Template Matching Based on Normalized Cross Correlation with Adaptive Multilevel Winner Update”**

* PubMed:
  https://pubmed.ncbi.nlm.nih.gov/18972660/

* DOI:
  https://doi.org/10.1109/TIP.2008.2004615

Why it matters:

It provides ideas for accelerating exhaustive normalized correlation without immediately replacing it with approximate retrieval.

Potential ZeroModel use:

* preserve deterministic maxima;
* reduce runtime;
* cache reusable patch statistics;
* make local evidence practical per frame.

## Generalized Hough transform

**Dana H. Ballard — “Generalizing the Hough Transform to Detect Arbitrary Shapes”**

* Publisher page:
  https://www.sciencedirect.com/science/article/pii/0031320381900091

* DOI:
  https://doi.org/10.1016/0031-3203(81)90009-1

Why it matters:

It provides a classical route from boundaries and local components to explicit shape and geometry evidence.

This is the primary bridge to deterministic Stage 3 if local correlation remains insufficient.

---

# 30. Research papers: temporal correspondence and tracking

These are references and later baselines. They do not automatically justify adding learned dependencies to 1.0.12.

## RAFT

**Teed and Deng — “RAFT: Recurrent All-Pairs Field Transforms for Optical Flow”**

* Paper:
  https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3526_ECCV_2020_paper.php

Why it matters:

* dense motion;
* all-pairs correlation;
* iterative flow refinement;
* a strong learned optical-flow comparator.

Potential use:

Compare deterministic local displacement against learned dense motion only after the deterministic temporal baseline is measured.

## TAPIR

**Doersch et al. — “TAPIR: Tracking Any Point with Per-Frame Initialization and Temporal Refinement”**

* Paper:
  https://openaccess.thecvf.com/content/ICCV2023/html/Doersch_TAPIR_Tracking_Any_Point_with_Per-Frame_Initialization_and_Temporal_Refinement_ICCV_2023_paper.html

Why it matters:

TAPIR explicitly separates:

```text
per-frame matching
from
temporal refinement
```

That separation is highly relevant to ZeroModel’s proposed local-evidence plus temporal-governance architecture.

## CoTracker

**Karaev et al. — “CoTracker: It Is Better to Track Together”**

* Paper:
  https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7890_ECCV_2024_paper.php

* Project:
  https://co-tracker.github.io/

Why it matters:

* jointly tracks many points;
* models dependencies among tracks;
* handles occlusion and long sequences;
* operates causally in windows.

Potential use:

Later learned temporal baseline for whether joint local evidence offers more stable identity than independent tracks.

## DEVA

**Cheng et al. — “Tracking Anything with Decoupled Video Segmentation”**

* Paper:
  https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_Tracking_Anything_with_Decoupled_Video_Segmentation_ICCV_2023_paper.html

Why it matters:

DEVA separates:

```text
image-level evidence
from
task-agnostic temporal propagation
```

This is conceptually close to ZeroModel’s requirement that the visual provider and temporal governance remain separable.

## SAM 2

**Ravi et al. — “SAM 2: Segment Anything in Images and Videos”**

* Paper:
  https://arxiv.org/abs/2408.00714

Why it matters:

* promptable video segmentation;
* streaming memory;
* persistent masks;
* temporal object identity.

Potential use:

A later segmentation-evidence provider, not the first deterministic implementation.

Do not confuse segmentation persistence with policy-state certainty.

## ByteTrack

**Zhang et al. — “ByteTrack: Multi-Object Tracking by Associating Every Detection Box”**

* Paper:
  https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/315_ECCV_2022_paper.php

Why it matters:

ByteTrack retains low-score detections during association rather than deleting them immediately.

Potential ZeroModel lesson:

> Weak evidence may remain useful for temporal association without being sufficient for governed acceptance.

Do not let weak carried evidence become an accepted state without separate gates.

---

# 31. Research papers: local learned matching

These are later baselines after deterministic local correlation is measured.

## LightGlue

**Lindenberger, Sarlin and Pollefeys — “LightGlue: Local Feature Matching at Light Speed”**

* Paper:
  https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html

Why it matters:

* sparse local feature matching;
* adaptive compute;
* local correspondence rather than global scene embedding.

## Efficient LoFTR

**Wang et al. — “Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed”**

* Paper:
  https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Efficient_LoFTR_Semi-Dense_Local_Feature_Matching_with_Sparse-Like_Speed_CVPR_2024_paper.html

Why it matters:

* semi-dense correspondences;
* texture-poor matching;
* two-stage correlation;
* a useful learned comparison for small structured frames.

## XFeat

**Potje et al. — “XFeat: Accelerated Features for Lightweight Image Matching”**

* Paper:
  https://openaccess.thecvf.com/content/CVPR2024/html/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.html

Why it matters:

* resource-efficient local features;
* sparse and semi-dense modes;
* potential edge comparison.

## RoMa

**Edstedt et al. — “RoMa: Robust Dense Feature Matching”**

* Paper:
  https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html

Why it matters:

RoMa combines coarse DINOv2 robustness with specialized fine features.

This may help explain why global DINOv2 CLS failed while local DINO-derived evidence could still remain viable.

---

# 32. Research papers: selective acceptance and governance

These papers directly address the distinction between prediction and acceptance.

## One-sided selective classification

**Gangrade, Kag and Saligrama — “Selective Classification via One-Sided Prediction”**

* Paper:
  https://proceedings.mlr.press/v130/gangrade21a.html

Why it matters:

It seeks large class-specific decision regions with very few false positives.

Potential ZeroModel use:

* row-specific acceptance regions;
* action-specific safety gates;
* coverage maximization under strict false-positive constraints.

## Selective classification with OOD data

**Xia and Bouganis — “Augmenting Softmax Information for Selective Classification with Out-of-Distribution Data”**

* Paper:
  https://arxiv.org/abs/2207.07506

Why it matters:

It distinguishes:

* correct in-domain predictions;
* incorrect in-domain predictions;
* out-of-distribution observations.

ZeroModel currently needs all three distinctions.

## LEC

**Wang et al. — “Linear Expectation Constraints for Selection-Conditioned Risk Control in Selective Prediction and Routing Systems”**

* Paper:
  https://openreview.net/forum?id=auEvgVBpSF

Why it matters:

It targets risk among the predictions actually selected or accepted.

Potential ZeroModel use:

* formal accepted-risk objectives;
* route or reject decisions;
* sequence-conditioned selection.

Do not claim a formal guarantee merely by citing the paper. Any guarantee must match the implemented assumptions and calibration protocol.

---

# 33. Research papers: visual primitives

## Thinking with Visual Primitives

**Lu et al. — “Thinking with Visual Primitives”**

The original official source has not remained reliably available. Use mirrors cautiously:

* Community mirror:
  https://github.com/mitkox/Thinking-with-Visual-Primitives

* Archived mirror with provenance warning:
  https://github.com/ailuntx/Thinking-with-Visual-Primitives

Why it matters:

The work frames points and boxes as explicit visual-reference units and identifies a “reference gap” where language cannot precisely anchor dense spatial reasoning.

Potential ZeroModel relevance:

* explicit point and box identities;
* referenceable local evidence;
* structured spatial claims;
* coordinates as first-class reasoning evidence.

Do not treat a community mirror as authoritative provenance.

## Action with Visual Primitives

**Guo et al. — “Action with Visual Primitives”**

* Paper:
  https://arxiv.org/abs/2605.22183

Why it matters:

It separates:

```text
instruction comprehension
spatial scene understanding
action generation
```

and uses visual primitives as an interface between perception and action.

That separation is strongly aligned with ZeroModel’s intended architecture:

```text
visual evidence
→ explicit state claims
→ independently identified policy
→ action
```

This is a conceptual reference, not evidence that ZeroModel should adopt a vision-language-action model.

---

# 34. How to use the papers

Do not implement every paper.

For each paper, record:

```text
paper
specific architectural idea
ZeroModel question it informs
smallest bounded experiment
new dependency required
evidence needed
reason to adopt or reject
```

The intended order is:

```text
1. Deterministic weighted local correlation
2. Deterministic temporal transition evidence
3. Deterministic geometry
4. Learned local correspondence baseline
5. Learned temporal tracking baseline
6. Fixed-camera validation
```

Do not jump directly to SAM 2, RAFT, TAPIR or CoTracker merely because they are more capable.

The programme’s discipline is to measure the simplest mechanism first.

---

# 35. 1.0.12 implementation sequence

## Phase A: inherited-state audit

* verify latest `main`;
* verify package version;
* run full tests;
* read logbook, status cut and claims audit;
* verify frozen evidence;
* inspect v3 preparation;
* write the Stage 2 protocol before final-data evaluation.

## Phase B: video transport

* deterministic frame identity;
* clip manifest;
* lossless canonical clip format;
* optional real-video adapter;
* decode and trace tests.

## Phase C: exact canonical video baseline

* generate known arcade trajectory;
* replay through exact visual reader;
* prove symbolic-policy equivalence;
* serialize temporal trace.

## Phase D: frame-local Stage 2 provider

* weighted local correlation;
* multiple regions;
* critical-region semantics;
* competing rows;
* explicit rejection reasons.

## Phase E: temporal provider

* candidate paths;
* transition specification;
* persistence;
* occlusion handling;
* no silent carry-forward;
* temporal rejection.

## Phase F: fresh benchmark and calibration

* freeze splits;
* calculate evidence once;
* choose gates using calibration only;
* evaluate final clips once;
* produce all reports and atlases.

## Phase G: documentation and claims

Update:

```text
README.md
docs/claims-audit.md
docs/research/visual-research-logbook.md
docs/research/visual-programme-status-cut-2026-07-18.md
```

Create:

```text
docs/research/visual-video-stage-two.md
docs/research/video-policy-reader.md
```

Update the dated status cut only through a new entry or successor status document when appropriate. Do not silently rewrite historical state.

## Phase H: release

Only after implementation, tests, evidence and claims pass:

* bump `pyproject.toml` to `1.0.12`;
* update installation examples;
* update release notes or changelog;
* build distributions;
* run Twine checks;
* verify clean install;
* tag only through the repository’s declared release process.

---

# 36. Branch and commit discipline

Create:

```text
research/video-policy-reader-1.0.12
```

Suggested commit sequence:

```text
feat(video): add deterministic frame and clip contracts
feat(video): add canonical video policy baseline
feat(visual): add local correlation evidence provider
feat(video): add temporal policy evidence
test(video): cover temporal safety and replay contracts
data(results): add bounded video stage-two evidence
docs(research): record temporal visual-address result
release: prepare zeromodel 1.0.12
```

Do not combine implementation, evidence and release metadata into one opaque commit.

Open a draft PR:

```text
research: add bounded temporal visual policy reader
```

Do not merge automatically.

---

# 37. Claims discipline

Permitted bounded claims may include:

> ZeroModel can consume a declared video clip, preserve frame and clip identity, and produce a reconstructable temporal policy-decision trace.

> On the declared canonical arcade clips, the video path reproduces the same exact rows and actions as the symbolic policy path.

> On the declared fresh synthetic temporal fixture, the local temporal provider achieved the recorded accepted exact-row coverage with zero observed distinguishable false accepts at the selected operating point.

Do not claim:

```text
general video understanding
general object permanence
general zero false-accept risk
production-ready camera perception
arbitrary motion invariance
semantic scene understanding
real-world safety
human-level temporal reasoning
a complete Visual State Compiler
```

If the approximate provider fails, state that plainly.

The release may still ship:

* video contracts;
* deterministic transport;
* canonical exact video baseline;
* research provider;
* negative evidence.

Do not promote a failed research provider as a production reader.

---

# 38. Stop conditions

Stop and report rather than improvising when:

* current tests fail before implementation;
* predecessor evidence identities fail;
* the v3 fixture is not reproducible;
* video decoding cannot be made deterministic enough for the declared contract;
* final data enters calibration;
* templates use final frames;
* temporal rules are derived from final outcomes;
* sequence metrics cannot be reconstructed;
* codec differences alter canonical evidence;
* frame ordering is ambiguous;
* the implementation requires silently carrying state;
* a learned dependency is required before deterministic baselines are measured;
* claims cannot be supported honestly;
* evidence manifests fail.

---

# 39. Definition of done

ZeroModel 1.0.12 is complete only when:

```text
[ ] latest main audited
[ ] inherited evidence verified
[ ] full baseline suite green
[ ] deterministic video frame contract implemented
[ ] clip manifest implemented
[ ] actual video adapter implemented
[ ] lossless canonical clip format implemented
[ ] exact canonical video baseline measured
[ ] symbolic and video action sequences compared
[ ] local region evidence implemented
[ ] competing-action evidence implemented
[ ] temporal transition specification implemented
[ ] no-silent-carry-forward rule enforced
[ ] temporal candidate paths implemented
[ ] outside-bound safety control implemented
[ ] fresh calibration/final isolation proven
[ ] final temporal benchmark executed once
[ ] frame metrics reconstructed from traces
[ ] sequence metrics reconstructed from traces
[ ] evidence bundle hashes verify
[ ] focused tests pass
[ ] full tests pass
[ ] package build passes
[ ] Twine check passes
[ ] claims audit updated
[ ] research logbook updated
[ ] public README wording remains bounded
[ ] package version changed to 1.0.12 only after all above
[ ] draft PR opened
```

---

# 40. First response in the new chat

After reading the repository, respond first with:

```text
## Inherited ZeroModel position

Current main:
Current version:
Current tests:

Stable policy core:
Measured visual systems:
Confirmed mechanisms:
Retired approaches:
Preparation-only work:
Frozen evidence:
Open Stage 2 question:

## Proposed 1.0.12 boundary

What “video working” will mean:
Minimum shippable capability:
Research provider:
Canonical positive baseline:
Fresh temporal benchmark:
Explicit non-goals:

## Architecture

Video input:
Frame-local evidence:
Temporal evidence:
Acceptance:
Policy delegation:
Evidence output:

## First implementation sequence

1.
2.
3.
4.
5.

## Risks requiring early tests

...
```

Do not ask the user to repeat information already present in the linked repository documents.

After presenting that inherited-state synthesis, proceed through the bounded implementation sequence unless a genuine stop condition is encountered.

---

# 41. Final completion report

At completion, report:

```text
Branch:
Draft PR:

Starting main SHA:
Release candidate SHA:
Version:

Baseline tests before work:
Focused tests:
Full tests:
Build:
Twine:

Video adapter:
Canonical clip format:
Canonical clip digest:
Frame count:
Nominal FPS:

Exact canonical video result:
Exact rows:
Exact actions:
Sequence equivalence:

Frame-local provider:
Provider ID:
Calibration ID:
Raw exact-row:
Raw action:
Accepted coverage:
False accepts:
Conflicting-action accepts:

Temporal provider:
Provider ID:
Transition spec ID:
Raw exact-row:
Raw action:
Accepted exact-row coverage:
Accepted exact-row precision:
Accepted action precision:
False accepts:
Conflicting-action accepts:
Identity switches:
Impossible-transition accepts:
Occlusion recovery:
Outside-bound result:
Outcome:

Evidence directory:
Dataset digest:
Clip manifest digest:
Selection digest:
Trace digest:
Bundle digest:

Files added:
Files modified:

Claims added:
Claims rejected:
Known limitations:
Next declared experiment:
```

Never report 1.0.12 as released unless the repository’s actual release process has been completed.
