# Visual Sign Reader: observation-addressed policy

Status: **Research / exhaustive exact-codeword closed-world validation**

ZeroModel's state-addressed policy reader expects a stable row ID. This experiment asks whether a bounded visual observation can provide that address directly:

```text
canonical frame
    ↓ deterministic integer feature contract
visual index artifact
    ↓ accepted row ID or rejection
policy artifact
    ↓ VPMPolicyLookup
action + complete trace
```

The lookup replaces the **decision-state address** in a closed renderable world. It does not replace the seeing or generalization required in an open visual world.

## Architecture

The implementation deliberately keeps two artifacts separate.

### Policy artifact

The existing arcade policy is unchanged:

- rows: the 112 finite shooter states;
- columns: `LEFT`, `RIGHT`, `STAY`, `FIRE`;
- consumer: `VPMPolicyLookup`.

### Visual index artifact

The visual index has the same ordered row IDs, but its columns are deterministic visual features. Its provenance declares:

```text
visual index --addresses--> policy artifact
```

The index stores:

- visual feature specification and digest;
- integer grayscale/pooling/quantization contract;
- complete separation-audit result;
- minimum distance between distinct state vectors;
- closest state pair;
- derived acceptance threshold;
- derived nearest/second-nearest margin requirement;
- explicit absolute-gap margin rule;
- exact policy artifact identity addressed.

Changing the renderer, feature contract, calibration, or addressed policy changes the visual index identity without changing the policy itself.

The corrected calibration contract is emitted as `zeromodel-visual-index/v2`. The reader is `zeromodel-visual-sign-reader/v2`. The feature contract remains `zeromodel-visual-feature/v1`.

## Deterministic feature contract

`VisualFeatureSpec` accepts a declared fixed-size grayscale, RGB, or RGBA integer frame. The extractor uses no graphics library and no floating-point image transformation:

1. RGB is converted with fixed integer BT.601-style weights.
2. The frame is divided into exact integer box-pooling regions.
3. Each pooled value is quantized to a declared number of integer levels.
4. The resulting vector and feature specification are independently digestible.

The canonicalization function always creates an owned copy before marking its internal grayscale buffer immutable. A read therefore cannot freeze or otherwise mutate caller-owned frame memory.

The arcade fixture renders directly into a `uint8` NumPy array. It does not use fonts, antialiasing, a GPU, Pillow, or Matplotlib, so the canonical visual fixture avoids the environment-sensitive raster path already excluded from ZeroModel's identity claims.

## Separation audit

Before an index can be built, `build_visual_index()` extracts every declared row vector and computes all pairwise Euclidean distances.

The build fails when any two decision states collide:

```text
min_between_distance <= 0
    → visual states are not separable
    → no index artifact is produced
```

For the committed 112-state arcade fixture:

```text
states                       112
features per state            112
minimum between-state distance 2.0
acceptance threshold            0.5
required distance margin        1.5
```

The threshold and margin are derived from the minimum separation rather than selected at runtime. Both participate in the index artifact identity.

Every policy-relevant state component is visible in the canonical frame:

- tank position;
- target position or target absence;
- cooldown state.

If cooldown were omitted, distinct policy rows could collide. The separation audit is designed to catch exactly that failure.

## Non-vacuous ambiguity contract

Let:

```text
δ  = minimum distance between codebook states
d1 = query distance to nearest state
d2 = query distance to second-nearest state
t  = threshold_fraction
m  = margin_fraction
```

An accepted-distance query satisfies:

```text
d1 <= tδ
```

By the triangle inequality:

```text
d2 - d1 >= δ(1 - 2t)
```

Therefore an absolute-gap ambiguity check can only add information when:

```text
m > 1 - 2t
```

The visual calibration now validates this inequality. Configurations that make `ambiguous_visual_address` redundant with the distance rule are rejected during index construction.

The committed fixture uses:

```text
threshold_fraction = 0.25
margin_fraction    = 0.75
```

A dedicated synthetic fixture proves the ambiguity branch is reachable:

```text
codebook values:      0 and 4
query:                1
nearest distance:     1
second distance:      3
acceptance threshold: 1
required margin:      3
observed margin:      2

result:
    ambiguous_visual_address
```

## Runtime reader

```python
from zeromodel import VisualSignReader

reader = VisualSignReader(
    visual_index,
    policy_artifact,
    action_metric_ids=("LEFT", "RIGHT", "STAY", "FIRE"),
)

decision = reader.read(frame)
```

The reader compiles:

- the quantized feature matrix;
- an exact feature-vector-to-row map;
- row norms for nearest-neighbour fallback;
- nearest-other-state distances;
- the existing optimized policy lookup plan.

Canonical frames take the exact feature path. Non-exact frames use deterministic exhaustive nearest-neighbour search, which is appropriate for 112 rows.

## Acceptance and rejection

A frame is accepted only when both conditions hold:

```text
nearest distance <= compiled acceptance threshold
and
second-nearest distance - nearest distance
    >= compiled required margin
```

Rejection is a normal trace outcome, not an exception. It records:

- canonical input digest;
- feature-vector digest;
- nearest and second-nearest row IDs;
- both distances and their margin;
- acceptance threshold and required margin;
- visual-index and policy artifact identities;
- feature and calibration digests;
- rejection reason.

A rejected frame produces no action.

## Exact frame versus exact feature codeword

`exact_feature_match=True` means the canonical quantized feature vector exactly matches a visual-index row. It does **not** mean the raw input bytes are identical to the original canonical frame.

A sub-quantization input change may therefore produce:

```text
different input digest
same feature digest
exact_feature_match = true
same matched row and action
```

This is intentional and is now covered by a regression test.

A feature-changing perturbation produces:

```text
different feature digest
exact_feature_match = false
```

In the current arcade calibration, the acceptance threshold is `0.5` while distinct integer feature vectors are at least distance `1.0` apart. Consequently the committed arcade fixture validates:

```text
exact feature-codeword addressing
plus calibrated refusal of feature-changing observations
```

It does not yet demonstrate accepted, non-exact nearest-neighbour tolerance. That is deferred to the perturbation benchmark.

## Decision trace

An accepted `VisualDecision` adds the visual-address evidence to the normal policy trace:

- input and feature digests;
- exact-feature flag;
- matched row ID;
- nearest/second-nearest evidence;
- visual-index artifact ID;
- policy artifact ID;
- selected action and value;
- candidates and evidence;
- source and view coordinates of the winning policy cell.

## Committed validation

The test suite establishes the following bounded claims:

1. The integer feature contract is deterministic for grayscale and equivalent RGB input.
2. Reading a frame does not mutate its values or writeability flag.
3. All 112 canonical visual states are separated.
4. The absolute-gap ambiguity rule is mathematically effective.
5. A synthetic query reaches `ambiguous_visual_address`.
6. All 112 canonical frames recover the exact policy row and action.
7. A sub-quantization frame change produces a new input digest but the same exact feature codeword.
8. Blank and deliberately feature-changing frames are rejected with evidence and no action.
9. Deliberately duplicated state frames prevent index construction.
10. An index cannot be paired with a different policy identity.
11. The visual index survives a `.vpm` bundle round trip with identity preserved.
12. Across all 2,401 four-target waves, 31,213 runtime observations produce the same action behaviour as the existing state-addressed policy and clear every wave.

The arcade is the calibration environment because symbolic ground truth and complete enumeration are available. The visual path is intentionally more complex than direct engine-state lookup; its value is demonstrating and validating observation-addressed policy for consumers that do not receive an internal symbolic state.

## Bertin relationship

The Bertin detector is optional and is not part of runtime addressing.

- `VisualSignReader` answers: **which known row does this observation address?**
- `MatrixPatternDetector` answers: **does the visual index contain recoverable non-random row structure?**

A later experiment may produce a discovered visual-topology view of the index. Lookup continues to use the source feature matrix and stable row IDs.

## Claim boundary

Valid wording:

> ZeroModel can compile a complete visual index for the committed bounded arcade world and use canonical observations to recover the same exact feature codewords, policy rows, and actions as the symbolic path across every tested state and four-target trajectory. It can reject the committed unfamiliar and feature-changing fixtures and exposes a separately tested non-vacuous ambiguity branch.

Invalid wording:

- “true visual AI” as a general claim;
- “images directly to solutions” in open worlds;
- “perception without a model” for traffic, natural scenes, or unbounded observations;
- robustness to arbitrary camera, lighting, scale, occlusion, compression, or renderer changes;
- learned visual generalization;
- demonstrated tolerant non-exact recognition in the arcade fixture;
- POMDP solution or universal sufficient-observation discovery.

For open observations, a learned or otherwise generalizing perception component remains necessary. ZeroModel may identify and govern a pinned extractor and compile the bounded policy downstream, but lookup replaces the policy address—not open-world seeing.

## Deferred enhancements

The corrective PR intentionally does not add:

- per-state acceptance radii;
- ratio-based ambiguity tests;
- `np.argpartition`;
- a compact visual consumer plan;
- learned extractors;
- a fixed-camera deployment fixture.

These should be evaluated after the perturbation benchmark rather than added speculatively.

## Next gate

The next experiment should systematically perturb all 112 observations and compare:

- global versus per-state radii;
- absolute-gap versus distance-ratio ambiguity;
- false rejection;
- false acceptance;
- row recovery;
- action recovery;
- quantization levels;
- target feature resolutions.

The direction should stop expanding if modest legitimate perturbations cannot be accepted without materially increasing false acceptance.
