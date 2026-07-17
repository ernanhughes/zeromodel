# Visual Sign Reader: observation-addressed policy

Status: **Research / exhaustive closed-world validation**

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
- exact policy artifact identity addressed.

Changing the renderer, feature contract, calibration, or addressed policy changes the visual index identity without changing the policy itself.

## Deterministic feature contract

`VisualFeatureSpec` accepts a declared fixed-size grayscale, RGB, or RGBA integer frame. The extractor uses no graphics library and no floating-point image transformation:

1. RGB is converted with fixed integer BT.601-style weights.
2. The frame is divided into exact integer box-pooling regions.
3. Each pooled value is quantized to a declared number of integer levels.
4. The resulting vector and feature specification are independently digestible.

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
required distance margin        0.5
```

The threshold and margin are derived from the minimum separation rather than selected at runtime. Both participate in the index artifact identity.

Every policy-relevant state component is visible in the canonical frame:

- tank position;
- target position or target absence;
- cooldown state.

If cooldown were omitted, distinct policy rows could collide. The separation audit is designed to catch exactly that failure.

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
second-nearest distance - nearest distance >= compiled required margin
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

## Decision trace

An accepted `VisualDecision` adds the visual-address evidence to the normal policy trace:

- input and feature digests;
- exact-match flag;
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
2. All 112 canonical visual states are separated.
3. All 112 frames recover the exact policy row and action.
4. The visual index survives a `.vpm` bundle round trip with identity preserved.
5. Blank and deliberately corrupted frames are rejected with evidence and no action.
6. Deliberately duplicated state frames prevent index construction.
7. An index cannot be paired with a different policy identity.
8. Across all 2,401 four-target waves, 31,213 runtime observations produce the same action behaviour as the existing state-addressed policy and clear every wave.

The arcade is the calibration environment because symbolic ground truth and complete enumeration are available. The visual path is intentionally more complex than direct engine-state lookup; its value is demonstrating and validating observation-addressed policy for consumers that do not receive an internal symbolic state.

## Bertin relationship

The Bertin detector is optional and is not part of runtime addressing.

- `VisualSignReader` answers: **which known row does this observation address?**
- `MatrixPatternDetector` answers: **does the visual index contain recoverable non-random row structure?**

A later experiment may produce a discovered visual-topology view of the index. Lookup continues to use the source feature matrix and stable row IDs.

## Claim boundary

Valid wording:

> ZeroModel can compile a complete visual index for the committed bounded arcade world, reject frames outside its calibrated visual neighbourhood, and address the same policy actions from canonical observations as from symbolic state across every tested state and reachable four-target trajectory.

Invalid wording:

- “true visual AI” as a general claim;
- “images directly to solutions” in open worlds;
- “perception without a model” for traffic, natural scenes, or unbounded observations;
- robustness to arbitrary camera, lighting, scale, occlusion, compression, or renderer changes;
- learned visual generalization;
- POMDP solution or universal sufficient-observation discovery.

For open observations, a learned or otherwise generalizing perception component remains necessary. ZeroModel may identify and govern a pinned extractor and compile the bounded policy downstream, but lookup replaces the policy address—not open-world seeing.

## Next gate

The next experiment should move only one rung outward: a fixed-camera, installation-calibrated scene with a bounded set of marked regions. It should retain explicit rejection and compare deterministic features against a pinned learned extractor. The direction should stop expanding if false acceptance cannot be controlled or if the visual index offers no operational benefit over direct instrumentation.
