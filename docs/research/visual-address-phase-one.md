# Governed visual addressing: Phase 1 benchmark

## Status

This phase implements the first executable held-out visual-address benchmark.
It is a **research benchmark**, not a validated learned-perception claim.

The package version remains `1.0.11`.

## What this phase adds

- deterministic image-corruption operators;
- a family-held-out bounded arcade dataset generator;
- normalized-pixel template matching (system B);
- a pinned optional DINOv2-small encoder adapter;
- frozen medoid prototype retrieval (system C);
- raw frozen-embedding k-NN (system D);
- a rejection-equipped ridge linear probe (system G);
- independent prototype, calibration, benign-test, distinguishable-rejection,
  information-theoretic-control, and OOD sets;
- executable false-acceptance and false-rejection accounting;
- per-family result counts and optional decision traces.

The current deterministic reader remains system A.

## Run

Core NumPy baselines:

```bash
python examples/arcade_visual_address_benchmark.py
```

Pinned DINOv2 research run:

```bash
pip install -e '.[vision]'
python examples/arcade_visual_address_benchmark.py \
  --encoder dinov2 \
  --output-dir build/visual-phase-one
```

To prohibit model downloads and require an existing local cache:

```bash
python examples/arcade_visual_address_benchmark.py \
  --encoder dinov2 \
  --local-files-only
```

The DINOv2 adapter pins:

- model: `facebook/dinov2-small`;
- revision: `ed25f3a31f01632728cabb09d1542f84ab7b0056`;
- representation: L2-normalized CLS token from `last_hidden_state`;
- loaded state-dictionary digest;
- crop-safe square-letterbox canonicalization;
- Hugging Face processor configuration and preprocessing digest;
- framework versions;
- source and licence record.

The arcade renderer is substantially wider than it is tall. Sending it directly
to the default processor would resize by the short edge and then centre-crop,
discarding policy-relevant pixels near the sides. The adapter therefore centres
the complete source image on an identity-bearing square canvas large enough that
the declared crop cannot remove source pixels.

The encoder is loaded only when explicitly requested. Normal package import and
normal CI remain NumPy-only.

## Dataset

The default fixture creates approximately forty in-domain variants per policy
row across disjoint families.

### Prototype families

- clean intensity jitter;
- lower brightness;
- one-pixel vertical translation;
- palette A.

### Calibration families

- held-out contrast;
- opposite vertical translation;
- palette B;
- low-amplitude integer noise.

### Benign test families

- stronger unseen brightness;
- two-pixel vertical translation;
- palette C;
- a patch placed only over source background pixels.

### Scored rejection interventions

- tank removed;
- cooldown indicator removed.

These observations are visibly outside the complete valid-state renderer and
are expected to reject.

### Information-theoretic control

- target removed when a target exists.

Removing the target produces pixels identical to a legitimate no-target state.
A single-frame visual reader cannot distinguish “target hidden” from “target
truly absent” when the observations are identical. These controls remain in the
report, including acceptance and rejection counts, but are excluded from false-
acceptance and false-rejection denominators. Solving this case requires temporal,
multiview, or independent sensor evidence rather than a stronger global image
embedding.

### OOD families

- blank frames;
- checkerboards;
- a genuinely impossible frame containing two separated tanks.

A family never appears in more than one split.

## Address construction

### System B — normalized template matching

A no-model encoder converts the canonical grayscale frame into a mean-centred,
L2-normalized pixel vector. One medoid is selected per policy row.

### System C — frozen embedding medoids

The frozen encoder embeds prototype examples. One cosine medoid is selected per
policy row.

### System D — raw embedding k-NN

Every prototype embedding is retained. Multiple prototype vectors may address
the same policy row through `VisualAddressManifest`.

### System G — rejection-equipped linear probe

A NumPy ridge least-squares row classifier is fitted on frozen prototype
embeddings. It uses the same independent calibration split and conflicting-action
margin rule as retrieval.

## Calibration

Calibration is performed only on the calibration split.

For every policy row, the build records:

- a lower empirical quantile of correct-row similarity or class score;
- a lower empirical quantile of the margin over the best conflicting-action
  candidate.

The default quantile is intentionally conservative (`0.0`) for the first
fixture. The benchmark must publish operating curves before any deployment
threshold is selected.

## Evaluation semantics

Benign test observations are expected to be accepted and recover the correct
action.

Distinguishable critical interventions and OOD observations are expected to be
rejected.

Information-theoretic controls have no scored accept/reject expectation. Their
behaviour is reported separately.

The report records:

- scored accepted and rejected counts;
- correct rows and actions;
- conflicting-action errors;
- false accepts with an explicit distinguishable-rejection denominator;
- false rejects with an explicit benign denominator;
- correct scored disposition;
- impossibility-control acceptance and rejection counts;
- per-family counts.

The existing `VisualBenchmarkMetrics.action_accuracy` retains its original whole-
scored-evaluation denominator. The executable evaluator additionally reports
`benign_action_accuracy` so correct expected rejections are not confused with
wrong actions.

## Evidence boundary

This phase implements the benchmark and the optional learned baselines. A CI
smoke run proves that the pinned encoder and all benchmark systems execute end to
end; that smoke run is not the final full-variant evidence report.

A measured claim requires a committed report from a full pinned encoder run,
including hardware, runtime, model identity, dataset identity, thresholds, and
raw per-family results.

## Decisions intentionally deferred

- policy-aware projection or metric learning;
- local patch verification;
- critical-evidence safety claims;
- video and multisensor observations;
- automatic architecture selection;
- bridge or physical-control deployment;
- a public claim of open-world visual understanding.

## Kill conditions

The direction should be reduced or stopped if:

1. normalized pixels or the deterministic reader match frozen embeddings within
   the declared operating region;
2. the linear probe matches retrieval with materially less complexity;
3. distinguishable critical interventions are accepted at an unsafe rate;
4. held-out family calibration does not transfer;
5. the governance wrapper reaches audit parity at materially lower complexity;
6. prototype count grows toward one stored vector per observation without a
   compensating governance or inspection benefit.
