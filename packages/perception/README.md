# zeromodel-perception

`zeromodel-perception` is the domain-neutral learning and inference runtime for
ZeroModel visual evidence.

## Phase P3 status

P3 is the first complete prediction slice:

```text
immutable image/action dataset
        +
unknown SourceVPMDTO
        ↓
ranked actions
+ confidence
+ nearest supporting observations
+ explicit rejection
```

The baseline deliberately uses the entire normalized VPM. Field relevance,
evidence weighting, sparse translation, semantic annotation, and temporal models
belong to later stages.

## Representation

P1 provides deterministic source-image and target-action artifacts:

```text
bounded image bytes or uint8 array -> SourceVPMDTO
bounded discrete action            -> TargetVPMDTO
```

`SourceImageEncoderSpecDTO` declares colour space and hard width, height, pixel,
and input-byte limits. P1 accepts `L`, `RGB`, and `RGBA` uint8 observations and
performs no implicit crop, resize, augmentation, denoising, EXIF transpose,
object detection, or learned embedding.

`DiscreteActionSchemaDTO` owns a canonical sorted action vocabulary. Each action
is encoded as a deterministic one-row one-hot grayscale PNG.

## Dataset ledger

P2 records authoritative pairings in `RecordedInteractionDTO` and builds immutable
`PerceptionDatasetManifestDTO` values. It detects conflicting actions for identical
pixels, duplicate sequence steps, non-monotonic timestamps, mixed schemas, and
missing identities. Dataset splits are deterministic from interaction identity and
an explicit seed.

## Baseline model

`fit_baseline_nearest_neighbor` accepts:

- a P2 dataset manifest;
- an explicit mapping of `source_vpm_id` to `SourceVPMDTO`;
- a bounded `BaselineInferenceConfigDTO`;
- a declared training split.

The fitter validates every source reference, requires one source shape and encoder
contract, and materializes a self-contained immutable
`BaselineNearestNeighborModelDTO`.

The model uses normalized mean absolute pixel distance:

```text
0.0 = pixel-identical
1.0 = maximum possible uint8 difference
```

The nearest observations vote with inverse-distance weights. Predictions preserve:

- ranked `ActionCandidateDTO` values;
- `NeighborEvidenceDTO` records with exact interaction identities;
- winning weight share as confidence;
- nearest distance;
- top-two action margin;
- deterministic prediction identity.

Prediction statuses are:

```text
accepted
rejected_out_of_distribution
rejected_ambiguous
```

A rejected prediction still retains its candidates and neighbour evidence.

## Example

```python
import numpy as np

from zeromodel.perception import (
    BaselineInferenceConfigDTO,
    DiscreteActionSchemaDTO,
    RecordedInteractionDTO,
    SourceImageEncoderSpecDTO,
    build_dataset_manifest,
    encode_discrete_action,
    encode_source_array,
    fit_baseline_nearest_neighbor,
    predict_baseline_action,
)

spec = SourceImageEncoderSpecDTO(color_space="L")
schema = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])

left = encode_source_array(np.zeros((4, 4), dtype=np.uint8), spec)
right = encode_source_array(np.full((4, 4), 255, dtype=np.uint8), spec)

interactions = [
    RecordedInteractionDTO.from_vpms(
        sequence_id="example",
        step_index=0,
        source=left,
        target=encode_discrete_action("LEFT", schema),
    ),
    RecordedInteractionDTO.from_vpms(
        sequence_id="example",
        step_index=1,
        source=right,
        target=encode_discrete_action("RIGHT", schema),
    ),
]
manifest = build_dataset_manifest(
    interactions,
    source_encoder_spec_ids=[spec.encoder_spec_id],
)
model = fit_baseline_nearest_neighbor(
    manifest,
    {left.source_vpm_id: left, right.source_vpm_id: right},
    training_split="all",
    config=BaselineInferenceConfigDTO(neighbor_count=2),
)
unknown = encode_source_array(np.full((4, 4), 10, dtype=np.uint8), spec)
prediction = predict_baseline_action(model, unknown)

assert prediction.selected_action == "LEFT"
```

## Dependencies and ownership

The production dependencies remain deliberately small:

- `numpy` for deterministic array operations and distance calculation;
- `pillow` for bounded image decoding and canonical PNG serialization;
- `zeromodel` for core VPM artifact contracts;
- `zeromodel-observation` for observation-owned contracts.

Pillow remains an input/output adapter. Perception internals operate on validated
NumPy arrays and immutable DTOs.

The package does not own the conservative core artifact kernel, closed-world
`zeromodel.vision` addressing, video capture, arcade semantics, SQLAlchemy
persistence, artifact signatures, or game-specific concepts.

See `docs/architecture/perception-runtime-design.md` for the comprehensive
architecture and staged delivery plan.
