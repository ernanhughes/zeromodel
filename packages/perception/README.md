# zeromodel-perception

`zeromodel-perception` is the domain-neutral learning and inference runtime for
ZeroModel visual evidence.

The package turns arbitrary image/action observations into inspectable visual
representations and, in later delivery stages, will use those representations to
infer likely actions for unseen images while retaining evidence, alternatives,
confidence, rejection, and provenance.

## Phase P1 status

Phase P1 implements the first deterministic representation slice:

```text
bounded image bytes or uint8 array -> SourceVPMDTO
bounded discrete action            -> TargetVPMDTO
```

It does not yet learn mappings, search neighbours, infer actions, estimate
confidence, produce evidence VPMs, or persist datasets.

The production dependencies remain deliberately small:

- `numpy` for deterministic array operations;
- `pillow` for bounded image decoding and canonical PNG serialization;
- `zeromodel` for core VPM artifact contracts;
- `zeromodel-observation` for observation-owned DTO and provider contracts.

Pillow remains an input/output adapter. Perception internals operate on validated
NumPy arrays and immutable DTOs rather than passing mutable Pillow objects across
runtime boundaries.

## Source representation

`SourceImageEncoderSpecDTO` declares the accepted colour space and hard limits for
width, height, decoded pixels, and input bytes. P1 accepts `L`, `RGB`, and `RGBA`
uint8 observations.

The encoder:

1. rejects malformed, oversized, or incorrectly shaped inputs;
2. performs only the declared colour-space conversion;
3. preserves encoded pixel orientation and coordinates;
4. emits deterministic PNG bytes;
5. computes separate identities for the encoder contract, normalized pixels,
   PNG bytes, and complete Source VPM.

No crop, resize, augmentation, denoising, histogram correction, EXIF transpose,
object detection, or learned embedding is performed.

## Target representation

`DiscreteActionSchemaDTO` owns a canonical sorted action vocabulary. Each discrete
action is represented as a one-row grayscale PNG with one stable field per action:

```text
0   = inactive field
255 = selected action field
```

`decode_discrete_action` validates schema identity, encoder version, dimensions,
PNG digest, canonical one-hot values, and metadata agreement before returning an
action. It rejects malformed or ambiguous target VPMs rather than guessing.

## Example

```python
import numpy as np

from zeromodel.perception import (
    DiscreteActionSchemaDTO,
    decode_discrete_action,
    encode_discrete_action,
    encode_source_array,
)

source = encode_source_array(np.zeros((8, 8, 3), dtype=np.uint8))
schema = DiscreteActionSchemaDTO.from_labels(["RIGHT", "LEFT", "FIRE"])
target = encode_discrete_action("RIGHT", schema)

assert source.width == 8
assert decode_discrete_action(target, schema) == "RIGHT"
```

## Ownership boundary

`zeromodel-perception` owns or will own:

- deterministic source-image and target-action visual representations;
- immutable image/action dataset manifests;
- similarity and sparse-translation inference for unseen observations;
- evidence VPMs and prediction evidence;
- expected-versus-observed evidence conformance;
- unexplained evidence discovery;
- temporal observation/action/next-observation learning.

It does not own:

- the conservative artifact kernel (`zeromodel.core`);
- closed-world deterministic visual codebook addressing (`zeromodel.vision`);
- video capture or arcade-domain semantics (`zeromodel.video`);
- SQLAlchemy persistence (`zeromodel-sqlalchemy`);
- artifact storage, signatures, or navigation;
- game-specific concepts such as players, aliens, bullets, or controls.

See `docs/architecture/perception-runtime-design.md` for the comprehensive
architecture and staged delivery plan.
