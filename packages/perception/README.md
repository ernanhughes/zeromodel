# zeromodel-perception

`zeromodel-perception` is the domain-neutral learning and inference runtime for
ZeroModel visual evidence.

The package is intended to turn arbitrary image/action observations into
inspectable visual representations and, in later delivery stages, use those
representations to infer likely actions for unseen images while retaining
supporting evidence, alternatives, confidence, rejection, and provenance.

## Phase P0 status

Phase P0 registers the package and its dependency boundary only. It contains no
scientific behavior, image encoder, learner, predictor, or evidence algorithm.

The initial production dependencies are deliberately small:

- `numpy` for deterministic array operations;
- `pillow` for bounded image decoding and PNG serialization at the package edge;
- `zeromodel` for core VPM artifact contracts;
- `zeromodel-observation` for observation-owned DTO and provider contracts.

Pillow is an input/output adapter. Future perception internals should operate on
validated NumPy arrays and immutable ZeroModel DTOs rather than passing mutable
Pillow objects across service boundaries.

## Ownership boundary

`zeromodel-perception` will own:

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

## Public API

During P0 the public API exposes only package identity and the delivery-stage
marker. Scientific APIs will be added only with focused tests in later vertical
slices.

```python
from zeromodel.perception import PERCEPTION_PACKAGE_VERSION, PERCEPTION_STAGE

assert PERCEPTION_PACKAGE_VERSION == "1.0.13"
assert PERCEPTION_STAGE == "P0"
```

See `docs/architecture/perception-runtime-design.md` for the comprehensive
architecture and staged delivery plan.
