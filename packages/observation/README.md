# zeromodel-observation

`zeromodel-observation` defines provider-neutral observation contracts for
ZeroModel. It is the boundary between raw observations and deterministic policy
addressing, exposed through `zeromodel.observation`.

The package depends on the validated core distribution, `zeromodel==1.0.13`, and
uses NumPy for immutable image-array contracts. It does not implement perception,
visual recognition, video policy behavior, model inference, or persistence.

## Install

```powershell
python -m pip install zeromodel==1.0.13 zeromodel-observation==1.0.13
```

## Provider-Neutral Contracts

`ImageObservation` stores an owned, read-only `uint8` image array plus source,
timestamp, and metadata. It accepts `HxW`, `HxWx3`, and `HxWx4` arrays and
derives a deterministic raw digest from the canonical shape and bytes.

`VisualAddressContract` describes what a provider promises before inference:
provider identity, score polarity, replay semantics, representation identity,
calibration identity, policy identity, and source scope.

`VisualAddressDecision` records an accepted address or evidence-bearing
rejection. It links the observation, representation, provider, address artifact,
calibration artifact, and policy artifact without depending on any concrete
vision implementation.

`VisualAddressManifest` and `PrototypeBinding` bind representation rows to
policy row IDs and link those rows to a content-addressed matrix blob. The
manifest is an address contract artifact, not a visual reader.

`DeploymentBinding` records an approved pairing of policy, address artifact,
calibration, source scope, and optional encoder identity. It is not a
cryptographic signature or authorization system.

## Minimal Provider

```python
import numpy as np

from zeromodel.observation import (
    ImageObservation,
    VisualAddressContract,
    VisualAddressDecision,
    VisualAddressProvider,
)


class TableProvider:
    def contract(self) -> VisualAddressContract:
        return VisualAddressContract(
            provider_kind="table-provider",
            provider_version="v1",
            score_semantics="distance",
            observation_spec_digest="obs-spec",
            representation_spec_digest="rep-spec",
            address_artifact_id="address-1",
            calibration_artifact_id="calibration-1",
            policy_artifact_id="policy-1",
            source_scope="fixture:camera",
            replay_contract="exact_decision",
        )

    def read(self, observation: ImageObservation) -> VisualAddressDecision:
        return VisualAddressDecision(
            accepted=True,
            reason="accepted",
            observation_digest=observation.raw_digest,
            representation_digest="sha256:representation",
            provider_kind="table-provider",
            provider_version="v1",
            score_semantics="distance",
            address_artifact_id="address-1",
            calibration_artifact_id="calibration-1",
            policy_artifact_id="policy-1",
            nearest_row_id="policy-row-1",
            nearest_score=0.0,
            second_row_id=None,
            second_score=None,
            ambiguity_measure=None,
            matched_row_id="policy-row-1",
            exact_match=True,
            accepted_by=("fixture",),
        )


provider: VisualAddressProvider = TableProvider()
decision = provider.read(ImageObservation(np.zeros((2, 2), dtype=np.uint8)))
```

## Exclusions

Concrete deterministic visual addressing belongs to `zeromodel-vision`.
Temporal policy and video-domain behavior belong to `zeromodel-video`.
Durable SQL persistence belongs to `zeromodel-sqlalchemy`.

This package does not import or require `zeromodel-analysis`, `zeromodel-vision`,
`zeromodel-video`, SQLAlchemy, Torch, TorchVision, Transformers, or Pillow.
