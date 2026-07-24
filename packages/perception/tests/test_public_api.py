from __future__ import annotations

from zeromodel.perception import (
    PERCEPTION_PACKAGE_VERSION,
    PERCEPTION_STAGE,
    DiscreteActionSchemaDTO,
    InMemoryPerceptionDatasetStore,
    SourceImageEncoderSpecDTO,
)


def test_phase_two_public_contract() -> None:
    assert PERCEPTION_PACKAGE_VERSION == "1.0.13"
    assert PERCEPTION_STAGE == "P2"
    assert SourceImageEncoderSpecDTO().color_space == "RGB"
    assert DiscreteActionSchemaDTO.from_labels(["RIGHT", "LEFT"]).labels == (
        "LEFT",
        "RIGHT",
    )
    assert InMemoryPerceptionDatasetStore().list_ids() == ()
