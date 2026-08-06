from __future__ import annotations

import pytest

from zeromodel.core.artifact import VPMValidationError
from zeromodel.search.dto import RepresentationSpecDTO


def test_representation_identity_is_stronger_than_dimensions():
    base = RepresentationSpecDTO("p", "m", "r1", 768, "float64", "cls", "unit", "pre")
    changed_revision = RepresentationSpecDTO(
        "p", "m", "r2", 768, "float64", "cls", "unit", "pre"
    )
    changed_pooling = RepresentationSpecDTO(
        "p", "m", "r1", 768, "float64", "mean", "unit", "pre"
    )

    assert base.representation_spec_id != changed_revision.representation_spec_id
    assert base.representation_spec_id != changed_pooling.representation_spec_id


def test_representation_rejects_bad_identity():
    with pytest.raises(VPMValidationError):
        RepresentationSpecDTO(
            "p",
            "m",
            "r1",
            2,
            "float64",
            "p",
            "n",
            "pre",
            representation_spec_id="sha256:" + "0" * 64,
        )
