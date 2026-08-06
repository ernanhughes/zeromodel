from __future__ import annotations

import pytest

from zeromodel.core.artifact import VPMValidationError
from zeromodel.search.dto import RelationContractDTO, RelationCoordinateSpecDTO


def test_coordinate_order_is_identity_bearing():
    x = RelationCoordinateSpecDTO("x", "x", "mx")
    y = RelationCoordinateSpecDTO("y", "y", "my")
    first = RelationContractDTO("r", "1", "item", (x, y))
    second = RelationContractDTO("r", "1", "item", (y, x))

    assert first.relation_contract_id != second.relation_contract_id


def test_duplicate_coordinates_are_rejected():
    with pytest.raises(VPMValidationError):
        RelationContractDTO(
            "r",
            "1",
            "item",
            (
                RelationCoordinateSpecDTO("x", "x", "mx"),
                RelationCoordinateSpecDTO("x", "x again", "mx2"),
            ),
        )
