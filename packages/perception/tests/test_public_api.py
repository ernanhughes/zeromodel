from __future__ import annotations

from zeromodel.perception import PERCEPTION_PACKAGE_VERSION, PERCEPTION_STAGE


def test_phase_zero_public_contract() -> None:
    assert PERCEPTION_PACKAGE_VERSION == "1.0.13"
    assert PERCEPTION_STAGE == "P0"
