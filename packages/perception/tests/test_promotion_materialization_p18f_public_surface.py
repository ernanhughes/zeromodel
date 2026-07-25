from __future__ import annotations

import zeromodel.perception as perception


def test_p18f_is_exposed_from_package_root() -> None:
    expected = {
        "PerceptionPromotionMaterializationError",
        "PromotionMaterializationDirectiveDTO",
        "PromotionMaterializationBaselineDTO",
        "PromotionMaterializationOperationDTO",
        "MaterializedPromotionChangeDTO",
        "PromotionMaterializationChangeSetDTO",
        "materialize_approved_candidate_promotions",
        "PROMOTION_MATERIALIZATION_CHANGE_SET_VERSION",
        "PROMOTION_MATERIALIZATION_DIRECTIVE_VERSION",
    }

    assert expected <= set(perception.__all__)
    for name in expected:
        assert getattr(perception, name) is not None
