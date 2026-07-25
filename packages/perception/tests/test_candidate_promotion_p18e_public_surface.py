from __future__ import annotations

import zeromodel.perception as perception


def test_p18e_is_exposed_from_package_root() -> None:
    expected = {
        "PerceptionCandidatePromotionError",
        "CandidatePromotionProposalDTO",
        "CandidatePromotionProposalSetDTO",
        "CandidatePromotionDecisionDTO",
        "CandidatePromotionReviewDTO",
        "propose_validated_candidate_promotions",
        "review_candidate_promotion_proposals",
        "CANDIDATE_PROMOTION_PROPOSAL_VERSION",
        "CANDIDATE_PROMOTION_DECISION_VERSION",
        "CANDIDATE_PROMOTION_REVIEW_VERSION",
    }

    assert expected <= set(perception.__all__)
    for name in expected:
        assert getattr(perception, name) is not None
