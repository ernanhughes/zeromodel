from __future__ import annotations

import zeromodel.perception as perception


def test_p18d_is_exposed_from_package_root() -> None:
    expected = {
        "PerceptionCandidateValidationError",
        "HeldOutTransitionObservationDTO",
        "CandidateValidationPolicyDTO",
        "CandidateValidationExpectationDTO",
        "CandidateValidationFindingDTO",
        "CandidateValidationResultDTO",
        "CandidateValidationReportDTO",
        "validate_discovered_transition_candidates",
        "CANDIDATE_VALIDATION_REPORT_VERSION",
        "CANDIDATE_VALIDATION_POLICY_VERSION",
    }

    assert expected <= set(perception.__all__)
    for name in expected:
        assert getattr(perception, name) is not None
