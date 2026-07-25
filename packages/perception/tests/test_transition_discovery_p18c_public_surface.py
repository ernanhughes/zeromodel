from __future__ import annotations

import zeromodel.perception as perception


def test_p18c_is_exposed_from_package_root() -> None:
    expected = {
        "PerceptionTransitionDiscoveryError",
        "UnexplainedFieldOccurrenceDTO",
        "TransitionDiscoveryObservationDTO",
        "TransitionDiscoveryPolicyDTO",
        "RecurrentUnexplainedStatisticDTO",
        "MissingComponentCandidateDTO",
        "TransitionDiscoveryReportDTO",
        "discover_recurrent_unexplained_transitions",
        "TRANSITION_DISCOVERY_REPORT_VERSION",
        "TRANSITION_DISCOVERY_POLICY_VERSION",
    }

    assert expected <= set(perception.__all__)
    for name in expected:
        assert getattr(perception, name) is not None
