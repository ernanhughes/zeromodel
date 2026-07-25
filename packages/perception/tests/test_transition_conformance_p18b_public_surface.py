from __future__ import annotations

import zeromodel.perception as perception


def test_p18b_is_exposed_from_package_root() -> None:
    expected = {
        "PerceptionTransitionConformanceError",
        "TransitionConformanceFindingDTO",
        "TransitionConformanceReportDTO",
        "TransitionExpectationDTO",
        "evaluate_transition_conformance",
        "TRANSITION_CONFORMANCE_REPORT_VERSION",
        "TRANSITION_EXPECTATION_VERSION",
    }

    assert expected <= set(perception.__all__)
    for name in expected:
        assert getattr(perception, name) is not None
