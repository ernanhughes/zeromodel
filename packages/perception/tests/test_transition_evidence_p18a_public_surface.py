from __future__ import annotations

import zeromodel.perception as perception


def test_p18a_is_exposed_from_package_root() -> None:
    expected = {
        "PerceptionTransitionEvidenceError",
        "TransitionEvidenceVPMDTO",
        "TransitionFieldEvidenceDTO",
        "build_transition_evidence_vpm",
        "TRANSITION_EVIDENCE_VPM_VERSION",
    }

    assert expected <= set(perception.__all__)
    for name in expected:
        assert getattr(perception, name) is not None
