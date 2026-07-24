import zeromodel.perception as perception


def test_phase_sixteen_public_contract() -> None:
    assert perception.PERCEPTION_STAGE == "P16"
    assert perception.OPERATIONAL_REFERENCE_PROFILE_VERSION.endswith("/1")
    assert perception.OPERATIONAL_DRIFT_POLICY_VERSION.endswith("/1")
    assert perception.OPERATIONAL_HEALTH_FINDING_VERSION.endswith("/1")
    assert perception.OPERATIONAL_HEALTH_REPORT_VERSION.endswith("/1")
    assert perception.OPERATIONAL_HEALTH_STATUSES == {
        "healthy",
        "drifted",
        "insufficient_evidence",
    }
    assert perception.OperationalDriftPolicyDTO().minimum_labeled_count == 20
    assert callable(perception.build_operational_reference_profile)
    assert callable(perception.diagnose_operational_health)
