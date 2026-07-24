import zeromodel.perception as perception


def test_operational_health_public_contract() -> None:
    assert perception.OPERATIONAL_REFERENCE_PROFILE_VERSION.endswith("/1")
    assert perception.OPERATIONAL_DRIFT_POLICY_VERSION.endswith("/2")
    assert perception.OPERATIONAL_HEALTH_FINDING_VERSION.endswith("/1")
    assert perception.OPERATIONAL_HEALTH_REPORT_VERSION.endswith("/1")
    assert perception.OPERATIONAL_HEALTH_STATUSES == {
        "healthy",
        "drifted",
        "insufficient_evidence",
    }
    policy = perception.OperationalDriftPolicyDTO()
    assert policy.minimum_reference_count == 20
    assert policy.minimum_inference_count == 20
    assert policy.minimum_labeled_count == 20
    assert policy.minimum_accepted_labeled_count == 20
    assert policy.minimum_label_coverage == 1.0
    assert policy.require_single_pointer_revision
    assert callable(perception.build_operational_reference_profile)
    assert callable(perception.diagnose_operational_health)
