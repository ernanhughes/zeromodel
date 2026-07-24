from zeromodel import perception


def test_p16d_partition_governance_public_contract() -> None:
    assert perception.PARTITION_OWNED_COMPARISON_VERSION.endswith("/1")
    assert perception.PARTITION_OWNED_TEST_VERSION.endswith("/1")
    assert perception.DATASET_PARTITION_VERSION.endswith("/1")
    assert callable(perception.bind_comparison_report_to_partition)
    assert callable(perception.calibrate_partition_owned_candidates)
    assert callable(perception.promote_partition_owned_model)
    assert callable(perception.evaluate_partition_owned_model_on_test)
