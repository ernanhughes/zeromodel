from zeromodel.perception import (
    PERCEPTION_STAGE,
    PRODUCTION_INFERENCE_SEMANTICS,
    PRODUCTION_METRICS_SEMANTICS,
    PRODUCTION_OUTCOME_SEMANTICS,
    InMemoryPerceptionProductionLedgerStore,
)


def test_phase_fourteen_public_contract() -> None:
    assert PERCEPTION_STAGE == "P14"
    assert PRODUCTION_INFERENCE_SEMANTICS == (
        "append_only_runtime_inference_bound_to_active_model_pointer_revision"
    )
    assert PRODUCTION_OUTCOME_SEMANTICS == (
        "append_only_observed_outcome_for_runtime_inference"
    )
    assert PRODUCTION_METRICS_SEMANTICS == (
        "windowed_operational_metrics_over_immutable_inference_and_outcome_records"
    )
    assert InMemoryPerceptionProductionLedgerStore().list_inferences() == ()
