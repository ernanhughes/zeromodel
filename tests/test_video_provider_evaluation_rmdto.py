from __future__ import annotations

from typing import Any, Sequence

import pytest

from test_video_episode_plan_rmdto import plan_dto, sample_identity
from test_video_observation_rmdto import _pixels, sample_record

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    PROVIDER_EVALUATION_RUN_VERSION,
)
from zeromodel.video.domains.video_action_set.observation_dto import ObservationDTO
from zeromodel.video.domains.video_action_set.provider_evaluation_dto import (
    CASE_OUTCOME_ACTION_CHANGING,
    CASE_OUTCOME_ACTION_EQUIVALENT,
    CASE_OUTCOME_EXACT,
    CASE_OUTCOME_REJECTED,
    MaterializedProviderEvaluationRunDTO,
    ProviderConfigurationDTO,
    ProviderEvaluationCaseContext,
    ProviderEvaluationCaseDTO,
    ProviderEvaluationRunDTO,
    ProviderEvaluationSummaryDTO,
    ProviderResponseEvidence,
    build_provider_evaluation_run,
)
from zeromodel.video.domains.video_action_set.store import (
    PROVIDER_EVALUATION_CASE_CONFLICT_MESSAGE,
    UNKNOWN_OBSERVATION_FOR_PROVIDER_EVALUATION_MESSAGE,
)
from zeromodel.video.stores.video_action_set_memory import InMemoryVideoActionSetStore

POLICY_ARTIFACT_ID = "sha256:" + "2" * 64
OTHER_POLICY_ARTIFACT_ID = "sha256:" + "3" * 64


def sample_configuration(**overrides: Any) -> ProviderConfigurationDTO:
    kwargs: dict[str, Any] = dict(
        provider_kind="fake",
        model_name="fake-model",
        model_digest="sha256:" + "0" * 64,
        runtime_name="fake-runtime",
        protocol_version="v1",
        prompt_digest="sha256:" + "1" * 64,
        seed=0,
        inference_options={"temperature": 0.0},
    )
    kwargs.update(overrides)
    return ProviderConfigurationDTO.build(**kwargs)


def sample_decision(
    row_id: str, action: str, *, artifact_id: str = POLICY_ARTIFACT_ID
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "row_id": row_id,
        "action": action,
        "metric_id": action,
        "value": 1.0,
        "source_row_index": 0,
        "source_metric_index": 0,
        "view_row": 0,
        "view_column": 0,
        "candidates": {action: 1.0},
        "evidence": {},
    }


def sample_case(
    *,
    case_ordinal: int = 0,
    frame_id: str = "development:x:frame-00",
    policy_artifact_id: str = POLICY_ARTIFACT_ID,
    provider_configuration_id: str | None = None,
    outcome: str = "exact",
    provider_latency_us: int | None = 1000,
    provider_confidence: float | None = 0.9,
) -> ProviderEvaluationCaseDTO:
    configuration_id = (
        provider_configuration_id or sample_configuration().provider_configuration_id
    )
    context = ProviderEvaluationCaseContext(
        policy_artifact_id=policy_artifact_id,
        provider_configuration_id=configuration_id,
    )
    expected_state = {"tank_column": 0, "cooldown": 0}
    expected_decision = sample_decision("r0", "STAY", artifact_id=policy_artifact_id)
    if outcome == CASE_OUTCOME_REJECTED:
        return ProviderEvaluationCaseDTO.build(
            case_ordinal=case_ordinal,
            frame_id=frame_id,
            context=context,
            expected_state=expected_state,
            expected_decision=expected_decision,
            accepted=False,
            evidence=ProviderResponseEvidence(
                rejection_reason="confidence_below_threshold",
                provider_confidence=0.1,
            ),
        )
    if outcome == CASE_OUTCOME_EXACT:
        predicted_state = dict(expected_state)
        predicted_decision = sample_decision(
            "r0", "STAY", artifact_id=policy_artifact_id
        )
    elif outcome == CASE_OUTCOME_ACTION_EQUIVALENT:
        predicted_state = {"tank_column": 1, "cooldown": 0}
        predicted_decision = sample_decision(
            "r1", "STAY", artifact_id=policy_artifact_id
        )
    elif outcome == CASE_OUTCOME_ACTION_CHANGING:
        predicted_state = {"tank_column": 2, "cooldown": 0}
        predicted_decision = sample_decision(
            "r2", "LEFT", artifact_id=policy_artifact_id
        )
    else:
        raise ValueError(f"unknown outcome {outcome!r}")
    return ProviderEvaluationCaseDTO.build(
        case_ordinal=case_ordinal,
        frame_id=frame_id,
        context=context,
        expected_state=expected_state,
        expected_decision=expected_decision,
        accepted=True,
        predicted_state=predicted_state,
        predicted_decision=predicted_decision,
        evidence=ProviderResponseEvidence(
            provider_confidence=provider_confidence,
            provider_latency_us=provider_latency_us,
        ),
    )


def store_with_observations(
    count: int,
) -> tuple[InMemoryVideoActionSetStore, tuple[str, ...]]:
    store = InMemoryVideoActionSetStore()
    identity = sample_identity()
    store.save_identity(identity)
    plan = plan_dto(identity=identity, split="development", frame_count=count)
    store.save_episode_plan(plan)
    frame_ids: list[str] = []
    for index in range(count):
        pixels = _pixels(offset=index * 7)
        record = sample_record(plan=plan, sequence_number=index, pixels=pixels)
        materialized = ObservationDTO.from_record(record)
        store.save_observation(
            materialized.observation, matrix_blob=materialized.matrix_blob
        )
        frame_ids.append(materialized.observation.frame_id)
    return store, tuple(frame_ids)


def exact_fixture(frame_ids: Sequence[str]) -> list[ProviderEvaluationCaseDTO]:
    """Fixture A: 8/8 exact, matching the brief's labelled-smoke shape."""
    configuration_id = sample_configuration().provider_configuration_id
    return [
        sample_case(
            case_ordinal=index,
            frame_id=frame_id,
            provider_configuration_id=configuration_id,
            outcome=CASE_OUTCOME_EXACT,
            provider_latency_us=1000 + index * 10,
        )
        for index, frame_id in enumerate(frame_ids)
    ]


def imperfect_fixture(frame_ids: Sequence[str]) -> list[ProviderEvaluationCaseDTO]:
    """Fixture B: 8 attempted, 8 accepted, 3 exact / 4 action_equivalent / 1
    action_changing / 7 action-correct - the observed unlabelled-fixture shape."""
    assert len(frame_ids) == 8
    outcomes = [
        CASE_OUTCOME_EXACT,
        CASE_OUTCOME_EXACT,
        CASE_OUTCOME_EXACT,
        CASE_OUTCOME_ACTION_EQUIVALENT,
        CASE_OUTCOME_ACTION_EQUIVALENT,
        CASE_OUTCOME_ACTION_EQUIVALENT,
        CASE_OUTCOME_ACTION_EQUIVALENT,
        CASE_OUTCOME_ACTION_CHANGING,
    ]
    configuration_id = sample_configuration().provider_configuration_id
    return [
        sample_case(
            case_ordinal=index,
            frame_id=frame_id,
            provider_configuration_id=configuration_id,
            outcome=outcome,
            provider_latency_us=1000 + index * 100,
        )
        for index, (frame_id, outcome) in enumerate(zip(frame_ids, outcomes))
    ]


# --------------------------------------------------------------------------
# ProviderConfigurationDTO
# --------------------------------------------------------------------------


def test_configuration_round_trip_and_determinism() -> None:
    configuration = sample_configuration()
    round_tripped = ProviderConfigurationDTO.from_dict(configuration.to_dict())
    assert round_tripped == configuration
    assert (
        sample_configuration().provider_configuration_id
        == configuration.provider_configuration_id
    )


def test_configuration_rejects_secret_keys_in_inference_options() -> None:
    with pytest.raises(VPMValidationError, match="must not contain secrets"):
        sample_configuration(inference_options={"api_key": "shh"})


def test_configuration_rejects_secret_keys_in_metadata() -> None:
    with pytest.raises(VPMValidationError, match="must not contain secrets"):
        sample_configuration(metadata={"Authorization": "Bearer xyz"})


def test_configuration_rejects_secret_keys_nested() -> None:
    with pytest.raises(VPMValidationError, match="must not contain secrets"):
        sample_configuration(metadata={"nested": {"password": "hunter2"}})


def test_configuration_rejects_invalid_model_digest() -> None:
    with pytest.raises(VPMValidationError):
        sample_configuration(model_digest="not-a-digest")


def test_configuration_id_tamper_detected() -> None:
    configuration = sample_configuration()
    payload = configuration.to_dict()
    payload["provider_kind"] = "ollama"
    with pytest.raises(VPMValidationError, match="provider configuration id mismatch"):
        ProviderConfigurationDTO.from_dict(
            payload
            | {"provider_configuration_id": configuration.provider_configuration_id}
        )


# --------------------------------------------------------------------------
# ProviderEvaluationCaseDTO
# --------------------------------------------------------------------------


def test_case_outcome_classification() -> None:
    assert sample_case(outcome=CASE_OUTCOME_EXACT).outcome == CASE_OUTCOME_EXACT
    assert (
        sample_case(outcome=CASE_OUTCOME_ACTION_EQUIVALENT).outcome
        == CASE_OUTCOME_ACTION_EQUIVALENT
    )
    assert (
        sample_case(outcome=CASE_OUTCOME_ACTION_CHANGING).outcome
        == CASE_OUTCOME_ACTION_CHANGING
    )
    assert sample_case(outcome=CASE_OUTCOME_REJECTED).outcome == CASE_OUTCOME_REJECTED


def test_case_exact_implies_both_matches() -> None:
    case = sample_case(outcome=CASE_OUTCOME_EXACT)
    assert case.exact_state_match is True
    assert case.action_match is True


def test_case_action_equivalent_state_wrong_action_right() -> None:
    case = sample_case(outcome=CASE_OUTCOME_ACTION_EQUIVALENT)
    assert case.exact_state_match is False
    assert case.action_match is True


def test_case_action_changing_action_wrong() -> None:
    case = sample_case(outcome=CASE_OUTCOME_ACTION_CHANGING)
    assert case.action_match is False


def test_case_rejected_has_no_predicted_result() -> None:
    case = sample_case(outcome=CASE_OUTCOME_REJECTED)
    assert case.predicted_state is None
    assert case.predicted_row_id is None
    assert case.predicted_action is None
    assert case.predicted_decision_trace is None
    assert case.exact_state_match is False
    assert case.action_match is False


def test_case_round_trip() -> None:
    case = sample_case(outcome=CASE_OUTCOME_ACTION_EQUIVALENT)
    assert ProviderEvaluationCaseDTO.from_dict(case.to_dict()) == case


def test_case_deterministic_identity() -> None:
    first = sample_case(outcome=CASE_OUTCOME_EXACT)
    second = sample_case(outcome=CASE_OUTCOME_EXACT)
    assert first.case_id == second.case_id


def test_case_id_changes_with_ordinal() -> None:
    first = sample_case(case_ordinal=0)
    second = sample_case(case_ordinal=1)
    assert first.case_id != second.case_id


def test_case_rejects_invalid_policy_artifact_id() -> None:
    with pytest.raises(VPMValidationError):
        sample_case(policy_artifact_id="not-sha256")


def test_case_rejects_invalid_confidence() -> None:
    with pytest.raises(VPMValidationError):
        sample_case(provider_confidence=1.5)


def test_case_rejects_negative_latency() -> None:
    with pytest.raises(VPMValidationError):
        sample_case(provider_latency_us=-1)


def test_case_rejects_fabricated_exact_state_match() -> None:
    case = sample_case(outcome=CASE_OUTCOME_ACTION_EQUIVALENT)
    payload = case.to_dict()
    payload["exact_state_match"] = True
    with pytest.raises(VPMValidationError, match="exact_state_match mismatch"):
        ProviderEvaluationCaseDTO.from_dict(payload | {"case_id": case.case_id})


def test_case_rejects_fabricated_action_match() -> None:
    case = sample_case(outcome=CASE_OUTCOME_ACTION_CHANGING)
    payload = case.to_dict()
    payload["action_match"] = True
    with pytest.raises(VPMValidationError, match="action_match mismatch"):
        ProviderEvaluationCaseDTO.from_dict(payload | {"case_id": case.case_id})


def test_case_rejects_fabricated_outcome() -> None:
    case = sample_case(outcome=CASE_OUTCOME_ACTION_EQUIVALENT)
    payload = case.to_dict()
    payload["outcome"] = CASE_OUTCOME_EXACT
    with pytest.raises(VPMValidationError, match="outcome mismatch"):
        ProviderEvaluationCaseDTO.from_dict(payload | {"case_id": case.case_id})


def test_case_rejects_fabricated_factor_matches() -> None:
    case = sample_case(outcome=CASE_OUTCOME_ACTION_EQUIVALENT)
    payload = case.to_dict()
    payload["factor_matches"] = {"tank_column": True, "cooldown": True}
    with pytest.raises(VPMValidationError, match="factor matches mismatch"):
        ProviderEvaluationCaseDTO.from_dict(payload | {"case_id": case.case_id})


def test_accepted_case_without_predicted_state_rejected() -> None:
    case = sample_case(outcome=CASE_OUTCOME_EXACT)
    payload = case.to_dict()
    payload["predicted_state"] = None
    payload["predicted_row_id"] = None
    payload["predicted_action"] = None
    payload["predicted_decision_trace"] = None
    with pytest.raises(VPMValidationError, match="missing predicted result"):
        ProviderEvaluationCaseDTO.from_dict(payload | {"case_id": case.case_id})


def test_rejected_case_with_predicted_action_rejected() -> None:
    case = sample_case(outcome=CASE_OUTCOME_REJECTED)
    payload = case.to_dict()
    payload["predicted_action"] = "STAY"
    with pytest.raises(VPMValidationError, match="must not carry a predicted result"):
        ProviderEvaluationCaseDTO.from_dict(payload | {"case_id": case.case_id})


def test_predicted_decision_trace_row_mismatch_rejected() -> None:
    case = sample_case(outcome=CASE_OUTCOME_EXACT)
    payload = case.to_dict()
    payload["predicted_row_id"] = "different-row"
    with pytest.raises(VPMValidationError, match="predicted decision trace mismatch"):
        ProviderEvaluationCaseDTO.from_dict(payload | {"case_id": case.case_id})


def test_expected_decision_trace_action_mismatch_rejected() -> None:
    case = sample_case(outcome=CASE_OUTCOME_EXACT)
    payload = case.to_dict()
    payload["expected_action"] = "FIRE"
    with pytest.raises(VPMValidationError, match="expected decision trace mismatch"):
        ProviderEvaluationCaseDTO.from_dict(payload | {"case_id": case.case_id})


def test_raw_response_digest_mismatch_rejected() -> None:
    case = ProviderEvaluationCaseDTO.build(
        case_ordinal=0,
        frame_id="development:x:frame-00",
        context=ProviderEvaluationCaseContext(
            policy_artifact_id=POLICY_ARTIFACT_ID,
            provider_configuration_id=sample_configuration().provider_configuration_id,
        ),
        expected_state={"tank_column": 0},
        expected_decision=sample_decision("r0", "STAY"),
        accepted=True,
        predicted_state={"tank_column": 0},
        predicted_decision=sample_decision("r0", "STAY"),
        evidence=ProviderResponseEvidence(provider_raw_response_text="TANK_COLUMN: 0"),
    )
    payload = case.to_dict()
    payload["provider_raw_response_text"] = "TANK_COLUMN: 1"
    with pytest.raises(VPMValidationError, match="raw response digest mismatch"):
        ProviderEvaluationCaseDTO.from_dict(payload | {"case_id": case.case_id})


def test_case_id_tamper_detected() -> None:
    case = sample_case(outcome=CASE_OUTCOME_EXACT)
    payload = case.to_dict()
    payload["provider_confidence"] = 0.5
    with pytest.raises(VPMValidationError, match="case id mismatch"):
        ProviderEvaluationCaseDTO.from_dict(payload | {"case_id": case.case_id})


# --------------------------------------------------------------------------
# ProviderEvaluationSummaryDTO
# --------------------------------------------------------------------------


def test_summary_from_exact_fixture() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(8))
    cases = exact_fixture(frame_ids)
    summary = ProviderEvaluationSummaryDTO.from_cases(cases)
    assert summary.attempted_count == 8
    assert summary.accepted_count == 8
    assert summary.rejected_count == 0
    assert summary.exact_count == 8
    assert summary.action_equivalent_count == 0
    assert summary.action_changing_count == 0
    assert summary.action_correct_count == 8


def test_summary_from_imperfect_fixture_matches_observed_shape() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(8))
    cases = imperfect_fixture(frame_ids)
    summary = ProviderEvaluationSummaryDTO.from_cases(cases)
    assert summary.attempted_count == 8
    assert summary.accepted_count == 8
    assert summary.rejected_count == 0
    assert summary.exact_count == 3
    assert summary.action_equivalent_count == 4
    assert summary.action_changing_count == 1
    assert summary.action_correct_count == 7


def test_summary_counts_rejections() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(3))
    cases = [
        sample_case(case_ordinal=0, frame_id=frame_ids[0], outcome=CASE_OUTCOME_EXACT),
        sample_case(
            case_ordinal=1, frame_id=frame_ids[1], outcome=CASE_OUTCOME_REJECTED
        ),
        sample_case(
            case_ordinal=2, frame_id=frame_ids[2], outcome=CASE_OUTCOME_REJECTED
        ),
    ]
    summary = ProviderEvaluationSummaryDTO.from_cases(cases)
    assert summary.attempted_count == 3
    assert summary.accepted_count == 1
    assert summary.rejected_count == 2
    assert summary.rejection_reason_counts.to_value() == {
        "confidence_below_threshold": 2
    }


def test_summary_factor_denominators_reflect_accepted_only() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(2))
    cases = [
        sample_case(case_ordinal=0, frame_id=frame_ids[0], outcome=CASE_OUTCOME_EXACT),
        sample_case(
            case_ordinal=1, frame_id=frame_ids[1], outcome=CASE_OUTCOME_REJECTED
        ),
    ]
    summary = ProviderEvaluationSummaryDTO.from_cases(cases)
    assert summary.factor_denominators.to_value() == {"tank_column": 1, "cooldown": 1}
    assert summary.factor_correct_counts.to_value() == {"tank_column": 1, "cooldown": 1}


def test_summary_deterministic_median_and_p95() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(4))
    latencies = [1000, 2000, 3000, 4000]
    cases = [
        sample_case(
            case_ordinal=index,
            frame_id=frame_ids[index],
            outcome=CASE_OUTCOME_EXACT,
            provider_latency_us=latency,
        )
        for index, latency in enumerate(latencies)
    ]
    summary = ProviderEvaluationSummaryDTO.from_cases(cases)
    # nearest-rank: rank(p) = ceil(p*n) clamped to [1,n]; n=4
    # median p=0.50 -> rank=2 -> sorted[1] = 2000
    # p95 p=0.95 -> rank=ceil(3.8)=4 -> sorted[3] = 4000
    assert summary.latency_median_us == 2000
    assert summary.latency_p95_us == 4000
    assert summary.latency_min_us == 1000
    assert summary.latency_max_us == 4000
    assert summary.latency_total_us == 10000


def test_summary_empty_cases_allowed() -> None:
    summary = ProviderEvaluationSummaryDTO.from_cases([])
    assert summary.attempted_count == 0
    assert summary.latency_sample_count == 0
    assert summary.latency_min_us is None
    assert summary.latency_median_us is None


def test_summary_reordered_input_is_normalized() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(3))
    cases = [
        sample_case(case_ordinal=0, frame_id=frame_ids[0], outcome=CASE_OUTCOME_EXACT),
        sample_case(
            case_ordinal=1,
            frame_id=frame_ids[1],
            outcome=CASE_OUTCOME_ACTION_EQUIVALENT,
        ),
        sample_case(
            case_ordinal=2, frame_id=frame_ids[2], outcome=CASE_OUTCOME_ACTION_CHANGING
        ),
    ]
    forwards = ProviderEvaluationSummaryDTO.from_cases(cases)
    backwards = ProviderEvaluationSummaryDTO.from_cases(list(reversed(cases)))
    assert forwards == backwards


def test_summary_rejects_fabricated_counts() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(2))
    cases = [
        sample_case(case_ordinal=0, frame_id=frame_ids[0], outcome=CASE_OUTCOME_EXACT),
        sample_case(
            case_ordinal=1,
            frame_id=frame_ids[1],
            outcome=CASE_OUTCOME_ACTION_EQUIVALENT,
        ),
    ]
    summary = ProviderEvaluationSummaryDTO.from_cases(cases)
    payload = summary.to_dict()
    payload["exact_count"] = 2
    with pytest.raises(
        VPMValidationError, match="summary accepted count mismatch|summary id mismatch"
    ):
        ProviderEvaluationSummaryDTO.from_dict(
            payload | {"summary_id": summary.summary_id}
        )


# --------------------------------------------------------------------------
# ProviderEvaluationRunDTO / MaterializedProviderEvaluationRunDTO
# --------------------------------------------------------------------------


def test_build_provider_evaluation_run_closure() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(8))
    cases = imperfect_fixture(frame_ids)
    materialized = build_provider_evaluation_run(
        fixture_identity="unit-test-imperfect",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="unlabelled",
        cases=cases,
    )
    assert materialized.run.case_ids == tuple(case.case_id for case in cases)
    assert materialized.summary.exact_count == 3
    assert materialized.summary.action_changing_count == 1


def test_run_round_trip() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(2))
    cases = exact_fixture(frame_ids[:2])
    materialized = build_provider_evaluation_run(
        fixture_identity="round-trip",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=cases,
    )
    assert (
        ProviderEvaluationRunDTO.from_dict(materialized.run.to_dict())
        == materialized.run
    )


def test_materialized_run_rejects_cases_from_another_run() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(2))
    cases_a = exact_fixture(frame_ids)
    cases_b = exact_fixture(tuple(f"development:y:frame-{i:02d}" for i in range(2)))
    run_a = build_provider_evaluation_run(
        fixture_identity="run-a",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=cases_a,
    ).run
    with pytest.raises(VPMValidationError, match="case identity mismatch"):
        MaterializedProviderEvaluationRunDTO(
            run=run_a,
            cases=tuple(cases_b),
            summary=ProviderEvaluationSummaryDTO.from_cases(cases_b),
        )


def test_materialized_run_rejects_unrelated_summary() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(2))
    cases = exact_fixture(frame_ids)
    materialized = build_provider_evaluation_run(
        fixture_identity="unrelated-summary",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=cases,
    )
    unrelated_summary = ProviderEvaluationSummaryDTO.from_cases(
        [
            sample_case(
                case_ordinal=0, frame_id=frame_ids[0], outcome=CASE_OUTCOME_REJECTED
            )
        ]
    )
    with pytest.raises(VPMValidationError, match="summary mismatch"):
        MaterializedProviderEvaluationRunDTO(
            run=materialized.run, cases=materialized.cases, summary=unrelated_summary
        )


def test_materialized_run_rejects_case_with_different_policy_artifact() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(2))
    configuration = sample_configuration()
    good_case = sample_case(
        case_ordinal=0,
        frame_id=frame_ids[0],
        provider_configuration_id=configuration.provider_configuration_id,
        policy_artifact_id=POLICY_ARTIFACT_ID,
        outcome=CASE_OUTCOME_EXACT,
    )
    bad_case = sample_case(
        case_ordinal=1,
        frame_id=frame_ids[1],
        provider_configuration_id=configuration.provider_configuration_id,
        policy_artifact_id=OTHER_POLICY_ARTIFACT_ID,
        outcome=CASE_OUTCOME_EXACT,
    )
    cases = [good_case, bad_case]
    summary = ProviderEvaluationSummaryDTO.from_cases(cases)
    payload = {
        "version": PROVIDER_EVALUATION_RUN_VERSION,
        "fixture_identity": "mixed-policy",
        "provider_configuration": configuration.to_dict(),
        "policy_artifact_id": POLICY_ARTIFACT_ID,
        "case_mode": "smoke",
        "representation_mode": "labelled",
        "case_ids": [c.case_id for c in cases],
        "summary": summary.to_dict(),
        "metadata": {},
    }
    run = ProviderEvaluationRunDTO.from_dict(
        payload | {"run_id": canonical_sha256(payload)}
    )
    with pytest.raises(VPMValidationError, match="policy artifact mismatch"):
        MaterializedProviderEvaluationRunDTO(
            run=run, cases=tuple(cases), summary=summary
        )


def test_materialized_run_rejects_noncontiguous_ordinals() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(2))
    cases = [
        sample_case(case_ordinal=0, frame_id=frame_ids[0], outcome=CASE_OUTCOME_EXACT),
        sample_case(case_ordinal=2, frame_id=frame_ids[1], outcome=CASE_OUTCOME_EXACT),
    ]
    payload = {
        "version": PROVIDER_EVALUATION_RUN_VERSION,
        "fixture_identity": "noncontiguous",
        "provider_configuration": sample_configuration().to_dict(),
        "policy_artifact_id": POLICY_ARTIFACT_ID,
        "case_mode": "smoke",
        "representation_mode": "labelled",
        "case_ids": [c.case_id for c in cases],
        "summary": ProviderEvaluationSummaryDTO.from_cases(cases).to_dict(),
        "metadata": {},
    }
    run = ProviderEvaluationRunDTO.from_dict(
        payload | {"run_id": canonical_sha256(payload)}
    )
    with pytest.raises(VPMValidationError, match="not contiguous"):
        MaterializedProviderEvaluationRunDTO(
            run=run,
            cases=tuple(cases),
            summary=ProviderEvaluationSummaryDTO.from_cases(cases),
        )


def test_materialized_run_rejects_missing_case() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(2))
    cases = exact_fixture(frame_ids)
    materialized = build_provider_evaluation_run(
        fixture_identity="missing-case",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=cases,
    )
    with pytest.raises(VPMValidationError, match="case count mismatch"):
        MaterializedProviderEvaluationRunDTO(
            run=materialized.run,
            cases=(cases[0],),
            summary=materialized.summary,
        )


def test_materialized_run_rejects_duplicate_case_id() -> None:
    frame_ids = tuple(f"development:x:frame-{i:02d}" for i in range(1))
    case = sample_case(
        case_ordinal=0, frame_id=frame_ids[0], outcome=CASE_OUTCOME_EXACT
    )
    payload = {
        "version": PROVIDER_EVALUATION_RUN_VERSION,
        "fixture_identity": "duplicate",
        "provider_configuration": sample_configuration().to_dict(),
        "policy_artifact_id": POLICY_ARTIFACT_ID,
        "case_mode": "smoke",
        "representation_mode": "labelled",
        "case_ids": [case.case_id, case.case_id],
        "summary": ProviderEvaluationSummaryDTO.from_cases([case, case]).to_dict(),
        "metadata": {},
    }
    with pytest.raises(VPMValidationError, match="duplicate"):
        ProviderEvaluationRunDTO.from_dict(
            payload | {"run_id": canonical_sha256(payload)}
        )


# --------------------------------------------------------------------------
# In-memory Store
# --------------------------------------------------------------------------


def test_store_save_and_load_materialized_run() -> None:
    store, frame_ids = store_with_observations(2)
    cases = exact_fixture(frame_ids)
    materialized = build_provider_evaluation_run(
        fixture_identity="store-test",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=cases,
    )
    saved = store.save_provider_evaluation_run(materialized)
    assert saved == materialized
    reloaded = store.get_materialized_provider_evaluation_run(saved.run.run_id)
    assert reloaded == materialized
    assert store.get_provider_evaluation_run(saved.run.run_id) == materialized.run


def test_store_save_is_idempotent() -> None:
    store, frame_ids = store_with_observations(1)
    cases = exact_fixture(frame_ids)
    materialized = build_provider_evaluation_run(
        fixture_identity="idempotent",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=cases,
    )
    first = store.save_provider_evaluation_run(materialized)
    second = store.save_provider_evaluation_run(materialized)
    assert first == second


def test_store_rejects_unknown_observation() -> None:
    store, frame_ids = store_with_observations(1)
    bad_case = sample_case(
        case_ordinal=0, frame_id="does-not-exist", outcome=CASE_OUTCOME_REJECTED
    )
    materialized = build_provider_evaluation_run(
        fixture_identity="unknown-obs",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=[bad_case],
    )
    with pytest.raises(
        VPMValidationError, match=UNKNOWN_OBSERVATION_FOR_PROVIDER_EVALUATION_MESSAGE
    ):
        store.save_provider_evaluation_run(materialized)


def test_store_rejects_case_shared_across_two_runs() -> None:
    """A case belongs to exactly one run. Since `case_ordinal` participates in
    `case_id`'s digest, byte-identical cases under two different runs can only
    happen if the same content is deliberately reused - the store must always
    reject that as a conflict, never silently dedup it (unlike matrix blobs or
    provider configurations, which are legitimately content-addressed and
    shared)."""
    store, frame_ids = store_with_observations(1)
    configuration = sample_configuration()
    shared_case = ProviderEvaluationCaseDTO.build(
        case_ordinal=0,
        frame_id=frame_ids[0],
        context=ProviderEvaluationCaseContext(
            policy_artifact_id=POLICY_ARTIFACT_ID,
            provider_configuration_id=configuration.provider_configuration_id,
        ),
        expected_state={"tank_column": 0},
        expected_decision=sample_decision("r0", "STAY"),
        accepted=True,
        predicted_state={"tank_column": 0},
        predicted_decision=sample_decision("r0", "STAY"),
    )
    run_a = build_provider_evaluation_run(
        fixture_identity="run-a",
        provider_configuration=configuration,
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=[shared_case],
    )
    store.save_provider_evaluation_run(run_a)

    run_b = build_provider_evaluation_run(
        fixture_identity="run-b-different-fixture",
        provider_configuration=configuration,
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=[shared_case],
    )
    assert run_b.run.run_id != run_a.run.run_id
    with pytest.raises(
        VPMValidationError, match=PROVIDER_EVALUATION_CASE_CONFLICT_MESSAGE
    ):
        store.save_provider_evaluation_run(run_b)


def test_store_stable_listing_and_filters() -> None:
    store, frame_ids = store_with_observations(8)
    cases = imperfect_fixture(frame_ids)
    materialized = build_provider_evaluation_run(
        fixture_identity="filters",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="unlabelled",
        cases=cases,
    )
    saved = store.save_provider_evaluation_run(materialized)

    runs = store.list_provider_evaluation_runs(provider_kind="fake")
    assert len(runs) == 1
    assert store.list_provider_evaluation_runs(provider_kind="ollama") == ()
    assert store.list_provider_evaluation_runs(case_mode="smoke") == (saved.run,)
    assert store.list_provider_evaluation_runs(representation_mode="unlabelled") == (
        saved.run,
    )
    assert store.list_provider_evaluation_runs(representation_mode="labelled") == ()

    all_cases = store.list_provider_evaluation_cases(run_id=saved.run.run_id)
    assert [case.case_ordinal for case in all_cases] == list(range(8))
    exact_cases = store.list_provider_evaluation_cases(
        run_id=saved.run.run_id, outcome=CASE_OUTCOME_EXACT
    )
    assert len(exact_cases) == 3
    changing_cases = store.list_provider_evaluation_cases(
        run_id=saved.run.run_id, outcome=CASE_OUTCOME_ACTION_CHANGING
    )
    assert len(changing_cases) == 1
    accepted_cases = store.list_provider_evaluation_cases(
        run_id=saved.run.run_id, accepted=True
    )
    assert len(accepted_cases) == 8


def test_store_get_missing_run_returns_none() -> None:
    store = InMemoryVideoActionSetStore()
    assert store.get_provider_evaluation_run("sha256:" + "f" * 64) is None
    assert store.get_materialized_provider_evaluation_run("sha256:" + "f" * 64) is None


# --------------------------------------------------------------------------
# Runtime wiring
# --------------------------------------------------------------------------


def test_runtime_facade_provider_evaluation_methods_share_one_store() -> None:
    from zeromodel.video.runtime import build_runtime

    runtime = build_runtime()
    facade = runtime.video_action_set
    assert facade.list_provider_evaluation_runs() == ()
