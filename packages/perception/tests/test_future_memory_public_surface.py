"""Public-surface coverage for structured future memory (no governance stack)."""

from __future__ import annotations

import zeromodel.perception as perception


def test_future_memory_is_exposed_from_package_root() -> None:
    expected = {
        "PerceptionExpectedTransitionError",
        "ExpectedTransitionFieldDTO",
        "ExpectedTransitionVPMDTO",
        "render_expected_transition_png",
        "EXPECTED_TRANSITION_FIELD_VERSION",
        "EXPECTED_TRANSITION_VPM_VERSION",
        "EXPECTED_TRANSITION_RENDER_SEMANTICS",
        "EXPECTED_TRANSITION_STATUSES",
        "PerceptionTransitionModelError",
        "TransitionModelConfigDTO",
        "TransitionTrainingExampleDTO",
        "EmpiricalTransitionModelDTO",
        "CompiledTransitionModelDTO",
        "TransitionModelDTO",
        "fit_action_conditioned_transition_model",
        "fit_compiled_transition_model",
        "TRANSITION_PROJECTION_DISTANCE_SEMANTICS",
        "TRANSITION_PROJECTION_WEIGHTED_DISTANCE_SEMANTICS",
        "project_expected_transition",
        "TRANSITION_MODEL_CONFIG_VERSION",
        "TRANSITION_TRAINING_EXAMPLE_VERSION",
        "EMPIRICAL_TRANSITION_MODEL_VERSION",
        "COMPILED_TRANSITION_MODEL_VERSION",
        "PerceptionWorldActionError",
        "WorldActionPolicyDTO",
        "WorldActionCandidateDTO",
        "CoupledActionPredictionDTO",
        "ExpectedConformanceDTO",
        "ExpectedConformanceFindingDTO",
        "check_expected_conformance",
        "predict_action_with_future_memory",
        "WORLD_ACTION_POLICY_VERSION",
        "WORLD_ACTION_CANDIDATE_STATUSES",
        "EXPECTED_CONFORMANCE_STATUSES",
        "PerceptionTransitionVerificationError",
        "FutureTransitionVerificationDTO",
        "verify_expected_transition",
        "FUTURE_TRANSITION_VERIFICATION_STATUSES",
        "PerceptionSharedRelevanceError",
        "SharedFieldRelevanceDTO",
        "fit_shared_field_relevance",
        "predict_relevance_weighted_action",
        "SHARED_RELEVANCE_VERSION",
        "SHARED_RELEVANCE_FUTURE_SEMANTICS",
    }

    assert expected <= set(perception.__all__)
    for name in expected:
        assert getattr(perception, name) is not None
