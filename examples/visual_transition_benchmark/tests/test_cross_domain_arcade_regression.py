"""Proves the arcade domain wrapper changes nothing observable.

Every frame, category, fault flag, and analyzer verdict produced through
``ArcadeTransitionDomain`` must be byte-for-byte identical to what stage 1/2's
own pipeline produces directly. If this test ever fails, the wrapper has
started doing something the original benchmark did not.
"""

import numpy as np

from visual_transition_benchmark import dataset as ds
from visual_transition_benchmark import value_metrics as vm
from visual_transition_benchmark import zeromodel_adapter as zm
from visual_transition_benchmark.domains.arcade.domain import ArcadeTransitionDomain
from visual_transition_benchmark.domains.protocol import AnalysisMetadata
from visual_transition_benchmark.value_adapter import ValueAwareZeroModelAnalyzer


def test_generated_transitions_are_bit_for_bit_identical():
    domain = ArcadeTransitionDomain()
    for seed in (0, 1, 7, 42):
        episode_id = f"regress-{seed:04d}"
        original = ds.generate_episode(episode_id, seed)
        wrapped = domain.generate_episode(seed=seed, episode_id=episode_id)
        assert len(original) == len(wrapped)
        for record, transition in zip(original, wrapped):
            assert transition.transition_id == record.transition_id
            assert transition.category == record.category
            assert transition.action == record.action
            assert transition.fault_type == record.fault_type
            assert transition.is_faulty == record.is_faulty
            assert transition.expected_changed_components == record.expected_changed_components
            assert transition.observed_changed_components == record.observed_changed_components
            assert np.array_equal(transition.frame_before, record.frame_before)
            assert np.array_equal(transition.frame_after, record.frame_after)


def test_component_analysis_matches_original_analyzer_exactly():
    domain = ArcadeTransitionDomain()
    original_analyzer = zm.ArcadeBandZeroModelAnalyzer()
    wrapped_analyzer = domain.build_component_analyzer()

    for category in ds.ALL_CATEGORIES:
        record = ds.build_transition(episode_id="regress-c", step_number=0, seed=3, category=category)
        original = original_analyzer.analyze(
            record.frame_before,
            record.frame_after,
            record.action,
            zm.TransitionMetadata(transition_id=record.transition_id, step_number=record.step_number),
        )
        wrapped = wrapped_analyzer.analyze(
            record.frame_before,
            record.frame_after,
            record.action,
            AnalysisMetadata(transition_id=record.transition_id, step_number=record.step_number),
        )
        assert wrapped.predicted_components == original.predicted_components
        assert wrapped.missing_components == original.missing_components
        assert wrapped.unexpected_components == original.unexpected_components
        assert wrapped.predicted_fields == original.predicted_fields
        assert np.array_equal(wrapped.predicted_region_mask, original.predicted_region_mask)
        assert wrapped.evidence_scores == original.evidence_scores


def test_value_analysis_matches_original_analyzer_exactly():
    domain = ArcadeTransitionDomain()
    original_analyzer = ValueAwareZeroModelAnalyzer()
    wrapped_analyzer = domain.build_value_analyzer()

    for category in ds.ALL_CATEGORIES + ds.VALUE_FAULT_CATEGORIES:
        if category in ds.VALUE_FAULT_CATEGORIES:
            record = ds.build_value_transition(episode_id="regress-v", step_number=0, seed=5, category=category)
        else:
            record = ds.build_transition(episode_id="regress-v", step_number=0, seed=5, category=category)
        original = original_analyzer.analyze(
            record.frame_before,
            record.frame_after,
            record.action,
            zm.TransitionMetadata(transition_id=record.transition_id, step_number=record.step_number),
        )
        wrapped = wrapped_analyzer.analyze(
            record.frame_before,
            record.frame_after,
            record.action,
            AnalysisMetadata(transition_id=record.transition_id, step_number=record.step_number),
        )
        assert wrapped.value_flags == original.value_flags
        assert wrapped.decoded["magnitude_decoded_delta"] == original.values.tank.delta_x
        assert wrapped.decoded["value_decoded_level"] == original.values.cooldown.after_level


def test_value_ground_truth_matches_value_metrics_functions():
    domain = ArcadeTransitionDomain()
    for category in ds.ALL_CATEGORIES:
        record = ds.build_transition(episode_id="regress-g", step_number=0, seed=9, category=category)
        transition = domain._to_domain_transition(record)
        expected_sign = -1 if vm.true_tank_delta(record) < 0 else (1 if vm.true_tank_delta(record) > 0 else 0)
        assert transition.value_ground_truth["direction_expected_sign"] == expected_sign
        assert transition.value_ground_truth["magnitude_expected_delta"] == vm.true_tank_delta(record)
        assert transition.value_ground_truth["value_expected_level"] == vm.true_cooldown_level(record)


def test_stage1_metrics_functions_run_unchanged_over_wrapped_transitions():
    """The whole point of matching TransitionRecord's field names: stage 1's
    existing metrics.py functions must run, unmodified, on DomainTransition
    objects too."""

    from visual_transition_benchmark import metrics as mx

    domain = ArcadeTransitionDomain()
    transitions = domain.generate_episode(seed=11, episode_id="regress-metrics")
    analyzer = domain.build_component_analyzer()
    outputs = [
        analyzer.analyze(
            t.frame_before, t.frame_after, t.action, AnalysisMetadata(t.transition_id, t.step_number)
        )
        for t in transitions
    ]
    result = mx.component_multilabel_metrics(
        [o.predicted_components for o in outputs], [t.observed_changed_components for t in transitions]
    )
    assert result["n"] == len(transitions)
    assert 0.0 <= result["micro_f1"] <= 1.0
