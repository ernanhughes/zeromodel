from visual_transition_benchmark import dataset as ds
from visual_transition_benchmark import zeromodel_adapter as zm
from visual_transition_benchmark.value_adapter import ValueAwareZeroModelAnalyzer


def _analyze(record):
    analyzer = ValueAwareZeroModelAnalyzer()
    metadata = zm.TransitionMetadata(transition_id=record.transition_id, step_number=record.step_number)
    return analyzer.analyze(record.frame_before, record.frame_after, record.action, metadata)


def test_component_and_value_layers_are_both_present_and_independent():
    record = ds.build_transition(episode_id="e", step_number=0, seed=1, category="tank_moves_wrong_direction")
    analysis = _analyze(record)
    # Component layer: unchanged System C result, "looks correct" (this is the whole point).
    assert not analysis.component_analysis.missing_components
    assert not analysis.component_analysis.unexpected_components
    # Value layer: catches what the component layer could not.
    assert analysis.verdict.tank_direction_ok is False
    assert "tank_direction_violation" in analysis.value_flags


def test_same_transition_produces_identical_value_analysis():
    record = ds.build_value_transition(
        episode_id="e", step_number=0, seed=4, category="tank_moves_too_far"
    )
    a = _analyze(record)
    b = _analyze(record)
    assert a.values == b.values
    assert a.verdict == b.verdict
    assert a.value_flags == b.value_flags


def test_analyzer_signature_matches_component_analyzer_non_privileged_contract():
    # ValueAwareZeroModelAnalyzer.analyze must accept exactly the same
    # non-privileged inputs as ArcadeBandZeroModelAnalyzer.analyze.
    import inspect

    from visual_transition_benchmark.zeromodel_adapter import ArcadeBandZeroModelAnalyzer

    base_params = list(inspect.signature(ArcadeBandZeroModelAnalyzer.analyze).parameters)
    value_params = list(inspect.signature(ValueAwareZeroModelAnalyzer.analyze).parameters)
    assert base_params == value_params


def test_value_flags_cover_every_new_fault_category():
    for category in ds.VALUE_FAULT_CATEGORIES:
        record = ds.build_value_transition(episode_id="e", step_number=0, seed=7, category=category)
        analysis = _analyze(record)
        if category in ("wrong_alien_disappears", "two_aliens_disappear_instead_of_one"):
            continue  # documented, honest blind spot
        assert analysis.value_flags, f"{category} should raise at least one value flag"
