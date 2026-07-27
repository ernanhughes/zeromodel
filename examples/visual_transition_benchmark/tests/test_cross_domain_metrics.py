from visual_transition_benchmark import cross_domain_metrics as cdm
from visual_transition_benchmark.domains.protocol import (
    DomainTransition,
    ValueAnalysisResult,
)


def _transition(**ground_truth) -> DomainTransition:
    return DomainTransition(
        transition_id="t",
        domain_name="fake",
        episode_id="e",
        step_number=0,
        seed=0,
        action="A",
        category="c",
        frame_before=None,
        frame_after=None,
        expected_changed_components=(),
        observed_changed_components=(),
        fault_type=None,
        is_faulty=True,
        expected_contracts=(),
        value_ground_truth=ground_truth,
    )


def _analysis(decoded, flags=()) -> ValueAnalysisResult:
    return ValueAnalysisResult(decoded=decoded, value_flags=flags, diagnostics={})


def test_direction_and_magnitude_not_applicable_when_no_ground_truth():
    transition = _transition()
    analysis = _analysis({})
    assert cdm.direction_correct(transition, analysis.decoded) is None
    assert cdm.magnitude_correct(transition, analysis.decoded) is None


def test_direction_correct_and_incorrect():
    transition = _transition(direction_expected_sign=-1)
    assert cdm.direction_correct(transition, {"direction_decoded_sign": -1}) is True
    assert cdm.direction_correct(transition, {"direction_decoded_sign": 1}) is False


def test_value_level_correct_handles_multiple_named_channels_generically():
    transition = _transition(value_expected_level="ready", door_expected_level="closed")
    ok = {"value_decoded_level": "ready", "door_decoded_level": "closed"}
    bad = {"value_decoded_level": "ready", "door_decoded_level": "open"}
    assert cdm.value_level_correct(transition, ok) is True
    assert cdm.value_level_correct(transition, bad) is False


def test_identity_correct_is_not_applicable_without_ground_truth():
    transition = _transition()
    assert cdm.identity_correct(transition, {"identity_decoded_id": 0}) is None


def test_value_fault_present_and_detection_rate():
    faulty = _transition(direction_expected_sign=-1)
    clean = _transition(direction_expected_sign=-1)
    faulty_analysis = _analysis(
        {"direction_decoded_sign": 1}, flags=("robot_direction_violation",)
    )
    clean_analysis = _analysis({"direction_decoded_sign": -1}, flags=())

    assert cdm.value_fault_present(faulty, faulty_analysis.decoded) is True
    assert cdm.value_fault_present(clean, clean_analysis.decoded) is False

    rate = cdm.value_fault_detection([faulty, clean], [faulty_analysis, clean_analysis])
    assert rate.n_relevant == 1
    assert rate.detection_rate == 1.0
    assert rate.n_clean == 1
    assert rate.false_alarm_rate_on_correct == 0.0


class _FakeComponentOutput:
    def __init__(self, missing=(), unexpected=()):
        self.missing_components = missing
        self.unexpected_components = unexpected


def test_label_correct_but_value_wrong():
    transition = _transition(direction_expected_sign=-1)
    component = _FakeComponentOutput()
    value = _analysis({"direction_decoded_sign": 1})
    result = cdm.label_correct_but_value_wrong([transition], [component], [value])
    assert result.n_faulty == 1
    assert result.label_clean_but_value_wrong == 1
