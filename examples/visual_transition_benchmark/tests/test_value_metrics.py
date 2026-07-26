from visual_transition_benchmark import value_metrics as vm
from visual_transition_benchmark.value_contracts import AlienValues, CooldownValues, DecodedValues, TankValues


class _FakeRecord:
    def __init__(self, tank_before, tank_after, cooldown_after, target_after, is_faulty=True):
        self.state_before = {"tank_x": tank_before}
        self.state_after = {"tank_x": tank_after, "cooldown": cooldown_after, "target_x": target_after}
        self.is_faulty = is_faulty


def _values(tank_delta, cooldown_level, target_after, tank_before=0):
    return DecodedValues(
        tank=TankValues(before_x=tank_before, after_x=tank_before + tank_delta, delta_x=tank_delta),
        alien=AlienValues(before_alive=True, after_alive=target_after is not None, before_x=0, after_x=target_after),
        cooldown=CooldownValues(before_intensity=0.0, after_intensity=0.0, before_level="ready", after_level=cooldown_level),
    )


def test_accuracy_perfect_match():
    record = _FakeRecord(tank_before=2, tank_after=1, cooldown_after=1, target_after=5)
    values = _values(tank_delta=-1, cooldown_level="blocked", target_after=5, tank_before=2)
    summary = vm.value_accuracy_summary([record], [values])
    assert summary.movement_direction_accuracy == 1.0
    assert summary.state_delta_accuracy == 1.0
    assert summary.cooldown_value_accuracy == 1.0
    assert summary.target_selection_accuracy == 1.0


def test_accuracy_wrong_direction_fails_both_direction_and_delta():
    record = _FakeRecord(tank_before=2, tank_after=1, cooldown_after=0, target_after=None)
    values = _values(tank_delta=1, cooldown_level="ready", target_after=None, tank_before=2)  # moved +1, true was -1
    summary = vm.value_accuracy_summary([record], [values])
    assert summary.movement_direction_accuracy == 0.0
    assert summary.state_delta_accuracy == 0.0


def test_value_fault_present_flags_any_wrong_dimension():
    record = _FakeRecord(tank_before=2, tank_after=2, cooldown_after=1, target_after=5)
    correct = _values(tank_delta=0, cooldown_level="blocked", target_after=5, tank_before=2)
    wrong_cooldown = _values(tank_delta=0, cooldown_level="out_of_domain", target_after=5, tank_before=2)
    assert not vm.value_fault_present(record, correct)
    assert vm.value_fault_present(record, wrong_cooldown)


def test_fault_localization_detects_and_does_not_false_alarm():
    faulty_record = _FakeRecord(tank_before=0, tank_after=0, cooldown_after=1, target_after=5)
    faulty_values = _values(tank_delta=0, cooldown_level="out_of_domain", target_after=5, tank_before=0)
    clean_record = _FakeRecord(tank_before=0, tank_after=1, cooldown_after=0, target_after=None, is_faulty=False)
    clean_values = _values(tank_delta=1, cooldown_level="ready", target_after=None, tank_before=0)

    hit_flags = [("cooldown_value_violation",), ()]
    summary = vm.value_fault_localization_summary(
        [faulty_record, clean_record], [faulty_values, clean_values], hit_flags
    )
    assert summary.n_relevant == 1
    assert summary.detection_rate == 1.0
    assert summary.n_clean == 1
    assert summary.false_alarm_rate_on_correct == 0.0

    miss_flags = [(), ()]
    miss_summary = vm.value_fault_localization_summary(
        [faulty_record, clean_record], [faulty_values, clean_values], miss_flags
    )
    assert miss_summary.detection_rate == 0.0


class _FakeComponentOutput:
    def __init__(self, missing=(), unexpected=()):
        self.missing_components = missing
        self.unexpected_components = unexpected


def test_label_correct_but_value_wrong_counts_the_hidden_failure_mode():
    record = _FakeRecord(tank_before=2, tank_after=1, cooldown_after=0, target_after=None, is_faulty=True)
    # component layer sees nothing wrong (label-correct)...
    component_output = _FakeComponentOutput(missing=(), unexpected=())
    # ...but the value layer disagrees with ground truth (wrong direction: true delta -1, decoded +1)
    wrong_values = _values(tank_delta=1, cooldown_level="ready", target_after=None, tank_before=2)

    summary = vm.label_correct_but_value_wrong([record], [component_output], [wrong_values])
    assert summary.n_faulty == 1
    assert summary.label_clean_but_value_wrong == 1
    assert summary.rate == 1.0
