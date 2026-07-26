from visual_transition_benchmark import metrics as mx


def test_field_precision_recall_perfect_prediction():
    truth = ("f1", "f2", "f3")
    precision, recall = mx.field_precision_recall(truth, truth)
    assert precision == 1.0
    assert recall == 1.0


def test_field_precision_recall_empty_prediction_has_zero_recall():
    truth = ("f1", "f2")
    precision, recall = mx.field_precision_recall((), truth)
    assert recall == 0.0
    # convention (documented in metrics.py): an empty prediction against a
    # non-empty truth set scores precision 0.0, matching scikit-learn's
    # zero_division=0 default for an empty predicted-positive set.
    assert precision == 0.0


def test_field_precision_recall_empty_prediction_and_empty_truth_is_trivially_correct():
    precision, recall = mx.field_precision_recall((), ())
    assert precision == 1.0
    assert recall == 1.0


def test_component_metrics_predicting_everything_hurts_precision():
    universe = ("tank", "alien", "cooldown", "background")
    predicted = [universe] * 4
    truth = [("tank",), ("alien",), (), ("cooldown",)]
    result = mx.component_multilabel_metrics(predicted, truth)
    assert result["micro_recall"] == 1.0
    assert result["micro_precision"] < 0.5
    assert result["exact_set_accuracy"] == 0.0


def test_component_metrics_perfect_prediction():
    truth = [("tank",), (), ("alien", "cooldown")]
    result = mx.component_multilabel_metrics(truth, truth)
    assert result["micro_precision"] == 1.0
    assert result["micro_recall"] == 1.0
    assert result["micro_f1"] == 1.0
    assert result["exact_set_accuracy"] == 1.0


class _FakeRecord:
    def __init__(self, expected, observed, is_faulty=True):
        self.expected_changed_components = expected
        self.observed_changed_components = observed
        self.is_faulty = is_faulty


class _FakeOutput:
    def __init__(self, predicted=(), missing=(), unexpected=()):
        self.predicted_components = predicted
        self.missing_components = missing
        self.unexpected_components = unexpected


def test_missing_expected_change_is_counted_correctly():
    # expected tank change, nothing observed -> a real missing-change case.
    record = _FakeRecord(expected=("tank",), observed=())
    assert mx.missing_ground_truth(record) == frozenset({"tank"})

    hit_output = _FakeOutput(missing=("tank",))
    miss_output = _FakeOutput(missing=())
    hit_summary = mx.missing_change_summary([record], [hit_output])
    miss_summary = mx.missing_change_summary([record], [miss_output])
    assert hit_summary.detection_rate == 1.0
    assert miss_summary.detection_rate == 0.0


def test_unexpected_extra_change_is_counted_correctly():
    # nothing expected, but background observed changed -> real unexpected case.
    record = _FakeRecord(expected=(), observed=("background",))
    assert mx.unexpected_ground_truth(record) == frozenset({"background"})

    hit_output = _FakeOutput(predicted=("background",), unexpected=("background",))
    miss_output = _FakeOutput(predicted=("background",), unexpected=())
    hit_summary = mx.unexpected_change_summary([record], [hit_output])
    miss_summary = mx.unexpected_change_summary([record], [miss_output])
    assert hit_summary.recall == 1.0
    assert miss_summary.recall == 0.0


def test_false_implicated_counts_only_components_that_did_not_change():
    record = _FakeRecord(expected=("tank",), observed=("tank",), is_faulty=False)
    output = _FakeOutput(predicted=("tank", "background"))
    implicated = mx.false_implicated_components(record, output)
    assert implicated == frozenset({"background"})
