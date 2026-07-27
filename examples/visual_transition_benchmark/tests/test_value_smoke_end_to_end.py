from pathlib import Path

from visual_transition_benchmark import baselines as bl
from visual_transition_benchmark import dataset as ds
from visual_transition_benchmark import report as rp
from visual_transition_benchmark import value_metrics as vm
from visual_transition_benchmark import zeromodel_adapter as zm
from visual_transition_benchmark.value_adapter import ValueAwareZeroModelAnalyzer


def test_small_deterministic_value_dataset_end_to_end(tmp_path: Path):
    original = ds.generate_split(prefix="v-smoke-orig", episode_count=1, seed_offset=0)
    value_new = ds.generate_value_split(
        prefix="v-smoke-new", episode_count=2, seed_offset=500
    )
    ds.assert_disjoint_splits(original, value_new)

    records = original.records + value_new.records
    assert len(records) == len(ds.ALL_CATEGORIES) + 2 * len(ds.VALUE_FAULT_CATEGORIES)

    analyzer = ValueAwareZeroModelAnalyzer()
    analyses, pd_outputs, priv_outputs = [], [], []
    for record in records:
        metadata = zm.TransitionMetadata(
            transition_id=record.transition_id, step_number=record.step_number
        )
        analyses.append(
            analyzer.analyze(
                record.frame_before, record.frame_after, record.action, metadata
            )
        )
        pd_outputs.append(
            bl.pixel_diff_baseline(record.frame_before, record.frame_after)
        )
        priv_outputs.append(bl.privileged_baseline(record))

    values_list = [a.values for a in analyses]
    flags_list = [a.value_flags for a in analyses]
    component_outputs = [a.component_analysis for a in analyses]

    # Stage-1 metrics still compute over the combined set (unchanged mechanism).
    component_report = rp.score_group(
        records, component_outputs, pd_outputs, priv_outputs
    )
    assert component_report["n"] == len(records)

    # Stage-2 metrics.
    accuracy = vm.value_accuracy_summary(records, values_list)
    assert accuracy.n == len(records)
    localization = vm.value_fault_localization_summary(records, values_list, flags_list)
    assert 0.0 <= localization.detection_rate <= 1.0
    hidden = vm.label_correct_but_value_wrong(records, component_outputs, values_list)
    assert hidden.n_faulty > 0
    # The direction fix must show up in this small sample too.
    direction_record = next(
        r for r in records if r.category == "tank_moves_wrong_direction"
    )
    direction_index = records.index(direction_record)
    assert values_list[direction_index].tank.delta_x is not None
    assert not vm.tank_direction_correct(direction_record, values_list[direction_index])

    results_path = tmp_path / "value-benchmark-results.json"
    rp.write_json(
        results_path,
        {"accuracy": accuracy.__dict__, "localization": localization.__dict__},
    )
    assert results_path.exists()
