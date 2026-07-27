import numpy as np

from visual_transition_benchmark import baselines as bl
from visual_transition_benchmark import dataset as ds


def test_pixel_diff_detects_a_visible_mutation():
    record = ds.build_transition(
        episode_id="e", step_number=0, seed=1, category="tank_moves_left"
    )
    output = bl.pixel_diff_baseline(record.frame_before, record.frame_after)
    assert output.predicted_region_mask.any()
    assert "tank" in output.predicted_components


def test_pixel_diff_cannot_represent_an_absent_expected_transition():
    # expected_target_remains_unchanged: tank should move but the frames are
    # rendered identically. Pixel diff has no way to flag the absence.
    record = ds.build_transition(
        episode_id="e",
        step_number=0,
        seed=1,
        category="expected_target_remains_unchanged",
    )
    assert np.array_equal(record.frame_before, record.frame_after)
    output = bl.pixel_diff_baseline(record.frame_before, record.frame_after)
    assert not output.predicted_region_mask.any()
    assert output.missing_components == ()  # structurally cannot express "missing"


def test_privileged_baseline_matches_declared_component_changes():
    for category in ds.ALL_CATEGORIES:
        record = ds.build_transition(
            episode_id="e", step_number=0, seed=2, category=category
        )
        output = bl.privileged_baseline(record)
        assert set(output.predicted_components) == set(
            record.observed_changed_components
        )
        assert output.missing_components == tuple(
            sorted(
                set(record.expected_changed_components)
                - set(record.observed_changed_components)
            )
        )
        assert output.unexpected_components == tuple(
            sorted(
                set(record.observed_changed_components)
                - set(record.expected_changed_components)
            )
        )
