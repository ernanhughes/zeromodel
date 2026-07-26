import numpy as np
import pytest

from visual_transition_benchmark import dataset as ds


def test_same_seed_produces_identical_transition_records():
    a = ds.generate_episode("dev-x", seed=7)
    b = ds.generate_episode("dev-x", seed=7)
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra.action == rb.action
        assert ra.category == rb.category
        assert ra.state_before == rb.state_before
        assert ra.state_after == rb.state_after
        assert ra.expected_changed_components == rb.expected_changed_components
        assert ra.observed_changed_components == rb.observed_changed_components
        assert np.array_equal(ra.frame_before, rb.frame_before)
        assert np.array_equal(ra.frame_after, rb.frame_after)


def test_different_episodes_do_not_cross_splits():
    dev = ds.generate_split(prefix="dev", episode_count=4, seed_offset=0)
    eva = ds.generate_split(prefix="eval", episode_count=4, seed_offset=1000)
    ds.assert_disjoint_splits(dev, eva)  # must not raise

    dev2 = ds.generate_split(prefix="dev", episode_count=2, seed_offset=0)
    with pytest.raises(ds.DatasetError):
        ds.assert_disjoint_splits(dev, dev2)


def test_frame_and_state_annotations_agree():
    for category in ds.ALL_CATEGORIES:
        record = ds.build_transition(episode_id="agree", step_number=0, seed=3, category=category)
        for name in ds.COMPONENT_NAMES:
            mask = record.component_annotations[name]
            assert mask.shape == record.frame_before.shape
        union = np.zeros_like(record.component_annotations["tank"])
        for name in ds.COMPONENT_NAMES:
            union |= record.component_annotations[name]
        assert union.all(), "component annotations must exactly partition the canvas"


def test_component_annotations_are_a_strict_partition():
    record = ds.build_transition(
        episode_id="partition", step_number=0, seed=1, category="tank_moves_left"
    )
    names = ds.COMPONENT_NAMES
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = record.component_annotations[a] & record.component_annotations[b]
            assert not overlap.any(), f"{a} and {b} overlap"


def test_fault_injection_changes_only_declared_targets():
    # background_changes_unexpectedly must not touch tank/alien/cooldown pixels.
    record = ds.build_transition(
        episode_id="fault-scope", step_number=0, seed=5, category="background_changes_unexpectedly"
    )
    assert record.observed_changed_components == ("background",)
    row, col = ds.BACKGROUND_PROBE_PIXEL
    assert record.frame_after[row, col] == ds.BACKGROUND_PROBE_VALUE
    assert record.frame_before[row, col] == 0


def test_ordinary_transitions_contain_no_injected_fault():
    for category in ds.ORDINARY_CATEGORIES:
        record = ds.build_transition(episode_id="ordinary", step_number=0, seed=9, category=category)
        assert record.is_faulty is False
        assert record.fault_type is None
        # ordinary transitions render exactly the true post-state: expected == observed.
        assert record.expected_changed_components == record.observed_changed_components


def test_all_categories_are_constructible_across_many_seeds():
    for seed in range(10):
        for category in ds.ALL_CATEGORIES:
            record = ds.build_transition(
                episode_id=f"seed-{seed}", step_number=0, seed=seed, category=category
            )
            assert record.transition_id
            assert record.frame_before.shape == record.frame_after.shape
