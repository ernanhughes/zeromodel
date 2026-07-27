import numpy as np

from visual_transition_benchmark.domains.warehouse import faults as wf
from visual_transition_benchmark.domains.warehouse import model as wm


def test_same_seed_produces_identical_transitions():
    a = wf.generate_episode("wh-x", seed=7)
    b = wf.generate_episode("wh-x", seed=7)
    for ra, rb in zip(a, b):
        assert ra.state_before == rb.state_before
        assert ra.state_after == rb.state_after
        assert np.array_equal(ra.frame_before, rb.frame_before)
        assert np.array_equal(ra.frame_after, rb.frame_after)


def test_split_disjointness():
    dev = wf.generate_split(prefix="wh-dev", episode_count=3, seed_offset=0)
    eval_split = wf.generate_split(prefix="wh-eval", episode_count=3, seed_offset=1000)
    assert set(dev.episode_ids).isdisjoint(set(eval_split.episode_ids))


def test_ordinary_transitions_are_label_correct():
    for category in wf.ORDINARY_CATEGORIES:
        for seed in range(20):
            record = wf.build_transition(
                episode_id="e", step_number=0, seed=seed, category=category
            )
            assert not record.is_faulty
            assert record.fault_type is None
            assert (
                record.expected_changed_components == record.observed_changed_components
            )


def test_value_and_identity_faults_are_label_correct():
    # The whole point of the value/identity fault families: they must look
    # entirely correct at the component-presence level.
    label_correct_families = wf.VALUE_FAULT_CATEGORIES + wf.IDENTITY_FAULT_CATEGORIES
    for category in label_correct_families:
        record = wf.build_transition(
            episode_id="e", step_number=0, seed=3, category=category
        )
        assert record.is_faulty
        assert set(record.observed_changed_components) == set(
            record.expected_changed_components
        ), f"{category} should be label-correct"


def test_push_based_relation_faults_are_label_correct():
    # Relation faults built on top of a legitimate push are label-correct
    # (robot+crate both show plausible changes); crate_moves_without_robot_
    # adjacency is built on WAIT instead, so it is presence-catchable too --
    # see test_presence_faults_are_not_label_correct's sibling assertions.
    for category in (
        "push_advances_robot_without_crate",
        "two_crates_move_during_single_push",
    ):
        record = wf.build_transition(
            episode_id="e", step_number=0, seed=3, category=category
        )
        assert record.is_faulty
        assert set(record.observed_changed_components) == set(
            record.expected_changed_components
        )


def test_presence_faults_are_not_label_correct():
    for category in wf.PRESENCE_FAULT_CATEGORIES:
        record = wf.build_transition(
            episode_id="e", step_number=0, seed=3, category=category
        )
        assert set(record.observed_changed_components) != set(
            record.expected_changed_components
        ), f"{category} should be visibly different from expected at the presence level"


def test_all_categories_constructible_across_many_seeds():
    for seed in range(30):
        for category in wf.ALL_CATEGORIES:
            record = wf.build_transition(
                episode_id=f"seed-{seed}", step_number=0, seed=seed, category=category
            )
            assert record.transition_id
            assert record.frame_before.shape == record.frame_after.shape


def test_battery_and_robot_bounds_never_violated():
    for seed in range(30):
        for category in wf.ALL_CATEGORIES:
            record = wf.build_transition(
                episode_id=f"bounds-{seed}", step_number=0, seed=seed, category=category
            )
            assert 0 <= record.state_after["battery"] <= wm.MAX_BATTERY
            row, col = record.state_after["robot"]
            assert (
                not wm.is_wall((row, col)) or record.category == "robot_blocked_by_wall"
            )
