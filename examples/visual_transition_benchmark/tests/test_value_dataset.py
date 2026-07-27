from visual_transition_benchmark import dataset as ds


def test_stage_one_categories_are_untouched():
    assert len(ds.ORDINARY_CATEGORIES) == 8
    assert len(ds.FAULT_CATEGORIES) == 10
    assert "tank_moves_too_far" not in ds.ALL_CATEGORIES


def test_value_fault_categories_are_disjoint_from_stage_one():
    assert set(ds.VALUE_FAULT_CATEGORIES).isdisjoint(set(ds.ALL_CATEGORIES))
    assert len(ds.VALUE_FAULT_CATEGORIES) == 5


def test_value_transitions_deterministic_across_many_seeds():
    for seed in range(50):
        episode_id = f"value-check-{seed:04d}"
        a = ds.generate_value_episode(episode_id, seed=seed)
        b = ds.generate_value_episode(episode_id, seed=seed)
        for ra, rb in zip(a, b):
            assert ra.state_before == rb.state_before
            assert ra.state_after == rb.state_after
            assert ra.notes == rb.notes


def test_new_faults_look_correct_at_component_label_level():
    # The whole point of stage 2: these faults are label-correct (component
    # attribution alone would pass them) but value-incorrect.
    ep = ds.generate_value_episode("value-label-check", seed=3)
    by_category = {r.category: r for r in ep}
    for category in ds.VALUE_FAULT_CATEGORIES:
        record = by_category[category]
        assert record.is_faulty
        assert record.fault_type == category
        assert set(record.observed_changed_components) == set(
            record.expected_changed_components
        ), (
            f"{category} should be label-correct; component metrics alone must not catch it"
        )


def test_wrong_alien_disappears_never_degenerates_to_no_visible_change():
    for seed in range(300):
        record = ds.build_value_transition(
            episode_id="degenerate-check",
            step_number=0,
            seed=seed,
            category="wrong_alien_disappears",
        )
        assert "alien" in record.observed_changed_components
