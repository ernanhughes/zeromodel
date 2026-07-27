from visual_transition_benchmark.domains.protocol import AnalysisMetadata
from visual_transition_benchmark.domains.warehouse import contracts as wc
from visual_transition_benchmark.domains.warehouse import faults as wf


def _analyze(record):
    comp = wc.WarehouseComponentAnalyzer()
    val = wc.WarehouseValueAnalyzer()
    md = AnalysisMetadata(record.transition_id, record.step_number)
    return (
        comp.analyze(record.frame_before, record.frame_after, record.action, md),
        val.analyze(record.frame_before, record.frame_after, record.action, md),
    )


def test_door_bar_is_not_diluted_by_the_cells_empty_majority():
    # Regression test: the door glyph is only 2px wide inside a 6px cell, so
    # mode-based whole-cell classification (used for robot/crate/wall) would
    # read every door state as "empty". classify_door_state must use the
    # bar's own sub-region, not the whole cell.
    record = wf.build_transition(
        episode_id="e", step_number=0, seed=1, category="door_opens"
    )
    grids = wc.build_transition_evidence(record.frame_before, record.frame_after)
    assert wc.classify_door_state(grids, "before_mean") == "closed"
    assert wc.classify_door_state(grids, "after_mean") == "open"


def test_no_false_alarms_on_ordinary_categories_except_documented_boundary_cases():
    documented_false_alarms = {
        "robot_blocked_by_wall",
        "push_attempt_with_no_crate_is_noop",
    }
    for category in wf.ORDINARY_CATEGORIES:
        for seed in range(15):
            record = wf.build_transition(
                episode_id="e", step_number=0, seed=seed, category=category
            )
            component, value = _analyze(record)
            flagged = bool(
                component.missing_components
                or component.unexpected_components
                or value.value_flags
            )
            if category in documented_false_alarms:
                assert flagged
            else:
                assert not flagged, (
                    f"{category} seed={seed} unexpectedly flagged: {component}, {value}"
                )


def test_presence_faults_are_caught_at_component_level():
    for category in wf.PRESENCE_FAULT_CATEGORIES:
        record = wf.build_transition(
            episode_id="e", step_number=0, seed=2, category=category
        )
        component, _ = _analyze(record)
        assert component.missing_components or component.unexpected_components, category


def test_value_faults_are_caught_only_at_value_level():
    for category in wf.VALUE_FAULT_CATEGORIES:
        record = wf.build_transition(
            episode_id="e", step_number=0, seed=2, category=category
        )
        component, value = _analyze(record)
        assert (
            not component.missing_components and not component.unexpected_components
        ), f"{category} should be component-clean"
        assert value.value_flags, f"{category} should raise a value flag"


def test_relation_faults_are_caught_via_the_adjacency_relation():
    # crate_moves_without_robot_adjacency is built on WAIT, so it is also
    # legitimately presence-catchable (component-level flags a real anomaly
    # too); the push-based relation faults are component-clean by design.
    component_clean_categories = {
        "push_advances_robot_without_crate",
        "two_crates_move_during_single_push",
    }
    for category in wf.RELATION_FAULT_CATEGORIES:
        record = wf.build_transition(
            episode_id="e", step_number=0, seed=2, category=category
        )
        component, value = _analyze(record)
        if category in component_clean_categories:
            assert (
                not component.missing_components and not component.unexpected_components
            )
        if category == "push_advances_robot_without_crate":
            # documented, honest z-order blind spot -- see faults.py docstring
            assert value.value_flags == ()
        else:
            assert any(flag.startswith("relation:") for flag in value.value_flags), (
                category
            )


def test_identity_faults_remain_an_honest_blind_spot_for_value_flags():
    # No contract here can name the *correct* crate identity -- only ground
    # truth comparison (in cross_domain_metrics) can score this.
    for category in wf.IDENTITY_FAULT_CATEGORIES:
        record = wf.build_transition(
            episode_id="e", step_number=0, seed=2, category=category
        )
        _, value = _analyze(record)
        assert not any(
            f.startswith("relation:") for f in value.value_flags
        ) or category in ("expected_crate_remains_while_another_moves",)


def test_decode_matches_ground_truth_on_ordinary_transitions():
    for category in wf.ORDINARY_CATEGORIES:
        record = wf.build_transition(
            episode_id="e", step_number=0, seed=4, category=category
        )
        grids = wc.build_transition_evidence(record.frame_before, record.frame_after)
        decoded_robot_after = wc._decode_robot_cell(grids, "after_mean")
        assert list(decoded_robot_after) == record.state_after["robot"]
        assert wc.battery_level(grids, "after_mean") == record.state_after["battery"]
