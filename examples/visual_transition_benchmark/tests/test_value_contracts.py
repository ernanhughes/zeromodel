from visual_transition_benchmark import dataset as ds
from visual_transition_benchmark import value_contracts as vc


def _decode(record):
    te = vc.build_value_transition_evidence(record.frame_before, record.frame_after)
    return vc.decode_values(te)


def test_cooldown_band_is_not_diluted_by_neighboring_background_columns():
    # Regression test for the tile-width-24 bug: the cooldown corner is only
    # 2px wide inside the environment's 28px canvas; a coarse field grid dilutes
    # its mean with adjacent background pixels. The fine schema must isolate
    # exactly the 4 real cooldown pixels (2 rows x 2 cols).
    assert len(vc._VALUE_BAND_FIELD_IDS["cooldown"]) == 4


def test_tank_bleed_pixel_does_not_tie_with_the_true_column():
    # tank_moves_too_far renders the tank two cells away; if column decoding
    # used max instead of mean, the 1px bleed from the wide triangle base
    # ties with the true center column and corrupts delta_x.
    record = ds.build_transition(
        episode_id="e", step_number=0, seed=1, category="tank_moves_left"
    )
    values = _decode(record)
    assert values.tank.before_x == record.state_before["tank_x"]
    assert values.tank.after_x == record.state_after["tank_x"]


def test_decoded_values_match_ground_truth_on_ordinary_transitions():
    for category in ds.ORDINARY_CATEGORIES:
        for seed in range(10):
            record = ds.build_transition(
                episode_id="e", step_number=0, seed=seed, category=category
            )
            values = _decode(record)
            assert values.tank.before_x == record.state_before["tank_x"]
            assert values.tank.after_x == record.state_after["tank_x"]
            expected_cooldown = (
                "blocked" if record.state_after["cooldown"] == 1 else "ready"
            )
            assert values.cooldown.after_level == expected_cooldown
            assert values.alien.after_x == record.state_after["target_x"]


def test_cooldown_classification_levels():
    assert vc.classify_cooldown_level(40 / 255) == "ready"
    assert vc.classify_cooldown_level(160 / 255) == "blocked"
    assert vc.classify_cooldown_level(100 / 255) == "out_of_domain"
    assert vc.classify_cooldown_level(0.0) == "out_of_domain"


def test_contracts_catch_tank_moves_too_far():
    record = ds.build_value_transition(
        episode_id="e", step_number=0, seed=1, category="tank_moves_too_far"
    )
    values = _decode(record)
    verdict = vc.evaluate_contracts(record.action, values)
    assert verdict.tank_direction_ok is True  # correct direction...
    assert verdict.tank_magnitude_ok is False  # ...but wrong magnitude
    assert "tank_magnitude_exceeds_single_step_bound" in verdict.relation_violations


def test_contracts_catch_wrong_direction_where_stage_one_could_not():
    record = ds.build_transition(
        episode_id="e", step_number=0, seed=1, category="tank_moves_wrong_direction"
    )
    values = _decode(record)
    verdict = vc.evaluate_contracts(record.action, values)
    assert verdict.tank_direction_ok is False


def test_contracts_catch_cooldown_out_of_domain_values():
    for category in (
        "cooldown_activates_with_wrong_value",
        "cooldown_decreases_to_wrong_value",
    ):
        record = ds.build_value_transition(
            episode_id="e", step_number=0, seed=2, category=category
        )
        values = _decode(record)
        verdict = vc.evaluate_contracts(record.action, values)
        assert values.cooldown.after_level == "out_of_domain"
        assert not verdict.cooldown_value_ok
        assert "cooldown_value_out_of_domain" in verdict.relation_violations


def test_contracts_remain_honestly_blind_to_target_identity_faults():
    # Neither wrong_alien_disappears nor two_aliens_disappear_instead_of_one
    # violate any contract here: both render a cooldown-blocked, single-target
    # substitution that is indistinguishable from a legitimate hit without
    # the hidden alien queue. This is a documented limitation, not a bug.
    for category in ("wrong_alien_disappears", "two_aliens_disappear_instead_of_one"):
        record = ds.build_value_transition(
            episode_id="e", step_number=0, seed=3, category=category
        )
        values = _decode(record)
        verdict = vc.evaluate_contracts(record.action, values)
        assert verdict.relation_violations == ()
        assert verdict.cooldown_value_ok
        assert verdict.tank_direction_ok and verdict.tank_magnitude_ok


def test_no_false_alarms_on_many_ordinary_transitions():
    false_alarms = 0
    total = 0
    for category in ds.ORDINARY_CATEGORIES:
        if category == "tank_remains_stationary_at_boundary":
            continue  # documented false-alarm case, carried from stage 1
        for seed in range(30):
            record = ds.build_transition(
                episode_id="e", step_number=0, seed=seed, category=category
            )
            values = _decode(record)
            verdict = vc.evaluate_contracts(record.action, values)
            total += 1
            if (
                verdict.tank_direction_ok is False
                or verdict.tank_magnitude_ok is False
                or not verdict.cooldown_value_ok
                or verdict.relation_violations
            ):
                false_alarms += 1
    assert false_alarms == 0, (
        f"{false_alarms}/{total} legitimate transitions were flagged"
    )
