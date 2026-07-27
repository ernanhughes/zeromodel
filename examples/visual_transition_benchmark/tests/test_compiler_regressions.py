"""Regression tripwire for the non-negotiable constraint that stage 4 (the
evidence contract compiler, ``compiler/`` + ``compiler_adapters/``) never
touches stage 1/2/3 behavior. ``git status`` at every point in this branch's
history confirms every stage 1/2/3 file is untouched (only new files were
added under ``compiler/``, ``compiler_adapters/``, and this test); this file
additionally pins exact frame hashes for a representative, deterministic
transition per category across all three stages, so any *future* accidental
edit to ``dataset.py`` / ``value_contracts.py`` / the warehouse domain fails
loudly here rather than silently.
"""

import hashlib

from visual_transition_benchmark import dataset as ds
from visual_transition_benchmark.domains.warehouse import faults as wf


def _hash(arr) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


STAGE1_EXPECTED = {
    "tank_moves_left": ("616f014bff2133c2", "b5393c572f5e88ae"),
    "tank_moves_right": ("e365314884bd0aed", "438cda414b57196a"),
    "tank_remains_stationary_at_boundary": ("46f9202e9b5ad858", "46f9202e9b5ad858"),
    "fire_hits_advances_target": ("49e46341a170608e", "eac18988b85d2579"),
    "fire_hits_clears_wave": ("364f4fde9a8dbdc2", "aaf14d30e789b02d"),
    "fire_miss_activates_cooldown": ("c03a4540943eff22", "213a6cfc7668967a"),
    "cooldown_clears": ("9bc1324201f64b13", "a7c3c084985eca2a"),
    "no_op": ("6e697e389d66f313", "6e697e389d66f313"),
    "fire_no_projectile": ("49e46341a170608e", "fbd8cdf59131334b"),
    "fire_no_cooldown": ("810e809223430036", "810e809223430036"),
    "alien_disappears_without_hit": ("7837ac41e12f1c3d", "513452d1553bba3b"),
    "tank_moves_wrong_action": ("84107e0461df98f4", "810e809223430036"),
    "tank_moves_wrong_direction": ("6b8f95d69895feca", "89ad54864c6aa1c1"),
    "cooldown_changes_incorrectly": ("81ebae285119f8c3", "2e7661a69aed18c4"),
    "unrelated_alien_change": ("810e809223430036", "8199fe7e9c692521"),
    "background_changes_unexpectedly": ("80e22e7d9c3e1b26", "8c74397deb501d44"),
    "expected_target_remains_unchanged": ("364f4fde9a8dbdc2", "364f4fde9a8dbdc2"),
    "extra_component_changes_alongside": ("e3e706fc1b35338d", "8503a64f921e245b"),
}

STAGE2_EXPECTED = {
    "tank_moves_too_far": ("1088e4fc746a2694", "7c6c9928191bd631"),
    "cooldown_activates_with_wrong_value": ("810e809223430036", "16a1e2ef7f9f7c89"),
    "cooldown_decreases_to_wrong_value": ("63f688e025cf7d01", "9404a08948b2f84d"),
    "wrong_alien_disappears": ("57051947a0af0940", "d0ab0e53035ace8d"),
    "two_aliens_disappear_instead_of_one": ("49e46341a170608e", "6c00ce54c4f6c89c"),
}

STAGE3_WAREHOUSE_EXPECTED = {
    "robot_moves_left": ("844d9563c6500cb6", "a4ac6f7d6dd5fccd"),
    "robot_moves_right": ("4dcf9795c0124a5e", "cadf1810dd03e8cd"),
    "robot_blocked_by_wall": ("195bce76bf29b25b", "195bce76bf29b25b"),
    "push_crate_moves_target": ("cd0d2175cc9369b2", "b120e4b82cbac298"),
    "push_crate_reaches_goal": ("62d175625792b7e9", "2d89829ab1db3a1d"),
    "push_attempt_with_no_crate_is_noop": ("145bc410c4d25b89", "145bc410c4d25b89"),
    "door_opens": ("4dcf9795c0124a5e", "13570530809a54c1"),
    "wait_no_op": ("4dcf9795c0124a5e", "4dcf9795c0124a5e"),
    "robot_moves_during_wait": ("4dcf9795c0124a5e", "aaa28680d5b629f1"),
    "wall_changes_unexpectedly": ("4dcf9795c0124a5e", "e76527b79c146d24"),
    "door_changes_during_move": ("ecdf0888ad721432", "0ecbacd7b526c327"),
    "unrelated_crate_moves_during_robot_move": ("b5e9676229f768b8", "880b0af6fab206e6"),
    "push_fails_silently": ("4720f0926fe94159", "4720f0926fe94159"),
    "robot_moves_wrong_direction": ("561c2bafc9aa2205", "c6d80323a450dd1a"),
    "robot_moves_too_far": ("ecdf0888ad721432", "f12473d858b54e38"),
    "battery_decreases_by_wrong_amount": ("4dcf9795c0124a5e", "eb5669916b2096e7"),
    "door_changes_to_wrong_visual_state": ("4dcf9795c0124a5e", "cb0e50d790e402cf"),
    "push_advances_robot_without_crate": ("cd0d2175cc9369b2", "cadf1810dd03e8cd"),
    "crate_moves_without_robot_adjacency": ("b5e9676229f768b8", "24bae98d73769748"),
    "two_crates_move_during_single_push": ("4a2116dfcdda1d50", "7a97bc9f68817cce"),
    "wrong_crate_moves": ("4a2116dfcdda1d50", "39f85f1dc1074bf4"),
    "two_crates_swap_identities": ("4a2116dfcdda1d50", "fbe490428aaf0176"),
    "expected_crate_remains_while_another_moves": (
        "4a2116dfcdda1d50",
        "a788e8670d75a060",
    ),
}


def test_stage1_arcade_frames_are_bit_for_bit_unchanged():
    for category, expected in STAGE1_EXPECTED.items():
        rec = ds.build_transition(
            episode_id="fp", step_number=0, seed=7, category=category
        )
        actual = (_hash(rec.frame_before), _hash(rec.frame_after))
        assert actual == expected, (
            f"stage1 category {category!r} frame hash changed: {actual} != {expected}"
        )


def test_stage2_value_fault_frames_are_bit_for_bit_unchanged():
    for category, expected in STAGE2_EXPECTED.items():
        rec = ds.build_value_transition(
            episode_id="fp2", step_number=0, seed=7, category=category
        )
        actual = (_hash(rec.frame_before), _hash(rec.frame_after))
        assert actual == expected, (
            f"stage2 category {category!r} frame hash changed: {actual} != {expected}"
        )


def test_stage3_warehouse_frames_are_bit_for_bit_unchanged():
    for category, expected in STAGE3_WAREHOUSE_EXPECTED.items():
        rec = wf.build_transition(
            episode_id="fp3", step_number=0, seed=7, category=category
        )
        actual = (_hash(rec.frame_before), _hash(rec.frame_after))
        assert actual == expected, (
            f"stage3 warehouse category {category!r} frame hash changed: {actual} != {expected}"
        )


def test_all_categories_are_covered():
    assert set(STAGE1_EXPECTED) == set(ds.ALL_CATEGORIES)
    assert set(STAGE2_EXPECTED) == set(ds.VALUE_FAULT_CATEGORIES)
    assert set(STAGE3_WAREHOUSE_EXPECTED) == set(wf.ALL_CATEGORIES)
