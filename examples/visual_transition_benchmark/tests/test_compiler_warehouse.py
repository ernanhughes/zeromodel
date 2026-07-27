"""End-to-end compiler runs over the warehouse domain adapter's declared
cases. Small per-category sample counts (fast enough for CI); the frozen,
full-scale evaluation lives in ``compiler_run.py`` /
``artifacts/evidence_contract_compiler/``.
"""

from visual_transition_benchmark.compiler.candidates import generate_candidates
from visual_transition_benchmark.compiler.compile import compile_requirement
from visual_transition_benchmark.compiler_adapters import warehouse as warehouse_adapter


def _compile(case, count=3, seed=0):
    samples = case.build_samples(count, seed)
    candidates = generate_candidates(case.requirement, case.region)
    return compile_requirement(
        case.requirement,
        case.region,
        candidates,
        samples,
        canonical_levels=case.canonical_levels,
        sub_patch_offsets=getattr(case, "sub_patch_offsets", ()),
        min_decoding_accuracy=case.min_decoding_accuracy,
    )


def _case(name):
    return next(c for c in warehouse_adapter.build_cases() if c.name == name)


def test_six_cases_are_declared_and_push_relation_is_deliberately_excluded():
    names = {c.name for c in warehouse_adapter.build_cases()}
    assert names == {
        "robot_position",
        "robot_direction",
        "robot_movement_magnitude",
        "battery_value",
        "door_state",
        "crate_identity",
    }


def test_robot_position_compiles_with_mean_not_max():
    compiled = _compile(_case("robot_position"))
    assert compiled.status == "compiled"
    assert compiled.selected_candidate.aggregation in ("mean", "centroid")


def test_robot_direction_compiles_and_rejects_max_aggregation_candidate():
    compiled = _compile(_case("robot_direction"), count=10)
    assert compiled.status == "compiled"
    max_candidates = [
        e
        for e in compiled.all_evaluations
        if not e.passed and "tied" in " ".join(e.rejection_reasons)
    ]
    assert max_candidates, (
        "expected at least one max-aggregation candidate to be rejected for ambiguity"
    )


def test_robot_movement_magnitude_compiles():
    compiled = _compile(_case("robot_movement_magnitude"))
    assert compiled.status == "compiled"


def test_battery_value_compiles():
    compiled = _compile(_case("battery_value"))
    assert compiled.status == "compiled"


def test_door_state_compiles_via_auto_narrowing():
    # Direct analogue of arcade's cooldown-dilution rediscovery: the door bar
    # is a thin glyph inside a mostly-background cell.
    compiled = _compile(_case("door_state"), count=10)
    assert compiled.status == "compiled"
    assert compiled.selected_candidate.decoder_kind == "dominant_field_value"


def test_crate_identity_compiles_via_marker_pattern_not_mean():
    compiled = _compile(_case("crate_identity"), count=2)
    assert compiled.status == "compiled"
    assert compiled.selected_candidate.decoder_kind == "local_marker_pattern"
