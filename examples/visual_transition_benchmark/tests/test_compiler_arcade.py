"""End-to-end compiler runs over the arcade domain adapter's declared cases.
Small per-category sample counts (fast enough for CI); the frozen, full-scale
evaluation lives in ``compiler_run.py`` / ``artifacts/evidence_contract_compiler/``.
"""

from visual_transition_benchmark.compiler.candidates import generate_candidates
from visual_transition_benchmark.compiler.compile import compile_requirement
from visual_transition_benchmark.compiler_adapters import arcade as arcade_adapter


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
    return next(c for c in arcade_adapter.build_cases() if c.name == name)


def test_all_six_required_cases_are_declared():
    names = {c.name for c in arcade_adapter.build_cases()}
    assert names == {
        "tank_presence",
        "tank_position",
        "tank_direction",
        "tank_movement_magnitude",
        "cooldown_value",
        "alien_target_identity",
    }


def test_tank_presence_compiles():
    compiled = _compile(_case("tank_presence"))
    assert compiled.status == "compiled"


def test_tank_position_compiles_with_mean_not_max():
    compiled = _compile(_case("tank_position"))
    assert compiled.status == "compiled"
    assert compiled.selected_candidate.aggregation in ("mean", "centroid")


def test_tank_direction_compiles():
    compiled = _compile(_case("tank_direction"))
    assert compiled.status == "compiled"


def test_tank_movement_magnitude_compiles():
    compiled = _compile(_case("tank_movement_magnitude"))
    assert compiled.status == "compiled"


def test_cooldown_value_compiles_via_auto_narrowing():
    # The rediscovery case: naive whole-region decoders must fail regardless
    # of resolution; only the auto-narrowing decoder should pass.
    compiled = _compile(_case("cooldown_value"), count=10)
    assert compiled.status == "compiled"
    assert compiled.selected_candidate.decoder_kind == "dominant_field_value"
    naive_passed = [
        e
        for e in compiled.all_evaluations
        if e.passed and e.candidate_id != compiled.selected_candidate.candidate_id
    ]
    assert naive_passed == []


def test_alien_target_identity_reports_insufficient_observability():
    # The alien sprite carries no identity marker at all -- this is the
    # documented, honest negative result, not a bug to be patched around.
    # Asserted as the exact status (not just "!= compiled") because the
    # compiled/insufficient_representation/insufficient_observability
    # trichotomy is itself one of this experiment's central findings.
    compiled = _compile(_case("alien_target_identity"), count=10)
    assert compiled.status == "insufficient_observability"
