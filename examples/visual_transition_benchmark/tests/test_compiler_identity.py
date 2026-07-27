"""Dedicated test for the identity-evidence asymmetry between domains: the
warehouse's crate dots are a genuine visible identity marker (recoverable),
while the arcade's alien sprite carries no such marker at all (not
recoverable). This is a documented, load-bearing finding (see
``MANUAL_REPRESENTATION_INVENTORY.md`` items 7 and 12) -- these two cases
must not resolve to the same compiler outcome.
"""

from visual_transition_benchmark.compiler.candidates import generate_candidates
from visual_transition_benchmark.compiler.compile import compile_requirement
from visual_transition_benchmark.compiler_adapters import arcade as arcade_adapter
from visual_transition_benchmark.compiler_adapters import warehouse as warehouse_adapter


def _compile(adapter, name, count, seed=0):
    case = next(c for c in adapter.build_cases() if c.name == name)
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


def test_arcade_alien_identity_is_not_recoverable():
    # Tightened per repair review: the alien sprite carries no marker at all,
    # so this must land specifically on insufficient_observability (evidence
    # absent from the frame), not merely "not compiled" -- a regression that
    # instead produced insufficient_representation (evidence exists but no
    # candidate captured it) would be a different, incorrect finding and
    # this assertion previously would not have caught it.
    compiled = _compile(arcade_adapter, "alien_target_identity", count=10)
    assert compiled.status == "insufficient_observability"


def test_warehouse_crate_identity_is_recoverable():
    compiled = _compile(warehouse_adapter, "crate_identity", count=2)
    assert compiled.status == "compiled"


def test_identity_evidence_kind_requires_permits_identity_marker_in_both_domains():
    for adapter, name in ((arcade_adapter, "alien_target_identity"), (warehouse_adapter, "crate_identity")):
        case = next(c for c in adapter.build_cases() if c.name == name)
        assert case.requirement.evidence_kind == "visible_identity"
        assert case.requirement.permits_identity_marker is True


def test_the_two_identity_outcomes_are_not_the_same_status():
    arcade_result = _compile(arcade_adapter, "alien_target_identity", count=10)
    warehouse_result = _compile(warehouse_adapter, "crate_identity", count=2)
    assert arcade_result.status != warehouse_result.status
