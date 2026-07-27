from visual_transition_benchmark.compiler.candidates import RegionGeometry, generate_candidates
from visual_transition_benchmark.compiler.contracts import VisualEvidenceRequirement

REGION = RegionGeometry(
    region_id="r", canvas_shape=(16, 28), y0=7, y1=9, x0=24, x1=28, cell_height=1, cell_width=4
)


def _req(evidence_kind, comparison, **overrides):
    defaults = dict(
        domain_name="arcade",
        component_type="cooldown",
        property_name="value",
        evidence_kind=evidence_kind,
        candidate_region_id="r",
        comparison=comparison,
    )
    defaults.update(overrides)
    return VisualEvidenceRequirement(**defaults)


def test_generation_is_deterministic():
    req = _req("numeric_value", "equal", expected_value_domain=(0.1, 0.6))
    a = generate_candidates(req, REGION)
    b = generate_candidates(req, REGION)
    assert [c.candidate_id for c in a] == [c.candidate_id for c in b]


def test_candidate_ids_are_unique_within_a_generation():
    req = _req("numeric_value", "equal", expected_value_domain=(0.1, 0.6))
    candidates = generate_candidates(req, REGION)
    assert len(candidates) == len({c.candidate_id for c in candidates})


def test_numeric_value_generates_bounded_small_set():
    req = _req("numeric_value", "equal", expected_value_domain=(0.1, 0.6))
    candidates = generate_candidates(req, REGION)
    # 2 resolutions x 3 decoders -- bounded, not a combinatorial search.
    assert len(candidates) == 6


def test_presence_generates_two_candidates():
    req = _req("presence", "changed")
    candidates = generate_candidates(req, REGION)
    assert len(candidates) == 2
    assert all(c.decoder_kind == "presence_threshold" for c in candidates)


def test_spatial_position_varies_aggregation_not_decoder():
    req = _req("spatial_position", "equal")
    candidates = generate_candidates(req, REGION)
    assert len(candidates) == 6
    assert all(c.decoder_kind == "argmax_field" for c in candidates)
    assert {c.aggregation for c in candidates} == {"mean", "max", "centroid"}


def test_relation_generates_two_threshold_variants():
    req = _req("relation", "relation_holds")
    candidates = generate_candidates(req, REGION)
    assert len(candidates) == 2
    assert all(c.decoder_kind == "relation_over_decoded" for c in candidates)


def test_unknown_evidence_kind_has_no_generator():
    # contracts.py itself rejects unknown evidence_kind at construction time;
    # this only guards generate_candidates' own dispatch never silently
    # falling through for a kind it doesn't recognize.
    from visual_transition_benchmark.compiler.candidates import _GENERATORS

    from visual_transition_benchmark.compiler.contracts import _EVIDENCE_KINDS

    assert set(_GENERATORS) == set(_EVIDENCE_KINDS)


def test_complexity_cost_increases_with_finer_resolution():
    req = _req("numeric_value", "equal", expected_value_domain=(0.1, 0.6))
    candidates = generate_candidates(req, REGION)
    coarse = [c for c in candidates if c.field_height == 1 and c.field_width == 4 and c.decoder_kind == "nearest_permitted_value"][0]
    fine = [c for c in candidates if c.field_height == 1 and c.field_width == 1 and c.decoder_kind == "nearest_permitted_value"][0]
    assert fine.complexity_cost > coarse.complexity_cost
