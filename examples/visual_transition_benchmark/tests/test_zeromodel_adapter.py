import numpy as np

from visual_transition_benchmark import dataset as ds
from visual_transition_benchmark import zeromodel_adapter as zm


def _analyze(record):
    analyzer = zm.ArcadeBandZeroModelAnalyzer()
    metadata = zm.TransitionMetadata(transition_id=record.transition_id, step_number=record.step_number)
    return analyzer.analyze(record.frame_before, record.frame_after, record.action, metadata)


def test_output_coordinates_align_with_original_frame():
    record = ds.build_transition(episode_id="e", step_number=0, seed=1, category="tank_moves_left")
    analysis = _analyze(record)
    assert analysis.predicted_region_mask.shape == record.frame_before.shape
    # every predicted field's rectangle must lie inside the frame bounds
    by_id = {f.field_id: f for f in zm.FIELD_SCHEMA.fields}
    for field_id in analysis.predicted_fields:
        field = by_id[field_id]
        assert 0 <= field.y0 < field.y1 <= record.frame_before.shape[0]
        assert 0 <= field.x0 < field.x1 <= record.frame_before.shape[1]


def test_field_to_component_mapping_is_deterministic():
    mapping_a = dict(zm.FIELD_ID_TO_BAND)
    # rebuilding the module-level schema/bands must be pure/deterministic;
    # verify by recomputing the band assignment from scratch and comparing.
    recomputed = {}
    for field in zm.FIELD_SCHEMA.fields:
        recomputed[field.field_id] = zm._band_for_field(field.y0, field.x0)
    assert mapping_a == recomputed


def test_no_component_is_invented_outside_the_declared_mapping():
    record = ds.build_transition(
        episode_id="e", step_number=0, seed=1, category="fire_hits_advances_target"
    )
    analysis = _analyze(record)
    allowed = set(ds.COMPONENT_NAMES)
    assert set(analysis.predicted_components) <= allowed
    assert set(analysis.expected_components) <= allowed
    assert set(analysis.missing_components) <= allowed
    assert set(analysis.unexpected_components) <= allowed
    assert set(analysis.evidence_scores.keys()) == allowed


def test_same_transition_produces_identical_evidence():
    record = ds.build_transition(episode_id="e", step_number=0, seed=4, category="cooldown_clears")
    a = _analyze(record)
    b = _analyze(record)
    assert a.predicted_fields == b.predicted_fields
    assert a.predicted_components == b.predicted_components
    assert a.missing_components == b.missing_components
    assert a.unexpected_components == b.unexpected_components
    assert a.evidence_scores == b.evidence_scores
    assert np.array_equal(a.predicted_region_mask, b.predicted_region_mask)
    assert a.transition_evidence.transition_evidence_id == b.transition_evidence.transition_evidence_id


def test_adapter_never_receives_privileged_state():
    # The Protocol signature only accepts frame_before/frame_after/action/metadata;
    # TransitionMetadata carries no state fields at all.
    field_names = zm.TransitionMetadata.__dataclass_fields__.keys()
    assert set(field_names) == {"transition_id", "step_number"}
