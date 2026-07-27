import pytest

from visual_transition_benchmark.compiler.contracts import (
    EvidenceContractError,
    VisualEvidenceRequirement,
)


def _make(**overrides):
    defaults = dict(
        domain_name="arcade",
        component_type="tank",
        property_name="position",
        evidence_kind="spatial_position",
        candidate_region_id="tank_band",
    )
    defaults.update(overrides)
    return VisualEvidenceRequirement(**defaults)


def test_requirement_id_is_deterministic():
    a = _make()
    b = _make()
    assert a.requirement_id == b.requirement_id


def test_requirement_id_changes_with_any_identity_field():
    base = _make()
    changed = _make(
        property_name="direction",
        evidence_kind="signed_delta",
        comparison="signed_delta",
    )
    assert base.requirement_id != changed.requirement_id


def test_rejects_unknown_evidence_kind():
    with pytest.raises(EvidenceContractError):
        _make(evidence_kind="not_a_real_kind")


def test_rejects_unknown_comparison():
    with pytest.raises(EvidenceContractError):
        _make(comparison="not_a_real_comparison")


def test_rejects_empty_identity_fields():
    with pytest.raises(EvidenceContractError):
        _make(component_type="")


def test_numeric_value_with_precision_requires_value_domain():
    with pytest.raises(EvidenceContractError):
        _make(
            evidence_kind="numeric_value", comparison="equal", required_precision=0.05
        )


def test_numeric_value_with_precision_and_value_domain_is_valid():
    req = _make(
        evidence_kind="numeric_value",
        comparison="equal",
        required_precision=0.05,
        expected_value_domain=(0.1, 0.6),
    )
    assert req.requirement_id


def test_visible_identity_requires_identity_marker_permission():
    with pytest.raises(EvidenceContractError):
        _make(evidence_kind="visible_identity", comparison="identity_equal")


def test_visible_identity_with_permission_is_valid():
    req = _make(
        evidence_kind="visible_identity",
        comparison="identity_equal",
        permits_identity_marker=True,
    )
    assert req.requirement_id
