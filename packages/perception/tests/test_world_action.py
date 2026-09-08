"""Unit tests for world-action coupling (future memory gates choice)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from zeromodel.perception import (
    build_dataset_manifest,
    build_grid_field_schema,
    encode_discrete_action,
    encode_source_array,
    fit_baseline_nearest_neighbor,
    predict_baseline_action,
)
from zeromodel.perception.dataset import RecordedInteractionDTO
from zeromodel.perception.expectations import PerceptionRegionAnnotationDTO
from zeromodel.perception.inference import BaselineInferenceConfigDTO
from zeromodel.perception.representation import (
    DiscreteActionSchemaDTO,
    SourceImageEncoderSpecDTO,
)
from zeromodel.perception.transition_conformance import TransitionExpectationDTO
from zeromodel.perception.transition_model import (
    fit_action_conditioned_transition_model,
)
from zeromodel.perception.transition_projection import project_expected_transition
from zeromodel.perception.memory_authority import MemoryAuthorityContextDTO
from zeromodel.perception.transition_verification import (
    FutureTransitionVerificationDTO,
)
from zeromodel.perception.world_action import (
    DeclarationScopeDTO,
    PerceptionWorldActionError,
    WorldActionPolicyDTO,
    check_expected_conformance,
    predict_action_with_future_memory,
)

_SPEC = SourceImageEncoderSpecDTO(color_space="L")
_WIDTH, _HEIGHT = 12, 8
_ACTION_SCHEMA = DiscreteActionSchemaDTO.from_labels(["left", "right"])


def _pattern(marker: int, action: str):
    before_array = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    before_array[marker % _HEIGHT, (marker * 5) % _WIDTH] = 200
    after_array = before_array.copy()
    if action == "left":
        after_array[:, 0:4] = 60
    else:
        after_array[:, 8:12] = 60
    return encode_source_array(before_array, _SPEC), encode_source_array(
        after_array, _SPEC
    )


def _build(count_per_action: int = 6, actions: tuple[str, ...] = ("left", "right")):
    interactions = []
    sources = {}
    step = 0
    marker = 0
    for action in actions:
        for _ in range(count_per_action):
            before, after = _pattern(marker, action)
            sources[before.source_vpm_id] = before
            sources[after.source_vpm_id] = after
            interactions.append(
                RecordedInteractionDTO.from_vpms(
                    sequence_id=f"seq-{action}",
                    step_index=step,
                    source=before,
                    target=encode_discrete_action(action, _ACTION_SCHEMA),
                    next_source=after,
                )
            )
            step += 1
            marker += 1
    manifest = build_dataset_manifest(
        interactions, source_encoder_spec_ids=[_SPEC.encoder_spec_id]
    )
    schema = build_grid_field_schema(
        next(iter(sources.values())), tile_width=4, tile_height=4, channel_mode="joint"
    )
    return manifest, sources, schema


def _fitted():
    manifest, sources, schema = _build()
    predictor = fit_baseline_nearest_neighbor(manifest, sources, training_split="all")
    model = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    query = next(
        sources[interaction.source_vpm_id]
        for interaction in manifest.interactions
        if interaction.action_label == "left"
    )
    return manifest, sources, schema, predictor, model, query


def _band_field_ids(schema, x0: int):
    return tuple(sorted(field.field_id for field in schema.fields if field.x0 == x0))


def test_baseline_result_is_preserved() -> None:
    _, _, schema, predictor, model, query = _fitted()
    out = predict_action_with_future_memory(predictor, model, query, schema)
    assert out.baseline == predict_baseline_action(predictor, query)
    assert out.baseline.selected_action == "left"
    assert [item.action_label for item in out.candidates[:2]] == [
        item.action_label for item in out.baseline.candidates[:2]
    ]
    assert [item.base_rank for item in out.candidates[:2]] == [0, 1]
    assert out.selected_action == "left"
    assert out.accepted


def test_contradiction_veto_falls_back_in_baseline_order() -> None:
    _, _, schema, predictor, model, query = _fitted()
    left_band = _band_field_ids(schema, 0)
    annotation = PerceptionRegionAnnotationDTO.create(
        schema, left_band, label="left-band"
    )
    stable = TransitionExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        annotation_ids=(annotation.annotation_id,),
        expected_change="stable",
        maximum_mean_absolute_change=0.0,
        maximum_changed_fraction=0.0,
    )
    out = predict_action_with_future_memory(
        predictor,
        model,
        query,
        schema,
        declarations=DeclarationScopeDTO.create({"left": (stable,)}, (annotation,)),
    )
    assert out.baseline.selected_action == "left"
    vetoed = next(item for item in out.candidates if item.action_label == "left")
    assert vetoed.status == "contradicted_by_transition_expectation"
    assert vetoed.memory_authority == "MAY_VETO"
    # Fallback follows baseline order among survivors, not confidence:
    # "right" was rank 1 and stays the only survivor.
    assert out.selected_action == "right"
    assert out.accepted
    survivor_ranks = [
        item.base_rank
        for item in out.candidates
        if item.status != "base_predictor_rejected"
    ]
    assert survivor_ranks == sorted(survivor_ranks)


def test_predictability_cannot_manufacture_utility() -> None:
    _, _, schema, predictor, model, query = _fitted()
    out = predict_action_with_future_memory(predictor, model, query, schema)
    ordered = [
        item for item in out.candidates if item.expected_transition_id is not None
    ]
    assert [item.base_rank for item in ordered] == sorted(
        item.base_rank for item in ordered
    )
    if out.accepted:
        first = next(
            item for item in out.candidates if item.action_label == out.selected_action
        )
        assert first.base_rank == min(
            item.base_rank
            for item in out.candidates
            if item.status
            not in {
                "contradicted_by_transition_expectation",
                "future_out_of_distribution",
                "base_predictor_rejected",
            }
            and item.expected_transition_id is not None
        )


def test_conformance_not_applicable_without_expectations() -> None:
    _, _, schema, _, model, query = _fitted()
    expected = project_expected_transition(model, query, "left", schema)
    conformance = check_expected_conformance(expected, schema, (), ())
    assert conformance.status == "not_applicable"


def test_ood_future_is_handled_explicitly() -> None:
    manifest, sources, schema = _build(count_per_action=6, actions=("left",))
    predictor = fit_baseline_nearest_neighbor(
        manifest,
        sources,
        config=BaselineInferenceConfigDTO(maximum_distance=1.0),
        training_split="all",
    )
    model = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    far = encode_source_array(np.full((_HEIGHT, _WIDTH), 255, dtype=np.uint8), _SPEC)
    out = predict_action_with_future_memory(predictor, model, far, schema)
    assert out.baseline.status == "accepted"
    ood = [item for item in out.candidates if item.expected_transition_id]
    assert ood and all(item.status == "future_out_of_distribution" for item in ood)
    assert not out.accepted
    assert out.selected_action is None


def test_baseline_rejection_abstains_without_execution() -> None:
    _, _, schema, predictor, model, _ = _fitted()
    far = encode_source_array(np.full((_HEIGHT, _WIDTH), 255, dtype=np.uint8), _SPEC)
    calls = {"count": 0}
    projector = project_expected_transition

    def _counting(model, source, action, field_schema, **kwargs):
        calls["count"] += 1
        return projector(model, source, action, field_schema, **kwargs)

    out = predict_action_with_future_memory(
        predictor, model, far, schema, projector=_counting
    )
    assert out.baseline.status != "accepted"
    assert not out.accepted
    assert out.selected_action is None
    assert out.baseline == predict_baseline_action(predictor, far)
    assert calls["count"] <= WorldActionPolicyDTO().candidate_count


def test_insufficient_evidence_policy_is_explicit() -> None:
    _, _, schema, predictor, model, query = _fitted()
    policy = WorldActionPolicyDTO(min_support=500, reject_on_insufficient=True)
    out = predict_action_with_future_memory(
        predictor, model, query, schema, policy=policy
    )
    assert all(
        item.status == "insufficient_future_evidence"
        for item in out.candidates
        if item.expected_transition_id is not None
    )
    assert not out.accepted


def test_baseline_override_ranking_is_honored() -> None:
    _, _, schema, predictor, model, query = _fitted()
    baseline = predict_baseline_action(predictor, query)
    assert len(baseline.candidates) == 2
    swapped = dataclasses.replace(
        baseline,
        candidates=tuple(reversed(baseline.candidates)),
        selected_action=baseline.candidates[1].action_label,
    )
    out = predict_action_with_future_memory(
        predictor, model, query, schema, baseline_override=swapped
    )
    assert out.baseline == swapped
    considered = [
        item for item in out.candidates if item.expected_transition_id is not None
    ]
    assert [item.base_rank for item in considered] == [0, 1]
    assert considered[0].action_label == swapped.candidates[0].action_label
    if out.accepted:
        assert out.selected_action == swapped.candidates[0].action_label


def test_baseline_override_validates_identity() -> None:
    _, _, schema, predictor, model, query = _fitted()
    baseline = predict_baseline_action(predictor, query)
    bad_model = dataclasses.replace(baseline, model_id="sha256:" + "f" * 64)
    with pytest.raises(PerceptionWorldActionError, match="different predictor model"):
        predict_action_with_future_memory(
            predictor, model, query, schema, baseline_override=bad_model
        )
    bad_source = dataclasses.replace(baseline, source_vpm_id="sha256:" + "e" * 64)
    with pytest.raises(PerceptionWorldActionError, match="different source"):
        predict_action_with_future_memory(
            predictor, model, query, schema, baseline_override=bad_source
        )


def _contradiction_fixture():
    manifest, _, schema, predictor, model, query = _fitted()
    left_band = _band_field_ids(schema, 0)
    annotation = PerceptionRegionAnnotationDTO.create(
        schema, left_band, label="left-band"
    )
    stable = TransitionExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        annotation_ids=(annotation.annotation_id,),
        expected_change="stable",
        maximum_mean_absolute_change=0.0,
        maximum_changed_fraction=0.0,
    )
    scope = DeclarationScopeDTO.create({"left": (stable,)}, (annotation,))
    return manifest, schema, predictor, model, query, scope


def _recorded_validity(model, manifest, schema, action, statuses):
    from zeromodel.perception.memory_authority import (
        record_verification_event as _record,
    )

    validity = None
    for index, status in enumerate(statuses):
        verification = FutureTransitionVerificationDTO(
            verification_id=f"verification-{index}",
            expected_transition_id="expected",
            observed_transition_evidence_id="observed",
            status=status,
            mean_absolute_error=0.1,
            direction_error_rate=0.2,
            changed_field_error_rate=0.3,
        )
        validity = _record(
            validity,
            verification,
            transition_model_id=model.model_id,
            action_label=action,
            field_schema_id=schema.field_schema_id,
            training_dataset_id=manifest.dataset_id,
        )
    return validity


def test_stale_authority_annotates_contradiction_without_veto() -> None:
    manifest, schema, predictor, model, query, scope = _contradiction_fixture()
    validity = _recorded_validity(
        model, manifest, schema, "left", ["future_projection_mismatch"] * 4
    )
    assert validity.staleness_score == 1.0
    out = predict_action_with_future_memory(
        predictor,
        model,
        query,
        schema,
        declarations=scope,
        authority=MemoryAuthorityContextDTO.create({"left": validity}),
    )
    left_candidate = next(
        item for item in out.candidates if item.action_label == "left"
    )
    assert left_candidate.memory_authority == "STALE"
    assert left_candidate.status != "contradicted_by_transition_expectation"
    assert any("without veto authority" in reason for reason in left_candidate.reasons)
    # Baseline top is kept: the stale memory may not command the decision.
    assert out.selected_action == "left"
    assert out.accepted


def test_fresh_authority_preserves_genuine_veto() -> None:
    manifest, schema, predictor, model, query, scope = _contradiction_fixture()
    validity = _recorded_validity(model, manifest, schema, "left", ["confirmed"] * 4)
    out = predict_action_with_future_memory(
        predictor,
        model,
        query,
        schema,
        declarations=scope,
        authority=MemoryAuthorityContextDTO.create({"left": validity}),
    )
    left_candidate = next(
        item for item in out.candidates if item.action_label == "left"
    )
    assert left_candidate.memory_authority == "MAY_VETO"
    assert left_candidate.status == "contradicted_by_transition_expectation"
    assert out.selected_action == "right"


def test_declaration_scope_sorts_and_misses_cleanly() -> None:
    _, _, schema, _, _, _ = _fitted()
    left_band = _band_field_ids(schema, 0)
    annotation = PerceptionRegionAnnotationDTO.create(
        schema, left_band, label="left-band"
    )
    stable = TransitionExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        annotation_ids=(annotation.annotation_id,),
        expected_change="stable",
        maximum_mean_absolute_change=0.0,
        maximum_changed_fraction=0.0,
    )
    scope = DeclarationScopeDTO.create({"right": (stable,), "left": (stable,)})
    assert [action for action, _ in scope.expectations_by_action] == [
        "left",
        "right",
    ]
    assert scope.scope_for("jump") == ()
    with pytest.raises(PerceptionWorldActionError):
        DeclarationScopeDTO(
            expectations_by_action=(("left", (stable,)),),
            annotations=(),
            relations=(),
            version="bogus",
        )
