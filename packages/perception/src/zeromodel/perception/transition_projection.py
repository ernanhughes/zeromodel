"""Frozen-state future projection (no fitting).

``project_expected_transition`` operates only on already-frozen transition
memory: it never fits, updates, or caches anything. The same
source/action/model/parameters always yield the identical
ExpectedTransitionVPM, including its content-derived identity.

Distance formulas mirror the P3 whole-image and P4C field-weighted metrics
(same semantics strings); only the aggregation target differs (remembered
change rather than action votes).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping, Union

import numpy as np

from .expected_transition import (
    EXPECTED_TRANSITION_RENDER_SEMANTICS,
    EXPECTED_TRANSITION_VPM_VERSION,
    ExpectedTransitionFieldDTO,
    ExpectedTransitionVPMDTO,
    render_expected_transition_png,
)
from .fields import VPMFieldSchemaDTO, validate_source_for_schema
from .representation import SourceVPMDTO
from .transition_model import (
    CompiledTransitionModelDTO,
    EmpiricalTransitionModelDTO,
    PerceptionTransitionModelError,
    TransitionTrainingExampleDTO,
)

TRANSITION_PROJECTION_DISTANCE_SEMANTICS: Final = (
    "normalized_mean_absolute_pixel_distance"
)
TRANSITION_PROJECTION_WEIGHTED_DISTANCE_SEMANTICS: Final = (
    "field_weighted_normalized_mean_absolute_pixel_distance"
)

TransitionModelDTO = Union[EmpiricalTransitionModelDTO, CompiledTransitionModelDTO]


class PerceptionTransitionProjectionError(PerceptionTransitionModelError):
    """Raised when a frozen-state future cannot be projected."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return f"sha256:{hasher.hexdigest()}"


def _source_pixels(source: SourceVPMDTO) -> bytes:
    array = source.to_array()
    return np.ascontiguousarray(array, dtype=np.uint8).reshape(-1).tobytes()


def _pixel_array(pixels: bytes, width: int, height: int, channels: int) -> np.ndarray:
    array = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, channels)
    return array.astype(np.float64) / 255.0


def _field_means(
    pixels: bytes,
    schema: VPMFieldSchemaDTO,
    width: int,
    height: int,
    channels: int,
) -> dict[str, float]:
    array = _pixel_array(pixels, width, height, channels)
    means: dict[str, float] = {}
    for field in schema.fields:
        region = array[
            field.y0 : field.y1,
            field.x0 : field.x1,
            field.channel_start : field.channel_end,
        ]
        means[field.field_id] = float(np.mean(region))
    return means


def _pixel_distance(left: bytes, right: bytes) -> float:
    left_values = np.frombuffer(left, dtype=np.uint8).astype(np.int16)
    right_values = np.frombuffer(right, dtype=np.uint8).astype(np.int16)
    return float(np.mean(np.abs(left_values - right_values)) / 255.0)


def _field_weighted_pixel_distance(
    left: bytes,
    right: bytes,
    schema: VPMFieldSchemaDTO,
    width: int,
    height: int,
    channels: int,
    weights: Mapping[str, float],
) -> float:
    left_array = _pixel_array(left, width, height, channels)
    right_array = _pixel_array(right, width, height, channels)
    total_weight = 0.0
    total = 0.0
    for field in schema.fields:
        weight = weights[field.field_id]
        if weight <= 0.0:
            continue
        left_region = left_array[
            field.y0 : field.y1,
            field.x0 : field.x1,
            field.channel_start : field.channel_end,
        ]
        right_region = right_array[
            field.y0 : field.y1,
            field.x0 : field.x1,
            field.channel_start : field.channel_end,
        ]
        total += weight * float(np.mean(np.abs(left_region - right_region)))
        total_weight += weight
    if total_weight <= 0.0:
        raise PerceptionTransitionProjectionError("no positive field weight")
    return total / total_weight


def _check_field_weights(
    field_weights: Mapping[str, float] | None,
    schema: VPMFieldSchemaDTO,
) -> dict[str, float] | None:
    if field_weights is None:
        return None
    expected = {field.field_id for field in schema.fields}
    if set(field_weights) != expected:
        raise PerceptionTransitionProjectionError(
            "field weights must cover exactly the schema fields"
        )
    weights = {key: float(value) for key, value in field_weights.items()}
    if not all(np.isfinite(value) and value >= 0.0 for value in weights.values()):
        raise PerceptionTransitionProjectionError(
            "field weights must be finite and non-negative"
        )
    if sum(weights.values()) <= 0.0:
        raise PerceptionTransitionProjectionError("field weights require positive mass")
    return weights


@dataclass(frozen=True)
class _ProjectedMoments:
    signed: dict[str, float]
    absolute: dict[str, float]
    fractions: dict[str, float]
    signed_dispersion: dict[str, float]
    absolute_dispersion: dict[str, float]


@dataclass(frozen=True)
class _ProjectionContext:
    source: SourceVPMDTO
    action_label: str
    field_schema: VPMFieldSchemaDTO
    model_id: str
    training_dataset_id: str
    source_encoder_spec_id: str
    before_means: dict[str, float]


def _assemble_expected_transition(
    context: _ProjectionContext,
    moments: _ProjectedMoments,
    support_count: int,
    confidence: float,
    status: str,
) -> ExpectedTransitionVPMDTO:
    schema = context.field_schema
    ordered = tuple(field.field_id for field in schema.fields)
    clipped_confidence = min(1.0, max(0.0, confidence))
    fields = tuple(
        ExpectedTransitionFieldDTO(
            field_id=field_id,
            expected_after_mean=min(
                1.0,
                max(0.0, context.before_means[field_id] + moments.signed[field_id]),
            ),
            expected_mean_signed_change=moments.signed[field_id],
            expected_mean_absolute_change=moments.absolute[field_id],
            expected_changed_fraction=min(1.0, max(0.0, moments.fractions[field_id])),
            signed_change_dispersion=moments.signed_dispersion[field_id],
            absolute_change_dispersion=moments.absolute_dispersion[field_id],
            support_count=support_count,
        )
        for field_id in ordered
    )
    png_bytes, png_digest = render_expected_transition_png(
        fields, schema, schema.width, schema.height
    )
    canonical: Mapping[str, object] = {
        "action_label": context.action_label,
        "confidence": clipped_confidence,
        "fields": [item.canonical_payload() for item in fields],
        "field_schema_id": schema.field_schema_id,
        "model_id": context.model_id,
        "png_digest": png_digest,
        "render_semantics": EXPECTED_TRANSITION_RENDER_SEMANTICS,
        "source_encoder_spec_id": context.source_encoder_spec_id,
        "source_vpm_id": context.source.source_vpm_id,
        "status": status,
        "support_count": support_count,
        "training_dataset_id": context.training_dataset_id,
        "version": EXPECTED_TRANSITION_VPM_VERSION,
    }
    return ExpectedTransitionVPMDTO(
        expected_transition_id=_digest(_canonical_json(canonical)),
        source_vpm_id=context.source.source_vpm_id,
        action_label=context.action_label,
        field_schema_id=schema.field_schema_id,
        source_encoder_spec_id=context.source_encoder_spec_id,
        fields=fields,
        model_id=context.model_id,
        training_dataset_id=context.training_dataset_id,
        support_count=support_count,
        confidence=clipped_confidence,
        status=status,
        png_digest=png_digest,
        png_bytes=png_bytes,
    )


def _empty_moments(field_ids: tuple[str, ...]) -> _ProjectedMoments:
    zeros = {field_id: 0.0 for field_id in field_ids}
    return _ProjectedMoments(
        signed=dict(zeros),
        absolute=dict(zeros),
        fractions=dict(zeros),
        signed_dispersion=dict(zeros),
        absolute_dispersion=dict(zeros),
    )


def _resolve_status(
    *,
    support_count: int,
    min_support: int,
    spread: float,
    nearest: float,
    ood_spread_factor: float,
    ambiguity_value: float,
    ambiguity_threshold: float,
    confidence: float,
) -> tuple[str, float]:
    if support_count < min_support:
        return "insufficient_examples", min(confidence, 0.25)
    if spread > 0.0 and nearest > ood_spread_factor * spread:
        return "out_of_distribution", min(confidence, 0.5)
    if ambiguity_value > ambiguity_threshold:
        return "ambiguous_future", min(confidence, 0.5)
    return "supported", confidence


def _rank_neighbors(
    subset: list[TransitionTrainingExampleDTO],
    query_pixels: bytes,
    field_schema: VPMFieldSchemaDTO,
    width: int,
    height: int,
    channels: int,
    weights: dict[str, float] | None,
    neighbor_count: int,
) -> tuple[list[TransitionTrainingExampleDTO], np.ndarray]:
    if weights is None:
        distances = [
            (_pixel_distance(query_pixels, item.before_pixels), item.interaction_id)
            for item in subset
        ]
    else:
        distances = [
            (
                _field_weighted_pixel_distance(
                    query_pixels,
                    item.before_pixels,
                    field_schema,
                    width,
                    height,
                    channels,
                    weights,
                ),
                item.interaction_id,
            )
            for item in subset
        ]
    ranked = sorted(distances, key=lambda value: (value[0], value[1]))
    chosen = ranked[: max(1, min(neighbor_count, len(subset)))]
    by_id = {item.interaction_id: item for item in subset}
    ordered = [by_id[interaction_id] for _, interaction_id in chosen]
    return ordered, np.array([distance for distance, _ in chosen])


def _aggregate_neighbor_moments(
    ordered: list[TransitionTrainingExampleDTO],
    neighbor_weights: np.ndarray,
    field_ids: tuple[str, ...],
) -> _ProjectedMoments:
    total = float(neighbor_weights.sum()) + 1e-12

    def _mean(attribute: str) -> dict[str, float]:
        return {
            field_id: float(
                sum(
                    weight * getattr(item, attribute)[position]
                    for weight, item in zip(neighbor_weights, ordered)
                )
                / total
            )
            for position, field_id in enumerate(field_ids)
        }

    def _dispersion(attribute: str, center: dict[str, float]) -> dict[str, float]:
        dispersions = {}
        for position, field_id in enumerate(field_ids):
            moment = (
                sum(
                    weight
                    * (getattr(item, attribute)[position] - center[field_id]) ** 2
                    for weight, item in zip(neighbor_weights, ordered)
                )
                / total
            )
            dispersions[field_id] = float(np.sqrt(max(0.0, moment)))
        return dispersions

    signed = _mean("signed_deltas")
    absolute = _mean("absolute_deltas")
    fractions = _mean("changed_fractions")
    return _ProjectedMoments(
        signed=signed,
        absolute=absolute,
        fractions=fractions,
        signed_dispersion=_dispersion("signed_deltas", signed),
        absolute_dispersion=_dispersion("absolute_deltas", absolute),
    )


def _project_empirical(
    model: EmpiricalTransitionModelDTO,
    context: _ProjectionContext,
    weights: dict[str, float] | None,
    min_support: int,
) -> ExpectedTransitionVPMDTO:
    schema = context.field_schema
    field_ids = tuple(field.field_id for field in schema.fields)
    query_pixels = _source_pixels(context.source)
    if context.action_label not in model.action_labels:
        return _assemble_expected_transition(
            context, _empty_moments(field_ids), 0, 0.0, "unsupported_action"
        )
    index = model.action_labels.index(context.action_label)
    subset = [
        item for item in model.examples if item.action_label == context.action_label
    ]
    count = len(subset)
    ordered, selected_distances = _rank_neighbors(
        subset,
        query_pixels,
        schema,
        model.width,
        model.height,
        model.channels,
        weights,
        model.config.neighbor_count,
    )
    neighbor_weights = 1.0 / (1.0 + selected_distances * selected_distances)
    moments = _aggregate_neighbor_moments(ordered, neighbor_weights, field_ids)
    spread = float(model.action_spreads[index])
    typical = float(model.action_typical_nn_distances[index])
    # Local support scale: typical same-action neighbour spacing describes
    # possibly multimodal support where a global centroid spread does not.
    local_scale = typical if typical > 0.0 else spread
    nearest = float(selected_distances[0])
    support_factor = min(1.0, count / max(1, 2 * min_support))
    if local_scale > 0.0:
        proximity = 1.0 / (1.0 + nearest / local_scale)
    else:
        proximity = 1.0
    ambiguity = float(np.median(list(moments.signed_dispersion.values())))
    status, confidence = _resolve_status(
        support_count=count,
        min_support=min_support,
        spread=local_scale,
        nearest=nearest,
        ood_spread_factor=model.config.ood_spread_factor,
        ambiguity_value=ambiguity,
        ambiguity_threshold=model.config.ambiguity_threshold,
        confidence=support_factor * proximity,
    )
    return _assemble_expected_transition(context, moments, count, confidence, status)


def _ridge_moments(
    model: CompiledTransitionModelDTO,
    index: int,
    before_means: dict[str, float],
    field_ids: tuple[str, ...],
    change_epsilon: float,
) -> tuple[_ProjectedMoments, float]:
    rows = np.asarray(model.action_coefficients[index], dtype=np.float64)
    residual = np.asarray(model.action_residual_std[index], dtype=np.float64)
    query = np.array([before_means[field_id] for field_id in field_ids])
    # Ridge predictions are unbounded estimates; clamp to contract range.
    delta = np.clip(np.concatenate([query, np.ones(1)]) @ rows.T, -1.0, 1.0)
    signed = {
        field_id: float(delta[position]) for position, field_id in enumerate(field_ids)
    }
    absolute = {
        field_id: min(1.0, abs(float(delta[position])))
        for position, field_id in enumerate(field_ids)
    }
    fractions = {
        field_id: 1.0 if abs(float(delta[position])) > change_epsilon else 0.0
        for position, field_id in enumerate(field_ids)
    }
    signed_dispersion = {
        field_id: float(residual[position])
        for position, field_id in enumerate(field_ids)
    }
    absolute_dispersion = {
        field_id: float(residual[position]) * 0.5
        for position, field_id in enumerate(field_ids)
    }
    return (
        _ProjectedMoments(
            signed=signed,
            absolute=absolute,
            fractions=fractions,
            signed_dispersion=signed_dispersion,
            absolute_dispersion=absolute_dispersion,
        ),
        float(np.median(residual)),
    )


def _project_compiled(
    model: CompiledTransitionModelDTO,
    context: _ProjectionContext,
    weights: dict[str, float] | None,
    min_support: int,
) -> ExpectedTransitionVPMDTO:
    schema = context.field_schema
    field_ids = tuple(field.field_id for field in schema.fields)
    query_pixels = _source_pixels(context.source)
    if context.action_label not in model.action_labels:
        return _assemble_expected_transition(
            context, _empty_moments(field_ids), 0, 0.0, "unsupported_action"
        )
    index = model.action_labels.index(context.action_label)
    count = int(model.action_counts[index])
    moments, residual_median = _ridge_moments(
        model, index, context.before_means, field_ids, model.config.change_epsilon
    )
    # Local support: distance to the nearest same-action training example,
    # not to the global centroid, so multimodal support is described by the
    # support itself rather than its middle.
    subset = [
        item for item in model.examples if item.action_label == context.action_label
    ]
    if weights is None:
        distances = [
            _pixel_distance(query_pixels, item.before_pixels) for item in subset
        ]
    else:
        distances = [
            _field_weighted_pixel_distance(
                query_pixels,
                item.before_pixels,
                schema,
                model.width,
                model.height,
                model.channels,
                weights,
            )
            for item in subset
        ]
    nearest = min(distances)
    spread = float(model.action_spreads[index])
    typical = float(model.action_typical_nn_distances[index])
    local_scale = typical if typical > 0.0 else spread
    support_factor = min(1.0, count / max(1, 2 * min_support))
    if local_scale > 0.0:
        proximity = 1.0 / (1.0 + nearest / local_scale)
    else:
        proximity = 1.0
    fit_quality = 1.0 / (1.0 + residual_median)
    status, confidence = _resolve_status(
        support_count=count,
        min_support=min_support,
        spread=local_scale,
        nearest=nearest,
        ood_spread_factor=model.config.ood_spread_factor,
        ambiguity_value=residual_median,
        ambiguity_threshold=model.config.ambiguity_threshold,
        confidence=support_factor * proximity * fit_quality,
    )
    return _assemble_expected_transition(context, moments, count, confidence, status)


def project_expected_transition(
    model: TransitionModelDTO,
    source: SourceVPMDTO,
    action_label: str,
    field_schema: VPMFieldSchemaDTO,
    *,
    field_weights: Mapping[str, float] | None = None,
    min_support: int | None = None,
) -> ExpectedTransitionVPMDTO:
    """Project a structured future from frozen model state (no fitting).

    The same source/action/model/parameters always yield the identical
    ExpectedTransitionVPM, including its content-derived identity.
    """
    if field_schema.field_schema_id != model.field_schema_id:
        raise PerceptionTransitionProjectionError(
            "field schema does not match transition model"
        )
    try:
        validate_source_for_schema(source, field_schema)
    except Exception as exc:
        raise PerceptionTransitionProjectionError(str(exc)) from exc
    if source.encoder_spec_id != model.source_encoder_spec_id:
        raise PerceptionTransitionProjectionError(
            "source encoder spec does not match transition model"
        )
    weights = _check_field_weights(field_weights, field_schema)
    support = min_support if min_support is not None else model.config.min_support
    if support <= 0:
        raise PerceptionTransitionProjectionError("min_support must be positive")
    context = _ProjectionContext(
        source=source,
        action_label=action_label,
        field_schema=field_schema,
        model_id=model.model_id,
        training_dataset_id=model.dataset_id,
        source_encoder_spec_id=model.source_encoder_spec_id,
        before_means=_field_means(
            _source_pixels(source),
            field_schema,
            model.width,
            model.height,
            model.channels,
        ),
    )
    if isinstance(model, EmpiricalTransitionModelDTO):
        return _project_empirical(model, context, weights, support)
    if isinstance(model, CompiledTransitionModelDTO):
        return _project_compiled(model, context, weights, support)
    raise PerceptionTransitionProjectionError(
        f"unknown transition model type: {type(model).__name__}"
    )
