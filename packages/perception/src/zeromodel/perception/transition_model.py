"""Compiled action-conditioned transition memory.

Strict API split between fitting and inference:

- ``fit_action_conditioned_transition_model`` compiles historical
  before/action/after triples (P2 interactions with an authoritative next
  source) into a frozen empirical nearest-neighbour memory. Per eligible
  transition the existing P18A evidence builder is reused; action, field
  schema, and sequence identity are preserved.
- ``fit_compiled_transition_model`` compiles the same triples into small
  per-action ridge maps (before field means -> delta field means).
- ``project_expected_transition`` operates only on already-frozen model
  state. It never fits, updates, or caches anything.

NumPy only. No video generation, diffusion, or large neural machinery.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np

from .dataset import PerceptionDatasetManifestDTO
from .expectations import PerceptionRegionAnnotationDTO
from .fields import VPMFieldSchemaDTO, validate_source_for_schema
from .representation import SourceVPMDTO
from .transition_evidence import build_transition_evidence_vpm

TRANSITION_MODEL_CONFIG_VERSION: Final = "perception-transition-model-config/1"
TRANSITION_TRAINING_EXAMPLE_VERSION: Final = "perception-transition-training-example/1"
EMPIRICAL_TRANSITION_MODEL_VERSION: Final = "perception-empirical-transition-model/1"
COMPILED_TRANSITION_MODEL_VERSION: Final = "perception-compiled-transition-model/1"


class PerceptionTransitionModelError(ValueError):
    """Raised when transition memory cannot be compiled or projected."""


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


@dataclass(frozen=True)
class TransitionModelConfigDTO:
    """Bounded deterministic memory compilation and projection contract."""

    neighbor_count: int = 16
    min_support: int = 5
    ood_spread_factor: float = 3.0
    ambiguity_threshold: float = 0.2
    change_epsilon: float = 1e-3
    version: str = TRANSITION_MODEL_CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.neighbor_count <= 0:
            raise PerceptionTransitionModelError("neighbor_count must be positive")
        if self.min_support <= 0:
            raise PerceptionTransitionModelError("min_support must be positive")
        if not np.isfinite(self.ood_spread_factor) or self.ood_spread_factor <= 0:
            raise PerceptionTransitionModelError(
                "ood_spread_factor must be finite and positive"
            )
        if not np.isfinite(self.ambiguity_threshold) or self.ambiguity_threshold < 0:
            raise PerceptionTransitionModelError(
                "ambiguity_threshold must be finite and non-negative"
            )
        if not np.isfinite(self.change_epsilon) or self.change_epsilon < 0:
            raise PerceptionTransitionModelError(
                "change_epsilon must be finite and non-negative"
            )
        if self.version != TRANSITION_MODEL_CONFIG_VERSION:
            raise PerceptionTransitionModelError(
                "unsupported transition model config version"
            )

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "ambiguity_threshold": self.ambiguity_threshold,
            "change_epsilon": self.change_epsilon,
            "min_support": self.min_support,
            "neighbor_count": self.neighbor_count,
            "ood_spread_factor": self.ood_spread_factor,
            "version": self.version,
        }


def _round12(value: float) -> float:
    return round(float(value), 12)


def _rounded(values: object) -> object:
    if isinstance(values, bool):
        return values
    if isinstance(values, (int, float)):
        return _round12(float(values))
    if isinstance(values, (tuple, list)):
        return [_rounded(value) for value in values]
    return values


@dataclass(frozen=True)
class TransitionTrainingExampleDTO:
    """One eligible historical interaction with its P18A evidence identity.

    Large arrays are not copied: the example references the stable
    transition-evidence identity and carries only the before-pixel payload
    (needed for neighbour distance, mirroring P3 precedent) plus compact
    per-field delta moments measured once at fit time.
    """

    interaction_id: str
    sequence_id: str
    before_source_vpm_id: str
    after_source_vpm_id: str
    action_label: str
    transition_evidence_id: str
    width: int
    height: int
    channels: int
    before_pixels: bytes
    signed_deltas: tuple[float, ...]
    absolute_deltas: tuple[float, ...]
    changed_fractions: tuple[float, ...]
    version: str = TRANSITION_TRAINING_EXAMPLE_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.interaction_id,
                self.sequence_id,
                self.before_source_vpm_id,
                self.after_source_vpm_id,
                self.action_label,
                self.transition_evidence_id,
            )
        ):
            raise PerceptionTransitionModelError(
                "training example identities must be non-empty"
            )
        expected = self.width * self.height * self.channels
        if expected <= 0 or len(self.before_pixels) != expected:
            raise PerceptionTransitionModelError(
                "training example pixel payload has invalid size"
            )
        field_count = len(self.signed_deltas)
        if (
            field_count == 0
            or len(self.absolute_deltas) != field_count
            or len(self.changed_fractions) != field_count
        ):
            raise PerceptionTransitionModelError(
                "training example delta moments must align across fields"
            )
        if self.version != TRANSITION_TRAINING_EXAMPLE_VERSION:
            raise PerceptionTransitionModelError(
                "unsupported transition training example version"
            )


@dataclass(frozen=True)
class EmpiricalTransitionModelDTO:
    """Frozen empirical action-conditioned future memory (nearest neighbour)."""

    model_id: str
    dataset_id: str
    action_schema_id: str
    source_encoder_spec_id: str
    field_schema_id: str
    change_threshold: int
    width: int
    height: int
    channels: int
    action_labels: tuple[str, ...]
    action_counts: tuple[int, ...]
    action_centroids: tuple[tuple[float, ...], ...]
    action_spreads: tuple[float, ...]
    action_typical_nn_distances: tuple[float, ...]
    examples: tuple[TransitionTrainingExampleDTO, ...]
    config: TransitionModelConfigDTO
    training_split: str
    version: str = EMPIRICAL_TRANSITION_MODEL_VERSION

    def __post_init__(self) -> None:
        _validate_model_identities(self, EMPIRICAL_TRANSITION_MODEL_VERSION)

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "action_centroids": _rounded(self.action_centroids),
            "action_counts": list(self.action_counts),
            "action_schema_id": self.action_schema_id,
            "action_spreads": _rounded(self.action_spreads),
            "action_typical_nn_distances": _rounded(self.action_typical_nn_distances),
            "change_threshold": self.change_threshold,
            "config": self.config.canonical_payload(),
            "dataset_id": self.dataset_id,
            "examples": [
                {
                    "action_label": item.action_label,
                    "before_pixel_digest": _digest(item.before_pixels),
                    "interaction_id": item.interaction_id,
                    "transition_evidence_id": item.transition_evidence_id,
                }
                for item in self.examples
            ],
            "field_schema_id": self.field_schema_id,
            "kind": "empirical",
            "shape": [self.width, self.height, self.channels],
            "source_encoder_spec_id": self.source_encoder_spec_id,
            "training_split": self.training_split,
            "version": self.version,
        }


@dataclass(frozen=True)
class CompiledTransitionModelDTO:
    """Frozen ridge action-conditioned future memory (compact linear maps)."""

    model_id: str
    dataset_id: str
    action_schema_id: str
    source_encoder_spec_id: str
    field_schema_id: str
    change_threshold: int
    width: int
    height: int
    channels: int
    action_labels: tuple[str, ...]
    action_counts: tuple[int, ...]
    action_centroids: tuple[tuple[float, ...], ...]
    action_spreads: tuple[float, ...]
    action_typical_nn_distances: tuple[float, ...]
    action_has_linear_fit: tuple[bool, ...]
    action_coefficients: tuple[tuple[tuple[float, ...], ...], ...]
    action_residual_std: tuple[tuple[float, ...], ...]
    ridge_alpha: float
    examples: tuple[TransitionTrainingExampleDTO, ...]
    config: TransitionModelConfigDTO
    training_split: str
    version: str = COMPILED_TRANSITION_MODEL_VERSION

    def __post_init__(self) -> None:
        _validate_model_identities(self, COMPILED_TRANSITION_MODEL_VERSION)
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha < 0:
            raise PerceptionTransitionModelError(
                "ridge_alpha must be finite and non-negative"
            )
        actions = len(self.action_labels)
        if not (
            len(self.action_has_linear_fit) == actions
            and len(self.action_coefficients) == actions
            and len(self.action_residual_std) == actions
        ):
            raise PerceptionTransitionModelError(
                "compiled per-action payloads must align with action labels"
            )

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "action_centroids": _rounded(self.action_centroids),
            "action_coefficients": _rounded(self.action_coefficients),
            "action_counts": list(self.action_counts),
            "action_has_linear_fit": list(self.action_has_linear_fit),
            "action_residual_std": _rounded(self.action_residual_std),
            "action_schema_id": self.action_schema_id,
            "action_spreads": _rounded(self.action_spreads),
            "action_typical_nn_distances": _rounded(self.action_typical_nn_distances),
            "change_threshold": self.change_threshold,
            "config": self.config.canonical_payload(),
            "dataset_id": self.dataset_id,
            "examples": [
                {
                    "action_label": item.action_label,
                    "before_pixel_digest": _digest(item.before_pixels),
                    "interaction_id": item.interaction_id,
                    "transition_evidence_id": item.transition_evidence_id,
                }
                for item in self.examples
            ],
            "field_schema_id": self.field_schema_id,
            "kind": "ridge",
            "ridge_alpha": self.ridge_alpha,
            "shape": [self.width, self.height, self.channels],
            "source_encoder_spec_id": self.source_encoder_spec_id,
            "training_split": self.training_split,
            "version": self.version,
        }


def _validate_model_identities(
    model: EmpiricalTransitionModelDTO | CompiledTransitionModelDTO,
    version: str,
) -> None:
    if not model.model_id or not model.dataset_id or not model.action_schema_id:
        raise PerceptionTransitionModelError("model identities must be non-empty")
    if not model.examples:
        raise PerceptionTransitionModelError("transition model requires examples")
    ids = tuple(item.interaction_id for item in model.examples)
    if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
        raise PerceptionTransitionModelError("model examples must be unique and sorted")
    if model.action_labels != tuple(sorted(set(model.action_labels))):
        raise PerceptionTransitionModelError("action_labels must be unique and sorted")
    if not (
        len(model.action_counts) == len(model.action_labels)
        and len(model.action_centroids) == len(model.action_labels)
        and len(model.action_spreads) == len(model.action_labels)
        and len(model.action_typical_nn_distances) == len(model.action_labels)
    ):
        raise PerceptionTransitionModelError(
            "per-action statistics must align with action labels"
        )
    if model.version != version:
        raise PerceptionTransitionModelError("unsupported transition model version")
    expected_id = _digest(_canonical_json(model.canonical_payload()))
    if model.model_id != expected_id:
        raise PerceptionTransitionModelError(
            "model identity does not bind the compiled runtime payload"
        )


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


def _eligible_transitions(
    manifest: PerceptionDatasetManifestDTO,
    source_vpms: Mapping[str, SourceVPMDTO],
    field_schema: VPMFieldSchemaDTO,
    training_split: str,
    change_threshold: int,
    annotations: tuple[PerceptionRegionAnnotationDTO, ...],
) -> tuple[list[TransitionTrainingExampleDTO], tuple[int, int, int], str]:
    if training_split not in {"train", "validation", "test", "all"}:
        raise PerceptionTransitionModelError(
            "training_split must be train, validation, test, or all"
        )
    interactions = {item.interaction_id: item for item in manifest.interactions}
    selected = tuple(
        assignment.interaction_id
        for assignment in manifest.split_assignments
        if training_split == "all" or assignment.split == training_split
    )
    if not selected:
        raise PerceptionTransitionModelError(
            f"dataset contains no {training_split!r} examples"
        )
    examples: list[TransitionTrainingExampleDTO] = []
    shape: tuple[int, int, int] | None = None
    encoder_spec_id: str | None = None
    for interaction_id in sorted(selected):
        interaction = interactions[interaction_id]
        if interaction.next_source_vpm_id is None:
            continue  # not eligible: no authoritative next state
        try:
            before = source_vpms[interaction.source_vpm_id]
            after = source_vpms[interaction.next_source_vpm_id]
        except KeyError as exc:
            raise PerceptionTransitionModelError(
                f"missing SourceVPMDTO for interaction {interaction_id}"
            ) from exc
        if before.pixel_digest != interaction.source_pixel_digest:
            raise PerceptionTransitionModelError(
                "source pixel identity disagrees with interaction"
            )
        if after.source_vpm_id != interaction.next_source_vpm_id:
            raise PerceptionTransitionModelError(
                "after source identity disagrees with interaction"
            )
        for source in (before, after):
            current = (source.width, source.height, source.channels)
            if shape is None:
                shape = current
                encoder_spec_id = source.encoder_spec_id
            elif current != shape:
                raise PerceptionTransitionModelError(
                    "transition model requires one source VPM shape"
                )
            elif source.encoder_spec_id != encoder_spec_id:
                raise PerceptionTransitionModelError(
                    "transition model requires one source encoder spec"
                )
        try:
            validate_source_for_schema(before, field_schema)
            validate_source_for_schema(after, field_schema)
        except Exception as exc:
            raise PerceptionTransitionModelError(str(exc)) from exc
        evidence = build_transition_evidence_vpm(
            before,
            after,
            field_schema,
            annotations=annotations,
            change_threshold=change_threshold,
        )
        by_id = {item.field_id: item for item in evidence.fields}
        ordered = tuple(field.field_id for field in field_schema.fields)
        examples.append(
            TransitionTrainingExampleDTO(
                interaction_id=interaction.interaction_id,
                sequence_id=interaction.sequence_id,
                before_source_vpm_id=before.source_vpm_id,
                after_source_vpm_id=after.source_vpm_id,
                action_label=interaction.action_label,
                transition_evidence_id=evidence.transition_evidence_id,
                width=before.width,
                height=before.height,
                channels=before.channels,
                before_pixels=_source_pixels(before),
                signed_deltas=tuple(
                    by_id[field_id].mean_signed_change for field_id in ordered
                ),
                absolute_deltas=tuple(
                    by_id[field_id].mean_absolute_change for field_id in ordered
                ),
                changed_fractions=tuple(
                    by_id[field_id].changed_fraction for field_id in ordered
                ),
            )
        )
    if not examples:
        raise PerceptionTransitionModelError(
            "no eligible transition with an authoritative next state"
        )
    assert shape is not None and encoder_spec_id is not None
    return examples, shape, encoder_spec_id


def _per_action_stats(
    examples: list[TransitionTrainingExampleDTO],
    labels: tuple[str, ...],
    width: int,
    height: int,
    channels: int,
) -> tuple[
    tuple[int, ...], tuple[tuple[float, ...], ...], tuple[float, ...], tuple[float, ...]
]:
    counts: list[int] = []
    centroids: list[tuple[float, ...]] = []
    spreads: list[float] = []
    typical_nn: list[float] = []
    for label in labels:
        rows = np.stack(
            [
                _pixel_array(item.before_pixels, width, height, channels).reshape(-1)
                for item in examples
                if item.action_label == label
            ]
        )
        centroid = rows.mean(axis=0)
        # Spread uses normalized mean absolute distance: the same units as
        # the neighbour distance, so the OOD comparison is dimensionally
        # consistent.
        spread = float(np.mean(np.abs(rows - centroid[None, :])))
        counts.append(int(rows.shape[0]))
        centroids.append(tuple(float(value) for value in centroid))
        spreads.append(spread)
        if rows.shape[0] < 2:
            typical_nn.append(0.0)
            continue
        nearest: list[float] = []
        for row in range(rows.shape[0]):
            others = np.delete(rows, row, axis=0)
            nearest.append(float(np.mean(np.abs(others - rows[row][None, :]))))
        # Typical local spacing: mean distance to the nearest same-action
        # neighbour. Small for dense (possibly multimodal) support, large
        # for sparse support — the local-density OOD scale.
        typical_nn.append(float(sum(nearest) / len(nearest)))
    return tuple(counts), tuple(centroids), tuple(spreads), tuple(typical_nn)


def _model_payload(
    manifest: PerceptionDatasetManifestDTO,
    field_schema: VPMFieldSchemaDTO,
    config: TransitionModelConfigDTO,
    training_split: str,
    change_threshold: int,
    examples: list[TransitionTrainingExampleDTO],
    action_counts: tuple[int, ...],
    fingerprint: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "action_counts": list(action_counts),
        "action_schema_id": manifest.action_schema_id,
        "change_threshold": change_threshold,
        "config": config.canonical_payload(),
        "dataset_id": manifest.dataset_id,
        "examples": [
            {
                "action_label": item.action_label,
                "before_pixel_digest": _digest(item.before_pixels),
                "interaction_id": item.interaction_id,
                "transition_evidence_id": item.transition_evidence_id,
            }
            for item in examples
        ],
        "field_schema_id": field_schema.field_schema_id,
        "training_split": training_split,
    }
    payload.update(fingerprint)
    return payload


def fit_action_conditioned_transition_model(
    manifest: PerceptionDatasetManifestDTO,
    source_vpms: Mapping[str, SourceVPMDTO],
    field_schema: VPMFieldSchemaDTO,
    *,
    config: TransitionModelConfigDTO | None = None,
    training_split: str = "train",
    change_threshold: int = 1,
    annotations: tuple[PerceptionRegionAnnotationDTO, ...] = (),
) -> EmpiricalTransitionModelDTO:
    """Compile eligible before/action/after triples into frozen memory."""
    resolved = config or TransitionModelConfigDTO()
    examples, shape, encoder_spec_id = _eligible_transitions(
        manifest,
        source_vpms,
        field_schema,
        training_split,
        change_threshold,
        annotations,
    )
    width, height, channels = shape
    labels = tuple(sorted({item.action_label for item in examples}))
    counts, centroids, spreads, typical_nn = _per_action_stats(
        examples, labels, width, height, channels
    )
    return EmpiricalTransitionModelDTO(
        model_id=_digest(
            _canonical_json(
                _model_payload(
                    manifest,
                    field_schema,
                    resolved,
                    training_split,
                    change_threshold,
                    examples,
                    counts,
                    {
                        "kind": "empirical",
                        "shape": list(shape),
                        "source_encoder_spec_id": encoder_spec_id,
                        "version": EMPIRICAL_TRANSITION_MODEL_VERSION,
                        "action_centroids": _rounded(centroids),
                        "action_spreads": _rounded(spreads),
                        "action_typical_nn_distances": _rounded(typical_nn),
                    },
                )
            )
        ),
        dataset_id=manifest.dataset_id,
        action_schema_id=manifest.action_schema_id,
        source_encoder_spec_id=encoder_spec_id,
        field_schema_id=field_schema.field_schema_id,
        change_threshold=change_threshold,
        width=width,
        height=height,
        channels=channels,
        action_labels=labels,
        action_counts=counts,
        action_centroids=centroids,
        action_spreads=spreads,
        action_typical_nn_distances=typical_nn,
        examples=tuple(examples),
        config=resolved,
        training_split=training_split,
    )


def _solve_ridge_action(
    before_means: np.ndarray,
    deltas: np.ndarray,
    count_fields: int,
    ridge_alpha: float,
    action_label: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Closed-form ridge before-to-delta map with unregularized intercept."""
    count = before_means.shape[0]
    if count >= 2:
        design = np.concatenate([before_means, np.ones((count, 1))], axis=1)
        penalty = np.eye(count_fields + 1) * ridge_alpha
        penalty[count_fields, count_fields] = 0.0
        try:
            solution = np.linalg.solve(design.T @ design + penalty, design.T @ deltas)
        except np.linalg.LinAlgError as exc:
            raise PerceptionTransitionModelError(
                f"ridge solve failed for {action_label!r}"
            ) from exc
        return solution, design @ solution, True
    solution = np.zeros((count_fields + 1, count_fields))
    solution[count_fields, :] = deltas.mean(axis=0)
    fitted = np.tile(deltas.mean(axis=0)[None, :], (count, 1))
    return solution, fitted, False


def _action_fit_matrices(
    subset: list[TransitionTrainingExampleDTO],
    field_schema: VPMFieldSchemaDTO,
    field_ids: tuple[str, ...],
    width: int,
    height: int,
    channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    before_means = np.stack(
        [
            np.array(
                [
                    _field_means(
                        item.before_pixels, field_schema, width, height, channels
                    )[field_id]
                    for field_id in field_ids
                ]
            )
            for item in subset
        ]
    )
    deltas = np.stack(
        [np.array(item.signed_deltas, dtype=np.float64) for item in subset]
    )
    return before_means, deltas


def fit_compiled_transition_model(
    manifest: PerceptionDatasetManifestDTO,
    source_vpms: Mapping[str, SourceVPMDTO],
    field_schema: VPMFieldSchemaDTO,
    *,
    config: TransitionModelConfigDTO | None = None,
    training_split: str = "train",
    change_threshold: int = 1,
    ridge_alpha: float = 1.0,
    annotations: tuple[PerceptionRegionAnnotationDTO, ...] = (),
) -> CompiledTransitionModelDTO:
    """Compile per-action linear before-to-delta maps (closed-form ridge)."""
    resolved = config or TransitionModelConfigDTO()
    if not np.isfinite(ridge_alpha) or ridge_alpha < 0:
        raise PerceptionTransitionModelError(
            "ridge_alpha must be finite and non-negative"
        )
    examples, shape, encoder_spec_id = _eligible_transitions(
        manifest,
        source_vpms,
        field_schema,
        training_split,
        change_threshold,
        annotations,
    )
    width, height, channels = shape
    field_ids = tuple(field.field_id for field in field_schema.fields)
    count_fields = len(field_ids)
    labels = tuple(sorted({item.action_label for item in examples}))
    counts, centroids, spreads, typical_nn = _per_action_stats(
        examples, labels, width, height, channels
    )
    coefficients: list[tuple[tuple[float, ...], ...]] = []
    residuals: list[tuple[float, ...]] = []
    linear_flags: list[bool] = []
    for label in labels:
        subset = [item for item in examples if item.action_label == label]
        before_means, deltas = _action_fit_matrices(
            subset, field_schema, field_ids, width, height, channels
        )
        solution, fitted, is_linear = _solve_ridge_action(
            before_means, deltas, count_fields, ridge_alpha, label
        )
        linear_flags.append(is_linear)
        residuals.append(tuple(float(value) for value in (deltas - fitted).std(axis=0)))
        coefficients.append(
            tuple(
                tuple(float(value) for value in solution.T[row])
                for row in range(count_fields)
            )
        )
    return CompiledTransitionModelDTO(
        model_id=_digest(
            _canonical_json(
                _model_payload(
                    manifest,
                    field_schema,
                    resolved,
                    training_split,
                    change_threshold,
                    examples,
                    counts,
                    {
                        "kind": "ridge",
                        "ridge_alpha": float(ridge_alpha),
                        "shape": list(shape),
                        "source_encoder_spec_id": encoder_spec_id,
                        "version": COMPILED_TRANSITION_MODEL_VERSION,
                        "action_centroids": _rounded(centroids),
                        "action_spreads": _rounded(spreads),
                        "action_typical_nn_distances": _rounded(typical_nn),
                        "action_has_linear_fit": list(linear_flags),
                        "action_coefficients": _rounded(coefficients),
                        "action_residual_std": _rounded(residuals),
                    },
                )
            )
        ),
        dataset_id=manifest.dataset_id,
        action_schema_id=manifest.action_schema_id,
        source_encoder_spec_id=encoder_spec_id,
        field_schema_id=field_schema.field_schema_id,
        change_threshold=change_threshold,
        width=width,
        height=height,
        channels=channels,
        action_labels=labels,
        action_counts=counts,
        action_centroids=centroids,
        action_spreads=spreads,
        action_typical_nn_distances=typical_nn,
        examples=tuple(examples),
        config=resolved,
        training_split=training_split,
        action_has_linear_fit=tuple(linear_flags),
        action_coefficients=tuple(coefficients),
        action_residual_std=tuple(residuals),
        ridge_alpha=float(ridge_alpha),
    )
