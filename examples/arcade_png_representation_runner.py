#!/usr/bin/env python3
"""Execution engine for the controlled PNG representation benchmark.

Builds one `MaterializedProviderEvaluationRunDTO` per representation variant
by reusing the Stage 2D provider-evaluation aggregate and the existing
`examples/local_model_zero_arcade_test.py` fixture/provider/policy machinery
exactly - this module adds only the PNG-intervention-specific provenance
construction (an ordered per-operation thumbnail-digest chain) and the
per-case execution loop. It does not reimplement `ArcadeState`, `render()`,
`Prediction`, `ScriptedProvider`, `OllamaProvider`, the policy compiler, or
the identity/episode-plan builders.

Every case's compiled-policy lookup, acceptance/rejection, and outcome
derivation is delegated to `ProviderEvaluationCaseDTO.build` (Stage 2D) -
this module only assembles the inputs it needs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image
from zeromodel.core.matrix_blob import MatrixBlob
from zeromodel.observation.visual_address import ImageObservation
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    BENCHMARK_VERSION,
    GENERATOR_VERSION,
    OBSERVATION_OPERATION_CHAIN_VERSION,
)
from zeromodel.video.domains.video_action_set.dto import CanonicalJsonDTO
from zeromodel.video.domains.video_action_set.facade import VideoActionSetFacade
from zeromodel.video.domains.video_action_set.observation_dto import (
    MaterializedObservationDTO,
    ObservationDTO,
)
from zeromodel.video.domains.video_action_set.observation_provenance_dto import (
    ObservationOperationChainDTO,
)
from zeromodel.video.domains.video_action_set.provider_evaluation_dto import (
    MaterializedProviderEvaluationRunDTO,
    ProviderConfigurationDTO,
    ProviderEvaluationCaseContext,
    ProviderEvaluationCaseDTO,
    ProviderResponseEvidence,
    build_provider_evaluation_run,
    confidence_to_basis_points,
)
from zeromodel.video.domains.video_action_set.provider_observation_dto import (
    ProviderObservationDescriptorDTO,
)

import examples.local_model_zero_arcade_test as arcade
from examples.arcade_png_interventions import ArcadePngInterventionRecipe, apply_recipe

# Held fixed for every case in every variant so the model is never told which
# representation it is looking at, and the prompt text/digest stays identical
# across every compared run (governing rule: no prompt changes between
# variants). `arcade.prompt_for("unlabelled")` is generic lane-counting
# guidance that does not presuppose printed labels, colour, or shape cues.
FIXED_PREDICT_RENDER_MODE = "unlabelled"


def _image_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _state_factors(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"row_id", "confidence"}
    }


def _decision_payload(
    decision: object, *, policy_artifact_id: str
) -> dict[str, object]:
    return dict(decision.to_dict()) | {"artifact_id": policy_artifact_id}


def build_provider_configuration(
    *, backend: str, model: str, seed: int
) -> ProviderConfigurationDTO:
    provider_kind = "fake" if backend == "fake" else "ollama"
    model_name = model if backend == "ollama" else "fake"
    model_digest = _image_digest(model_name.encode("utf-8"))
    prompt_digest = _image_digest(
        arcade.prompt_for(FIXED_PREDICT_RENDER_MODE).encode("utf-8")
    )
    return ProviderConfigurationDTO.build(
        provider_kind=provider_kind,
        model_name=model_name,
        model_digest=model_digest,
        runtime_name="ollama" if backend == "ollama" else "in-process-fake",
        protocol_version=arcade.SCHEMA_VERSION,
        prompt_digest=prompt_digest,
        seed=seed,
        inference_options=(
            {"temperature": 0.0, "num_predict": 128} if backend == "ollama" else {}
        ),
        metadata={
            "backend": backend,
            "fixed_predict_render_mode": FIXED_PREDICT_RENDER_MODE,
        },
    )


def build_scripted_replies_for_variant(
    states: Sequence[arcade.ArcadeState],
    steps_by_state: Sequence[list[tuple[Image.Image, bytes]]],
) -> dict[str, arcade.ProviderReply]:
    """A wiring-only fake reply map: keyed by the exact transformed-image
    digest that will be sent to `predict()`, each value is a perfect scripted
    reply. Fixture construction (this function) knows the ground truth;
    `predict()` itself never receives it - see `arcade.Provider`'s docstring.
    Callers merge this per-variant map with other variants' maps (digests
    never collide across variants because their pixels differ) to share one
    `ScriptedProvider` across an entire experiment."""
    replies: dict[str, arcade.ProviderReply] = {}
    for state, steps in zip(states, steps_by_state, strict=True):
        payload = {
            "tank_column": state.tank_column,
            "target_present": state.target_present,
            "target_column": -1 if state.target_column is None else state.target_column,
            "cooldown": state.cooldown,
            "confidence": 1.0,
        }
        digest = _image_digest(steps[-1][1])
        replies[digest] = arcade.ProviderReply(
            json.dumps(payload), payload, 0.0, {"backend": "fake"}
        )
    return replies


def _thumbnail_pixel_digest(image: Image.Image) -> tuple[np.ndarray, str]:
    thumbnail = np.asarray(image.convert("L").resize((28, 16)), dtype=np.uint8)
    digest = _image_digest(np.ascontiguousarray(thumbnail).tobytes(order="C"))
    return thumbnail, digest


def _build_operation_payload(
    *,
    index: int,
    operation: str,
    operation_version: str,
    input_digests: Sequence[str | None],
    parameters: Mapping[str, object],
    output_digest: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "index": index,
        "operation": operation,
        "operation_version": operation_version,
        "input_digests": list(input_digests),
        "parameters": dict(parameters),
        "output_digest": output_digest,
    }
    payload["parameter_digest"] = canonical_sha256(payload["parameters"])
    return payload | {"operation_digest": canonical_sha256(payload)}


def build_intervention_observation_chain(
    recipe: ArcadePngInterventionRecipe,
    steps: Sequence[tuple[Image.Image, bytes]],
) -> tuple[ObservationOperationChainDTO, np.ndarray, str]:
    """`steps[0]` is the untransformed canonical base render; `steps[1:]` are
    the declared recipe operations' outputs, one-to-one with
    `recipe.operations`, in order. Each step's 16x28 grayscale-thumbnail pixel
    digest becomes that operation's `output_digest` and the next operation's
    sole `input_digest` - an ordered, closure-checked chain exactly like
    `ObservationOperationChainDTO` already enforces for the single-operation
    case in `local_model_zero_arcade_test._build_observation`.
    """
    if len(steps) != len(recipe.operations) + 1:
        raise ValueError("intervention steps do not match recipe operation count")
    base_image, base_bytes = steps[0]
    thumbnail, pixel_digest = _thumbnail_pixel_digest(base_image)
    operations = [
        _build_operation_payload(
            index=0,
            operation="render_frame",
            operation_version=arcade.SCHEMA_VERSION,
            input_digests=[None],
            parameters={
                "render_mode": recipe.base_render_mode,
                "full_resolution_png_sha256": _image_digest(base_bytes),
            },
            output_digest=pixel_digest,
        )
    ]
    current_pixel_digest = pixel_digest
    final_thumbnail = thumbnail
    for offset, spec in enumerate(recipe.operations, start=1):
        image, raw_bytes = steps[offset]
        final_thumbnail, next_pixel_digest = _thumbnail_pixel_digest(image)
        operations.append(
            _build_operation_payload(
                index=offset,
                operation=spec.operation,
                operation_version=spec.operation_version,
                input_digests=[current_pixel_digest],
                parameters=dict(spec.parameters)
                | {"full_resolution_png_sha256": _image_digest(raw_bytes)},
                output_digest=next_pixel_digest,
            )
        )
        current_pixel_digest = next_pixel_digest
    chain_payload = {
        "version": OBSERVATION_OPERATION_CHAIN_VERSION,
        "operations": operations,
        "final_emitted_digest": current_pixel_digest,
    }
    chain = ObservationOperationChainDTO.from_dict(
        chain_payload | {"operation_chain_digest": canonical_sha256(chain_payload)}
    )
    return chain, final_thumbnail, current_pixel_digest


def build_intervention_observation(
    *,
    identity,
    plan,
    frame_index: int,
    recipe: ArcadePngInterventionRecipe,
    steps: Sequence[tuple[Image.Image, bytes]],
    truth_row_id: str,
    truth_action: str,
) -> MaterializedObservationDTO:
    chain, final_thumbnail, pixel_digest = build_intervention_observation_chain(
        recipe, steps
    )
    blob = MatrixBlob.from_array(
        final_thumbnail,
        dtype="uint8",
        metadata={
            "kind": "video_action_set_frame_pixels",
            "pixel_digest": pixel_digest,
        },
    )
    frame_id = f"{plan.split}:{plan.episode_id}:frame-{frame_index:02d}"
    descriptor = ProviderObservationDescriptorDTO.from_dict(
        ImageObservation(final_thumbnail, source_id=frame_id).to_descriptor()
    )
    observation = ObservationDTO(
        benchmark_version=BENCHMARK_VERSION,
        generator_version=GENERATOR_VERSION,
        benchmark_seed_digest=identity.seed_digest,
        episode_plan_digest=plan.plan_digest,
        split=plan.split,
        episode_id=plan.episode_id,
        clip_id=f"{plan.split}:{plan.episode_id}:clip",
        frame_id=frame_id,
        sequence_number=frame_index,
        event_type="frame",
        family="arcade_png_representation_benchmark",
        expected_disposition="valid",
        episode_family=plan.episode_family,
        episode_disposition=plan.episode_disposition,
        frame_disposition="valid_frame_payload",
        denominator_class=plan.denominator_class,
        expected_row=truth_row_id,
        expected_action=truth_action,
        actual_executed_action=truth_action,
        action_known=True,
        gap_declaration=None,
        observation_pixel_digest=pixel_digest,
        matrix_blob_id=blob.blob_id,
        provider_observation_descriptor=descriptor,
        provider_observation_digest=descriptor.descriptor_digest,
        operation_chain=chain,
        metadata=CanonicalJsonDTO.from_value(
            {
                "variant_id": recipe.variant_id,
                "recipe_id": recipe.recipe_id,
                "base_render_mode": recipe.base_render_mode,
                "source_full_resolution_image_sha256": _image_digest(steps[0][1]),
                "final_full_resolution_image_sha256": _image_digest(steps[-1][1]),
                "operation_count": len(recipe.operations),
                "pixel_note": (
                    "matrix_blob/provider descriptor are a deterministic 16x28 grayscale "
                    "thumbnail of the final (post-intervention) full-resolution PNG "
                    "referenced by final_full_resolution_image_sha256; "
                    "source_full_resolution_image_sha256 identifies the untransformed "
                    "canonical base render this recipe started from."
                ),
            }
        ),
        final_access_id=None,
    )
    return MaterializedObservationDTO(observation, blob)


@dataclass(frozen=True, slots=True)
class CaseExecutionResult:
    case: ProviderEvaluationCaseDTO
    record: dict[str, object]


def _run_single_case(
    *,
    case_ordinal: int,
    frame_id: str,
    truth: arcade.ArcadeState,
    truth_decision,
    policy_artifact_id: str,
    provider_configuration: ProviderConfigurationDTO,
    provider: arcade.Provider,
    final_bytes: bytes,
    confidence_threshold: float,
    reader,
) -> CaseExecutionResult:
    context = ProviderEvaluationCaseContext(
        policy_artifact_id=policy_artifact_id,
        provider_configuration_id=provider_configuration.provider_configuration_id,
    )
    case_kwargs: dict[str, object] = {
        "case_ordinal": case_ordinal,
        "frame_id": frame_id,
        "context": context,
        "expected_state": _state_factors(truth.payload()),
        "expected_decision": _decision_payload(
            truth_decision, policy_artifact_id=policy_artifact_id
        ),
    }
    record: dict[str, object] = {
        "case_ordinal": case_ordinal,
        "frame_id": frame_id,
        "truth": truth.payload(),
        "truth_action": truth_decision.action,
    }
    try:
        reply = provider.predict(final_bytes, FIXED_PREDICT_RENDER_MODE)
        prediction = arcade.Prediction.parse(reply.parsed)
        latency_us = round(reply.duration_ms * 1000)
        confidence_basis_points = confidence_to_basis_points(prediction.confidence)
        record["prediction"] = prediction.payload()
        evidence_kwargs = {
            "provider_confidence_basis_points": confidence_basis_points,
            "provider_latency_us": latency_us,
            "provider_raw_response_text": reply.raw_text,
            "provider_response_metadata": reply.metadata,
        }
        if prediction.confidence < confidence_threshold:
            case = ProviderEvaluationCaseDTO.build(
                **case_kwargs,
                accepted=False,
                evidence=ProviderResponseEvidence(
                    rejection_reason="confidence_below_threshold", **evidence_kwargs
                ),
            )
        else:
            decision = reader.read(prediction.row_id)
            case = ProviderEvaluationCaseDTO.build(
                **case_kwargs,
                accepted=True,
                predicted_state=_state_factors(prediction.payload()),
                predicted_decision=_decision_payload(
                    decision, policy_artifact_id=policy_artifact_id
                ),
                evidence=ProviderResponseEvidence(**evidence_kwargs),
            )
    except Exception as exc:  # noqa: BLE001 - recorded as a rejected case, not raised
        reason = f"{type(exc).__name__}: {exc}"
        record["error"] = reason
        case = ProviderEvaluationCaseDTO.build(
            **case_kwargs,
            accepted=False,
            evidence=ProviderResponseEvidence(rejection_reason=reason),
        )
    record["outcome"] = case.outcome
    record["case_id"] = case.case_id
    return CaseExecutionResult(case=case, record=record)


def run_variant(
    *,
    variant_id: str,
    recipe: ArcadePngInterventionRecipe,
    states: Sequence[arcade.ArcadeState],
    provider: arcade.Provider,
    provider_configuration: ProviderConfigurationDTO,
    policy_artifact_id: str,
    reader,
    facade: VideoActionSetFacade,
    identity,
    plan,
    fixture_identity: str,
    case_mode: str,
    confidence_threshold: float,
    metadata: Mapping[str, object],
    on_case_steps: Callable[
        [int, arcade.ArcadeState, list[tuple[Image.Image, bytes]]], None
    ]
    | None = None,
) -> tuple[MaterializedProviderEvaluationRunDTO, list[dict[str, object]]]:
    cases: list[ProviderEvaluationCaseDTO] = []
    records: list[dict[str, object]] = []
    for frame_index, state in enumerate(states):
        base_image = arcade.render(state, recipe.base_render_mode)
        steps = apply_recipe(recipe, base_image)
        if on_case_steps is not None:
            on_case_steps(frame_index, state, steps)
        truth_decision = reader.read(state.row_id)
        observation = build_intervention_observation(
            identity=identity,
            plan=plan,
            frame_index=frame_index,
            recipe=recipe,
            steps=steps,
            truth_row_id=state.row_id,
            truth_action=truth_decision.action,
        )
        saved_observation = facade.save_materialized_observation(observation)
        result = _run_single_case(
            case_ordinal=frame_index,
            frame_id=saved_observation.frame_id,
            truth=state,
            truth_decision=truth_decision,
            policy_artifact_id=policy_artifact_id,
            provider_configuration=provider_configuration,
            provider=provider,
            final_bytes=steps[-1][1],
            confidence_threshold=confidence_threshold,
            reader=reader,
        )
        result.record["final_image_sha256"] = _image_digest(steps[-1][1])
        result.record["step_image_sha256"] = [_image_digest(raw) for _, raw in steps]
        cases.append(result.case)
        records.append(result.record)

    materialized_run = build_provider_evaluation_run(
        fixture_identity=fixture_identity,
        provider_configuration=provider_configuration,
        policy_artifact_id=policy_artifact_id,
        case_mode=case_mode,
        representation_mode=variant_id,
        cases=cases,
        metadata=metadata,
    )
    saved_run = facade.save_provider_evaluation_run(materialized_run)
    reloaded_run = facade.get_materialized_provider_evaluation_run(saved_run.run.run_id)
    if reloaded_run is None or reloaded_run != materialized_run:
        raise RuntimeError(
            "provider evaluation aggregate failed to reload identically after save"
        )
    return saved_run, records


def find_resumable_run(
    *,
    facade: VideoActionSetFacade,
    fixture_identity: str,
    provider_configuration_id: str,
    policy_artifact_id: str,
    case_mode: str,
    variant_id: str,
) -> MaterializedProviderEvaluationRunDTO | None:
    """Look up an already-saved run compatible with the requested variant.

    Returns `None` when nothing matches (a fresh run must be executed).
    Raises if more than one candidate is found - `--resume` must never guess
    which of several incompatible-looking prior runs to reuse.
    """
    candidates = facade.list_provider_evaluation_runs(
        fixture_identity=fixture_identity,
        policy_artifact_id=policy_artifact_id,
        case_mode=case_mode,
        representation_mode=variant_id,
    )
    matching = [
        run
        for run in candidates
        if run.provider_configuration.provider_configuration_id
        == provider_configuration_id
    ]
    if not matching:
        return None
    if len(matching) > 1:
        raise RuntimeError(
            f"--resume found {len(matching)} candidate runs for variant "
            f"{variant_id!r}; refusing to guess which one to reuse"
        )
    return facade.get_materialized_provider_evaluation_run(matching[0].run_id)


__all__ = [
    "FIXED_PREDICT_RENDER_MODE",
    "CaseExecutionResult",
    "build_intervention_observation",
    "build_intervention_observation_chain",
    "build_provider_configuration",
    "build_scripted_replies_for_variant",
    "find_resumable_run",
    "run_variant",
]
