from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...artifact import VPMValidationError
from ...domains.video_action_set.canonical_json import canonical_json_text
from ...domains.video_action_set.contracts import BENCHMARK_VERSION, GENERATOR_VERSION
from ...domains.video_action_set.dto import CanonicalJsonDTO
from ...domains.video_action_set.observation_dto import (
    MaterializedObservationDTO,
    ObservationDTO,
    ObservationOperationChainDTO,
    ObservationOperationDTO,
    ProviderObservationDescriptorDTO,
)
from ...domains.video_action_set.store import raise_observation_sequence_conflict
from ...matrix_blob import MatrixBlob
from ..orm.video_action_set import (
    EpisodePlanORM,
    MatrixBlobORM,
    ObservationORM,
    ObservationOperationChainORM,
    ObservationOperationInputORM,
    ObservationOperationORM,
)


def to_matrix_blob_orm(blob: MatrixBlob) -> MatrixBlobORM:
    return MatrixBlobORM(
        blob_id=blob.blob_id,
        version=blob.version,
        dtype=blob.dtype,
        shape_json=canonical_json_text(list(blob.shape)),
        scale=blob.scale,
        zero_point=blob.zero_point,
        metadata_json=canonical_json_text(dict(blob.metadata)),
        data=blob.data,
        byte_length=len(blob.data),
    )


def to_matrix_blob(row: MatrixBlobORM) -> MatrixBlob:
    if row.byte_length != len(row.data):
        raise VPMValidationError("matrix blob byte length mismatch")
    return MatrixBlob(
        dtype=row.dtype,
        shape=_json_value(row.shape_json, "matrix blob shape mismatch"),
        data=row.data,
        scale=row.scale,
        zero_point=row.zero_point,
        metadata=_json_mapping(row.metadata_json, "matrix blob metadata mismatch"),
        version=row.version,
        blob_id=row.blob_id,
    )


def to_observation_orm(observation: ObservationDTO) -> ObservationORM:
    return ObservationORM(
        frame_id=observation.frame_id,
        benchmark_seed_digest=observation.benchmark_seed_digest,
        episode_id=observation.episode_id,
        episode_plan_digest=observation.episode_plan_digest,
        split=observation.split,
        clip_id=observation.clip_id,
        sequence_number=observation.sequence_number,
        event_type=observation.event_type,
        family=observation.family,
        expected_disposition=observation.expected_disposition,
        episode_family=observation.episode_family,
        episode_disposition=observation.episode_disposition,
        frame_disposition=observation.frame_disposition,
        denominator_class=observation.denominator_class,
        expected_row=observation.expected_row,
        expected_action=observation.expected_action,
        actual_executed_action=observation.actual_executed_action,
        action_known=observation.action_known,
        gap_declaration_json=_canonical_or_none(observation.gap_declaration),
        observation_pixel_digest=observation.observation_pixel_digest,
        matrix_blob_id=observation.matrix_blob_id,
        final_access_id=observation.final_access_id,
        provider_descriptor_json=_provider_descriptor_json(observation),
        provider_observation_digest=observation.provider_observation_digest,
        metadata_json=observation.metadata.canonical_text,
        operation_chain_digest=observation.operation_chain.operation_chain_digest,
    )


def to_observation_dto(
    session: Session,
    row: ObservationORM,
    *,
    chain: ObservationOperationChainDTO | None = None,
    plan: EpisodePlanORM | None = None,
) -> ObservationDTO:
    chain = chain if chain is not None else chain_for_frame(session, row.frame_id)
    validate_observation_row(session, row, chain, plan=plan)
    return ObservationDTO(
        benchmark_version=BENCHMARK_VERSION,
        generator_version=GENERATOR_VERSION,
        benchmark_seed_digest=row.benchmark_seed_digest,
        episode_plan_digest=row.episode_plan_digest,
        split=row.split,
        episode_id=row.episode_id,
        clip_id=row.clip_id,
        frame_id=row.frame_id,
        sequence_number=row.sequence_number,
        event_type=row.event_type,
        family=row.family,
        expected_disposition=row.expected_disposition,
        episode_family=row.episode_family,
        episode_disposition=row.episode_disposition,
        frame_disposition=row.frame_disposition,
        denominator_class=row.denominator_class,
        expected_row=row.expected_row,
        expected_action=row.expected_action,
        actual_executed_action=row.actual_executed_action,
        action_known=row.action_known,
        gap_declaration=_canonical_from_text_or_none(row.gap_declaration_json),
        observation_pixel_digest=row.observation_pixel_digest,
        matrix_blob_id=row.matrix_blob_id,
        provider_observation_descriptor=_provider_descriptor_from_json(
            row.provider_descriptor_json
        ),
        provider_observation_digest=row.provider_observation_digest,
        operation_chain=chain,
        metadata=CanonicalJsonDTO(row.metadata_json),
        final_access_id=row.final_access_id,
    )


def validate_observation_row(
    session: Session,
    row: ObservationORM,
    chain: ObservationOperationChainDTO,
    *,
    plan: EpisodePlanORM | None = None,
) -> None:
    if row.operation_chain_digest != chain.operation_chain_digest:
        raise VPMValidationError("observation operation chain mismatch")
    plan = plan if plan is not None else session.get(EpisodePlanORM, row.episode_id)
    if plan is None:
        raise VPMValidationError("observation references unknown episode plan")
    if (
        row.episode_plan_digest != plan.plan_digest
        or row.benchmark_seed_digest != plan.benchmark_seed_digest
        or row.split != plan.split
        or row.episode_family != plan.family_label
        or row.episode_disposition != plan.episode_disposition
        or row.denominator_class != plan.denominator_class
    ):
        raise VPMValidationError("observation episode plan mismatch")


def to_operation_chain_orm(
    observation: ObservationDTO,
) -> ObservationOperationChainORM:
    chain = observation.operation_chain
    return ObservationOperationChainORM(
        frame_id=observation.frame_id,
        version=chain.version,
        final_emitted_digest=chain.final_emitted_digest,
        operation_chain_digest=chain.operation_chain_digest,
        operation_count=len(chain.operations),
    )


def to_operation_orms(
    observation: ObservationDTO,
) -> tuple[ObservationOperationORM, ...]:
    return tuple(
        ObservationOperationORM(
            frame_id=observation.frame_id,
            operation_index=operation.index,
            operation=operation.operation,
            operation_version=operation.operation_version,
            parameters_json=operation.parameters.canonical_text,
            parameter_digest=operation.parameter_digest,
            output_digest=operation.output_digest,
            operation_digest=operation.operation_digest,
        )
        for operation in observation.operation_chain.operations
    )


def to_operation_input_orms(
    observation: ObservationDTO,
) -> tuple[ObservationOperationInputORM, ...]:
    return tuple(
        ObservationOperationInputORM(
            frame_id=observation.frame_id,
            operation_index=operation.index,
            input_index=input_index,
            input_digest=input_digest,
        )
        for operation in observation.operation_chain.operations
        for input_index, input_digest in enumerate(operation.input_digests)
    )


def to_operation_chain_dto(
    session: Session,
    row: ObservationOperationChainORM,
) -> ObservationOperationChainDTO:
    operations = operation_rows_for_frame(session, row.frame_id)
    return _operation_chain_from_rows(row, operations)


def chain_for_frame(
    session: Session,
    frame_id: str,
) -> ObservationOperationChainDTO:
    row = session.get(ObservationOperationChainORM, frame_id)
    if row is None:
        raise VPMValidationError("observation operation chain mismatch")
    return to_operation_chain_dto(session, row)


def operation_rows_for_frame(
    session: Session,
    frame_id: str,
) -> tuple[ObservationOperationDTO, ...]:
    rows = session.scalars(
        select(ObservationOperationORM)
        .where(ObservationOperationORM.frame_id == frame_id)
        .order_by(ObservationOperationORM.operation_index)
    ).all()
    inputs_by_operation = _input_digests_by_operation(
        session.scalars(
            select(ObservationOperationInputORM)
            .where(ObservationOperationInputORM.frame_id == frame_id)
            .order_by(
                ObservationOperationInputORM.operation_index,
                ObservationOperationInputORM.input_index,
            )
        ).all()
    )
    return tuple(
        _to_operation_dto(
            row,
            inputs_by_operation.get((row.frame_id, row.operation_index), ()),
        )
        for row in rows
    )


def observations_from_rows(
    session: Session,
    rows: Sequence[ObservationORM],
) -> tuple[ObservationDTO, ...]:
    if not rows:
        return ()
    frame_ids = [row.frame_id for row in rows]
    episode_ids = {row.episode_id for row in rows}
    chains = chains_for_frames(session, frame_ids)
    plans = {
        row.episode_id: row
        for row in session.scalars(
            select(EpisodePlanORM).where(EpisodePlanORM.episode_id.in_(episode_ids))
        ).all()
    }
    return tuple(
        to_observation_dto(
            session,
            row,
            chain=chains[row.frame_id],
            plan=plans.get(row.episode_id),
        )
        for row in rows
    )


def materialized_observations_from_rows(
    session: Session,
    rows: Sequence[ObservationORM],
) -> tuple[MaterializedObservationDTO, ...]:
    observations = observations_from_rows(session, rows)
    blob_ids = {
        observation.matrix_blob_id
        for observation in observations
        if observation.matrix_blob_id is not None
    }
    if not blob_ids:
        return tuple(
            MaterializedObservationDTO(
                observation,
                None,
                final_access_id=observation.final_access_id,
            )
            for observation in observations
        )
    blobs = {
        row.blob_id: to_matrix_blob(row)
        for row in session.scalars(
            select(MatrixBlobORM).where(MatrixBlobORM.blob_id.in_(blob_ids))
        ).all()
    }
    return tuple(
        MaterializedObservationDTO(
            observation,
            None
            if observation.matrix_blob_id is None
            else blobs.get(observation.matrix_blob_id),
            final_access_id=observation.final_access_id,
        )
        for observation in observations
    )


def chains_for_frames(
    session: Session,
    frame_ids: Sequence[str],
) -> dict[str, ObservationOperationChainDTO]:
    if not frame_ids:
        return {}
    chain_rows = {
        row.frame_id: row
        for row in session.scalars(
            select(ObservationOperationChainORM).where(
                ObservationOperationChainORM.frame_id.in_(frame_ids)
            )
        ).all()
    }
    operation_rows = session.scalars(
        select(ObservationOperationORM)
        .where(ObservationOperationORM.frame_id.in_(frame_ids))
        .order_by(
            ObservationOperationORM.frame_id,
            ObservationOperationORM.operation_index,
        )
    ).all()
    input_rows = session.scalars(
        select(ObservationOperationInputORM)
        .where(ObservationOperationInputORM.frame_id.in_(frame_ids))
        .order_by(
            ObservationOperationInputORM.frame_id,
            ObservationOperationInputORM.operation_index,
            ObservationOperationInputORM.input_index,
        )
    ).all()
    inputs_by_operation = _input_digests_by_operation(input_rows)
    operations_by_frame: dict[str, list[ObservationOperationDTO]] = defaultdict(list)
    for row in operation_rows:
        operations_by_frame[row.frame_id].append(
            _to_operation_dto(
                row,
                inputs_by_operation.get((row.frame_id, row.operation_index), ()),
            )
        )
    chains: dict[str, ObservationOperationChainDTO] = {}
    for frame_id in frame_ids:
        chain_row = chain_rows.get(frame_id)
        if chain_row is None:
            raise VPMValidationError("observation operation chain mismatch")
        chains[frame_id] = _operation_chain_from_rows(
            chain_row,
            tuple(operations_by_frame.get(frame_id, ())),
        )
    return chains


def matrix_blob_for_observation(
    session: Session,
    observation: ObservationDTO,
) -> MatrixBlob | None:
    if observation.matrix_blob_id is None:
        return None
    row = session.get(MatrixBlobORM, observation.matrix_blob_id)
    if row is None:
        raise VPMValidationError("observation matrix blob mismatch")
    return to_matrix_blob(row)


def observation_select(
    *,
    benchmark_seed_digest: str | None,
    split: str | None,
    episode_id: str | None,
    family: str | None,
    event_type: str | None,
    denominator_class: str | None,
    has_pixels: bool | None,
):
    statement = select(ObservationORM)
    predicates = optional_observation_predicates(
        benchmark_seed_digest=benchmark_seed_digest,
        split=split,
        episode_id=episode_id,
        family=family,
        event_type=event_type,
        denominator_class=denominator_class,
        has_pixels=has_pixels,
    )
    if predicates:
        statement = statement.where(*predicates)
    return statement.order_by(
        ObservationORM.split,
        ObservationORM.episode_id,
        ObservationORM.sequence_number,
        ObservationORM.frame_id,
    )


def operation_observation_select():
    return (
        select(ObservationORM)
        .join(
            ObservationOperationORM,
            ObservationORM.frame_id == ObservationOperationORM.frame_id,
        )
        .distinct()
        .order_by(
            ObservationORM.split,
            ObservationORM.episode_id,
            ObservationORM.sequence_number,
            ObservationORM.frame_id,
        )
    )


def optional_observation_predicates(
    *,
    benchmark_seed_digest: str | None = None,
    split: str | None = None,
    episode_id: str | None = None,
    family: str | None = None,
    event_type: str | None = None,
    denominator_class: str | None = None,
    has_pixels: bool | None = None,
) -> tuple[object, ...]:
    predicates: list[object] = []
    if benchmark_seed_digest is not None:
        predicates.append(ObservationORM.benchmark_seed_digest == benchmark_seed_digest)
    if split is not None:
        predicates.append(ObservationORM.split == split)
    if episode_id is not None:
        predicates.append(ObservationORM.episode_id == episode_id)
    if family is not None:
        predicates.append(ObservationORM.family == family)
    if event_type is not None:
        predicates.append(ObservationORM.event_type == event_type)
    if denominator_class is not None:
        predicates.append(ObservationORM.denominator_class == denominator_class)
    if has_pixels is True:
        predicates.append(ObservationORM.matrix_blob_id.is_not(None))
    elif has_pixels is False:
        predicates.append(ObservationORM.matrix_blob_id.is_(None))
    return tuple(predicates)


def preflight_observation_sequence(
    session: Session,
    observation: ObservationDTO,
    seen_sequences: dict[tuple[str, int], str],
) -> None:
    key = (observation.episode_id, observation.sequence_number)
    existing_frame_id = session.scalars(
        select(ObservationORM.frame_id).where(
            ObservationORM.episode_id == observation.episode_id,
            ObservationORM.sequence_number == observation.sequence_number,
        )
    ).first()
    if existing_frame_id is not None and existing_frame_id != observation.frame_id:
        raise_observation_sequence_conflict()
    seen_frame_id = seen_sequences.get(key)
    if seen_frame_id is not None and seen_frame_id != observation.frame_id:
        raise_observation_sequence_conflict()
    seen_sequences[key] = observation.frame_id


def _operation_chain_from_rows(
    row: ObservationOperationChainORM,
    operations: tuple[ObservationOperationDTO, ...],
) -> ObservationOperationChainDTO:
    if row.operation_count != len(operations):
        raise VPMValidationError("observation operation chain mismatch")
    return ObservationOperationChainDTO.from_dict(
        {
            "version": row.version,
            "operations": [operation.to_dict() for operation in operations],
            "final_emitted_digest": row.final_emitted_digest,
            "operation_chain_digest": row.operation_chain_digest,
        }
    )


def _to_operation_dto(
    row: ObservationOperationORM,
    input_digests: Sequence[str | None],
) -> ObservationOperationDTO:
    return ObservationOperationDTO.from_dict(
        {
            "index": row.operation_index,
            "operation": row.operation,
            "operation_version": row.operation_version,
            "input_digests": list(input_digests),
            "parameters": _json_value(
                row.parameters_json,
                "observation operation payload keys mismatch",
            ),
            "parameter_digest": row.parameter_digest,
            "output_digest": row.output_digest,
            "operation_digest": row.operation_digest,
        }
    )


def _input_digests_by_operation(
    rows: Sequence[ObservationOperationInputORM],
) -> dict[tuple[str, int], tuple[str | None, ...]]:
    indexed: dict[tuple[str, int], list[tuple[int, str | None]]] = defaultdict(list)
    for row in rows:
        indexed[(row.frame_id, row.operation_index)].append(
            (row.input_index, row.input_digest)
        )
    grouped: dict[tuple[str, int], tuple[str | None, ...]] = {}
    for key, values in indexed.items():
        sorted_values = sorted(values)
        if [index for index, _input_digest in sorted_values] != list(
            range(len(sorted_values))
        ):
            raise VPMValidationError("observation operation payload keys mismatch")
        grouped[key] = tuple(input_digest for _index, input_digest in sorted_values)
    return grouped


def _json_mapping(text: str, message: str) -> Mapping[str, object]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VPMValidationError(message) from exc
    if not isinstance(value, Mapping):
        raise VPMValidationError(message)
    return cast(Mapping[str, object], value)


def _json_value(text: str, message: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise VPMValidationError(message) from exc


def _canonical_or_none(dto: CanonicalJsonDTO | None) -> str | None:
    return None if dto is None else dto.canonical_text


def _canonical_from_text_or_none(text: str | None) -> CanonicalJsonDTO | None:
    return None if text is None else CanonicalJsonDTO(text)


def _provider_descriptor_json(observation: ObservationDTO) -> str | None:
    descriptor = observation.provider_observation_descriptor
    return None if descriptor is None else canonical_json_text(descriptor.to_dict())


def _provider_descriptor_from_json(
    text: str | None,
) -> ProviderObservationDescriptorDTO | None:
    if text is None:
        return None
    return ProviderObservationDescriptorDTO.from_dict(
        _json_mapping(text, "provider observation descriptor keys mismatch")
    )


__all__ = [
    "chain_for_frame",
    "chains_for_frames",
    "materialized_observations_from_rows",
    "matrix_blob_for_observation",
    "observation_select",
    "observations_from_rows",
    "operation_observation_select",
    "optional_observation_predicates",
    "preflight_observation_sequence",
    "to_matrix_blob",
    "to_matrix_blob_orm",
    "to_observation_dto",
    "to_observation_orm",
    "to_operation_chain_dto",
    "to_operation_chain_orm",
    "to_operation_input_orms",
    "to_operation_orms",
    "validate_observation_row",
]
