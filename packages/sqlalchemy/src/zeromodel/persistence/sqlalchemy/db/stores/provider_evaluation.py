from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_json_text
from zeromodel.video.domains.video_action_set.contracts import (
    PROVIDER_EVALUATION_CASE_VERSION,
    PROVIDER_EVALUATION_RUN_VERSION,
)
from zeromodel.video.domains.video_action_set.provider_evaluation_dto import (
    MaterializedProviderEvaluationRunDTO,
    ProviderConfigurationDTO,
    ProviderEvaluationCaseDTO,
    ProviderEvaluationRunDTO,
    ProviderEvaluationSummaryDTO,
)
from zeromodel.video.domains.video_action_set.store import (
    raise_provider_evaluation_case_conflict,
    raise_provider_evaluation_configuration_conflict,
    raise_provider_evaluation_policy_mismatch,
    raise_provider_evaluation_run_conflict,
)
from zeromodel.persistence.sqlalchemy.db.orm.provider_evaluation import (
    ProviderEvaluationCaseORM,
    ProviderEvaluationConfigurationORM,
    ProviderEvaluationRunORM,
)
from zeromodel.persistence.sqlalchemy.db.orm.video_action_set import ObservationORM


def to_configuration_orm(
    configuration: ProviderConfigurationDTO,
) -> ProviderEvaluationConfigurationORM:
    return ProviderEvaluationConfigurationORM(
        provider_configuration_id=configuration.provider_configuration_id,
        provider_kind=configuration.provider_kind,
        model_digest=configuration.model_digest,
        payload_json=canonical_json_text(configuration.to_dict()),
    )


def to_configuration_dto(
    row: ProviderEvaluationConfigurationORM,
) -> ProviderConfigurationDTO:
    payload = _json_mapping(row.payload_json, "provider configuration digest mismatch")
    dto = ProviderConfigurationDTO.from_dict(payload)
    if (
        row.provider_configuration_id != dto.provider_configuration_id
        or row.provider_kind != dto.provider_kind
        or row.model_digest != dto.model_digest
    ):
        raise VPMValidationError("provider configuration digest mismatch")
    return dto


def to_run_orm(run: ProviderEvaluationRunDTO) -> ProviderEvaluationRunORM:
    summary = run.summary
    return ProviderEvaluationRunORM(
        run_id=run.run_id,
        provider_configuration_id=run.provider_configuration.provider_configuration_id,
        fixture_identity=run.fixture_identity,
        policy_artifact_id=run.policy_artifact_id,
        case_mode=run.case_mode,
        representation_mode=run.representation_mode,
        case_count=len(run.case_ids),
        attempted_count=summary.attempted_count,
        accepted_count=summary.accepted_count,
        rejected_count=summary.rejected_count,
        exact_count=summary.exact_count,
        action_equivalent_count=summary.action_equivalent_count,
        action_changing_count=summary.action_changing_count,
        action_correct_count=summary.action_correct_count,
        latency_sample_count=summary.latency_sample_count,
        latency_min_us=summary.latency_min_us,
        latency_max_us=summary.latency_max_us,
        latency_total_us=summary.latency_total_us,
        latency_median_us=summary.latency_median_us,
        latency_p95_us=summary.latency_p95_us,
        summary_json=canonical_json_text(summary.to_dict()),
        metadata_json=run.metadata.canonical_text,
    )


def to_run_dto(
    session: Session,
    row: ProviderEvaluationRunORM,
    *,
    case_ids: Sequence[str],
) -> ProviderEvaluationRunDTO:
    configuration_row = session.get(
        ProviderEvaluationConfigurationORM, row.provider_configuration_id
    )
    if configuration_row is None:
        raise VPMValidationError(
            "provider evaluation run provider configuration mismatch"
        )
    configuration = to_configuration_dto(configuration_row)
    summary = ProviderEvaluationSummaryDTO.from_dict(
        _json_mapping(row.summary_json, "provider evaluation run digest mismatch")
    )
    payload = {
        "version": PROVIDER_EVALUATION_RUN_VERSION,
        "run_id": row.run_id,
        "fixture_identity": row.fixture_identity,
        "provider_configuration": configuration.to_dict(),
        "policy_artifact_id": row.policy_artifact_id,
        "case_mode": row.case_mode,
        "representation_mode": row.representation_mode,
        "case_ids": list(case_ids),
        "summary": summary.to_dict(),
        "metadata": _json_value(row.metadata_json, "run metadata mismatch"),
    }
    dto = ProviderEvaluationRunDTO.from_dict(payload)
    if (
        row.run_id != dto.run_id
        or row.fixture_identity != dto.fixture_identity
        or row.policy_artifact_id != dto.policy_artifact_id
        or row.case_mode != dto.case_mode
        or row.representation_mode != dto.representation_mode
        or row.case_count != len(dto.case_ids)
        or row.attempted_count != dto.summary.attempted_count
        or row.accepted_count != dto.summary.accepted_count
        or row.rejected_count != dto.summary.rejected_count
        or row.exact_count != dto.summary.exact_count
        or row.action_equivalent_count != dto.summary.action_equivalent_count
        or row.action_changing_count != dto.summary.action_changing_count
        or row.action_correct_count != dto.summary.action_correct_count
    ):
        raise VPMValidationError("provider evaluation run digest mismatch")
    return dto


def to_case_orm(
    case: ProviderEvaluationCaseDTO, *, run_id: str
) -> ProviderEvaluationCaseORM:
    return ProviderEvaluationCaseORM(
        case_id=case.case_id,
        run_id=run_id,
        case_ordinal=case.case_ordinal,
        frame_id=case.frame_id,
        policy_artifact_id=case.policy_artifact_id,
        provider_configuration_id=case.provider_configuration_id,
        accepted=case.accepted,
        exact_state_match=case.exact_state_match,
        action_match=case.action_match,
        outcome=case.outcome,
        expected_row_id=case.expected_row_id,
        expected_action=case.expected_action,
        predicted_row_id=case.predicted_row_id,
        predicted_action=case.predicted_action,
        rejection_reason=case.rejection_reason,
        provider_confidence=case.provider_confidence,
        provider_latency_us=case.provider_latency_us,
        provider_raw_response_digest=case.provider_raw_response_digest,
        provider_raw_response_text=case.provider_raw_response_text,
        expected_state_json=case.expected_state.canonical_text,
        predicted_state_json=(
            None
            if case.predicted_state is None
            else case.predicted_state.canonical_text
        ),
        expected_decision_trace_json=case.expected_decision_trace.canonical_text,
        predicted_decision_trace_json=(
            None
            if case.predicted_decision_trace is None
            else case.predicted_decision_trace.canonical_text
        ),
        factor_matches_json=case.factor_matches.canonical_text,
        provider_response_metadata_json=case.provider_response_metadata.canonical_text,
        metadata_json=case.metadata.canonical_text,
    )


def to_case_dto(row: ProviderEvaluationCaseORM) -> ProviderEvaluationCaseDTO:
    payload = {
        "version": PROVIDER_EVALUATION_CASE_VERSION,
        "case_ordinal": row.case_ordinal,
        "frame_id": row.frame_id,
        "policy_artifact_id": row.policy_artifact_id,
        "provider_configuration_id": row.provider_configuration_id,
        "expected_state": _json_value(
            row.expected_state_json, "case expected state mismatch"
        ),
        "expected_row_id": row.expected_row_id,
        "expected_action": row.expected_action,
        "expected_decision_trace": _json_value(
            row.expected_decision_trace_json,
            "case expected decision trace mismatch",
        ),
        "accepted": row.accepted,
        "rejection_reason": row.rejection_reason,
        "predicted_state": _optional_json_value(row.predicted_state_json),
        "predicted_row_id": row.predicted_row_id,
        "predicted_action": row.predicted_action,
        "predicted_decision_trace": _optional_json_value(
            row.predicted_decision_trace_json
        ),
        "exact_state_match": row.exact_state_match,
        "action_match": row.action_match,
        "factor_matches": _json_value(
            row.factor_matches_json, "case factor matches mismatch"
        ),
        "outcome": row.outcome,
        "provider_confidence": row.provider_confidence,
        "provider_latency_us": row.provider_latency_us,
        "provider_raw_response_digest": row.provider_raw_response_digest,
        "provider_raw_response_text": row.provider_raw_response_text,
        "provider_response_metadata": _json_value(
            row.provider_response_metadata_json,
            "case provider response metadata mismatch",
        ),
        "metadata": _json_value(row.metadata_json, "case metadata mismatch"),
        "case_id": row.case_id,
    }
    dto = ProviderEvaluationCaseDTO.from_dict(payload)
    if (
        row.case_id != dto.case_id
        or row.case_ordinal != dto.case_ordinal
        or row.frame_id != dto.frame_id
        or row.policy_artifact_id != dto.policy_artifact_id
        or row.provider_configuration_id != dto.provider_configuration_id
        or row.accepted != dto.accepted
        or row.exact_state_match != dto.exact_state_match
        or row.action_match != dto.action_match
        or row.outcome != dto.outcome
    ):
        raise VPMValidationError("provider evaluation case digest mismatch")
    return dto


def case_ids_for_run(session: Session, run_id: str) -> tuple[str, ...]:
    rows = session.scalars(
        select(ProviderEvaluationCaseORM.case_id)
        .where(ProviderEvaluationCaseORM.run_id == run_id)
        .order_by(ProviderEvaluationCaseORM.case_ordinal)
    ).all()
    return tuple(rows)


def cases_for_run(
    session: Session, run_id: str
) -> tuple[ProviderEvaluationCaseDTO, ...]:
    rows = session.scalars(
        select(ProviderEvaluationCaseORM)
        .where(ProviderEvaluationCaseORM.run_id == run_id)
        .order_by(ProviderEvaluationCaseORM.case_ordinal)
    ).all()
    return tuple(to_case_dto(row) for row in rows)


def run_select(
    *,
    fixture_identity: str | None,
    provider_kind: str | None,
    model_digest: str | None,
    policy_artifact_id: str | None,
    case_mode: str | None,
    representation_mode: str | None,
):
    statement = select(ProviderEvaluationRunORM)
    predicates = cast(
        tuple[Any, ...],
        optional_run_predicates(
            fixture_identity=fixture_identity,
            provider_kind=provider_kind,
            model_digest=model_digest,
            policy_artifact_id=policy_artifact_id,
            case_mode=case_mode,
            representation_mode=representation_mode,
        ),
    )
    if provider_kind is not None or model_digest is not None:
        statement = statement.join(
            ProviderEvaluationConfigurationORM,
            ProviderEvaluationRunORM.provider_configuration_id
            == ProviderEvaluationConfigurationORM.provider_configuration_id,
        )
    if predicates:
        statement = statement.where(*predicates)
    return statement.order_by(
        ProviderEvaluationRunORM.fixture_identity, ProviderEvaluationRunORM.run_id
    )


def optional_run_predicates(
    *,
    fixture_identity: str | None = None,
    provider_kind: str | None = None,
    model_digest: str | None = None,
    policy_artifact_id: str | None = None,
    case_mode: str | None = None,
    representation_mode: str | None = None,
) -> tuple[object, ...]:
    predicates: list[object] = []
    if fixture_identity is not None:
        predicates.append(ProviderEvaluationRunORM.fixture_identity == fixture_identity)
    if provider_kind is not None:
        predicates.append(
            ProviderEvaluationConfigurationORM.provider_kind == provider_kind
        )
    if model_digest is not None:
        predicates.append(
            ProviderEvaluationConfigurationORM.model_digest == model_digest
        )
    if policy_artifact_id is not None:
        predicates.append(
            ProviderEvaluationRunORM.policy_artifact_id == policy_artifact_id
        )
    if case_mode is not None:
        predicates.append(ProviderEvaluationRunORM.case_mode == case_mode)
    if representation_mode is not None:
        predicates.append(
            ProviderEvaluationRunORM.representation_mode == representation_mode
        )
    return tuple(predicates)


def case_select(
    *,
    run_id: str | None,
    outcome: str | None,
    accepted: bool | None,
    exact_state_match: bool | None,
    action_match: bool | None,
    frame_id: str | None,
):
    statement = select(ProviderEvaluationCaseORM)
    predicates = cast(
        tuple[Any, ...],
        optional_case_predicates(
            run_id=run_id,
            outcome=outcome,
            accepted=accepted,
            exact_state_match=exact_state_match,
            action_match=action_match,
            frame_id=frame_id,
        ),
    )
    if predicates:
        statement = statement.where(*predicates)
    return statement.order_by(
        ProviderEvaluationCaseORM.case_ordinal,
        ProviderEvaluationCaseORM.frame_id,
        ProviderEvaluationCaseORM.case_id,
    )


def optional_case_predicates(
    *,
    run_id: str | None = None,
    outcome: str | None = None,
    accepted: bool | None = None,
    exact_state_match: bool | None = None,
    action_match: bool | None = None,
    frame_id: str | None = None,
) -> tuple[object, ...]:
    predicates: list[object] = []
    if run_id is not None:
        predicates.append(ProviderEvaluationCaseORM.run_id == run_id)
    if outcome is not None:
        predicates.append(ProviderEvaluationCaseORM.outcome == outcome)
    if accepted is not None:
        predicates.append(ProviderEvaluationCaseORM.accepted == accepted)
    if exact_state_match is not None:
        predicates.append(
            ProviderEvaluationCaseORM.exact_state_match == exact_state_match
        )
    if action_match is not None:
        predicates.append(ProviderEvaluationCaseORM.action_match == action_match)
    if frame_id is not None:
        predicates.append(ProviderEvaluationCaseORM.frame_id == frame_id)
    return tuple(predicates)


def preflight_case_observations(
    session: Session, cases: Sequence[ProviderEvaluationCaseDTO]
) -> None:
    frame_ids = {case.frame_id for case in cases}
    if not frame_ids:
        return
    known = set(
        session.scalars(
            select(ObservationORM.frame_id).where(
                ObservationORM.frame_id.in_(frame_ids)
            )
        ).all()
    )
    missing = frame_ids - known
    if missing:
        raise VPMValidationError(
            "provider evaluation case references unknown observation frame"
        )


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


def _optional_json_value(text: str | None) -> object | None:
    if text is None:
        return None
    return json.loads(text)


class ProviderEvaluationSqlStoreMixin:
    """SQL Store methods for the provider-evaluation aggregate.

    Mixed into `SqlAlchemyVideoActionSetStore` (see `video_action_set.py`);
    kept in its own module, alongside its own DTO<->ORM mapping helpers
    above, so that legacy file already tracked against a fixed line-count
    ceiling does not have to grow to host it.
    """

    _session_factory: sessionmaker[Session]

    def save_provider_evaluation_run(
        self,
        run: MaterializedProviderEvaluationRunDTO,
    ) -> MaterializedProviderEvaluationRunDTO:
        session = self._session_factory()
        try:
            with session.begin():
                return self._save_provider_evaluation_run_in_session(session, run)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_provider_evaluation_run(
        self,
        run_id: str,
    ) -> ProviderEvaluationRunDTO | None:
        session = self._session_factory()
        try:
            with session.begin():
                row = session.get(ProviderEvaluationRunORM, run_id)
                if row is None:
                    return None
                case_ids = case_ids_for_run(session, run_id)
                return to_run_dto(session, row, case_ids=case_ids)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_materialized_provider_evaluation_run(
        self,
        run_id: str,
    ) -> MaterializedProviderEvaluationRunDTO | None:
        session = self._session_factory()
        try:
            with session.begin():
                row = session.get(ProviderEvaluationRunORM, run_id)
                if row is None:
                    return None
                case_ids = case_ids_for_run(session, run_id)
                run_dto = to_run_dto(session, row, case_ids=case_ids)
                cases = cases_for_run(session, run_id)
                return MaterializedProviderEvaluationRunDTO(
                    run=run_dto, cases=cases, summary=run_dto.summary
                )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_provider_evaluation_runs(
        self,
        *,
        fixture_identity: str | None = None,
        provider_kind: str | None = None,
        model_digest: str | None = None,
        policy_artifact_id: str | None = None,
        case_mode: str | None = None,
        representation_mode: str | None = None,
    ) -> tuple[ProviderEvaluationRunDTO, ...]:
        session = self._session_factory()
        try:
            with session.begin():
                rows = session.scalars(
                    run_select(
                        fixture_identity=fixture_identity,
                        provider_kind=provider_kind,
                        model_digest=model_digest,
                        policy_artifact_id=policy_artifact_id,
                        case_mode=case_mode,
                        representation_mode=representation_mode,
                    )
                ).all()
                return tuple(
                    to_run_dto(
                        session, row, case_ids=case_ids_for_run(session, row.run_id)
                    )
                    for row in rows
                )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_provider_evaluation_cases(
        self,
        *,
        run_id: str | None = None,
        outcome: str | None = None,
        accepted: bool | None = None,
        exact_state_match: bool | None = None,
        action_match: bool | None = None,
        frame_id: str | None = None,
    ) -> tuple[ProviderEvaluationCaseDTO, ...]:
        session = self._session_factory()
        try:
            with session.begin():
                rows = session.scalars(
                    case_select(
                        run_id=run_id,
                        outcome=outcome,
                        accepted=accepted,
                        exact_state_match=exact_state_match,
                        action_match=action_match,
                        frame_id=frame_id,
                    )
                ).all()
                return tuple(to_case_dto(row) for row in rows)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _save_provider_evaluation_run_in_session(
        self,
        session: Session,
        run: MaterializedProviderEvaluationRunDTO,
    ) -> MaterializedProviderEvaluationRunDTO:
        existing_run_row = session.get(ProviderEvaluationRunORM, run.run.run_id)
        if existing_run_row is not None:
            existing_case_ids = case_ids_for_run(session, run.run.run_id)
            existing_run_dto = to_run_dto(
                session, existing_run_row, case_ids=existing_case_ids
            )
            if existing_run_dto != run.run:
                raise_provider_evaluation_run_conflict()
            existing_cases = cases_for_run(session, run.run.run_id)
            return MaterializedProviderEvaluationRunDTO(
                run=existing_run_dto,
                cases=existing_cases,
                summary=existing_run_dto.summary,
            )

        preflight_case_observations(session, run.cases)

        configuration_id = run.run.provider_configuration.provider_configuration_id
        existing_configuration_row = session.get(
            ProviderEvaluationConfigurationORM, configuration_id
        )
        if existing_configuration_row is not None:
            existing_configuration_dto = to_configuration_dto(
                existing_configuration_row
            )
            if existing_configuration_dto != run.run.provider_configuration:
                raise_provider_evaluation_configuration_conflict()
        else:
            session.add(to_configuration_orm(run.run.provider_configuration))
            session.flush()

        for case in run.cases:
            if case.policy_artifact_id != run.run.policy_artifact_id:
                raise_provider_evaluation_policy_mismatch()
            # A case belongs to exactly one run; since case_ordinal participates
            # in case_id's digest, a pre-existing row here can only mean this
            # case_id already belongs to a different, already-saved run.
            if session.get(ProviderEvaluationCaseORM, case.case_id) is not None:
                raise_provider_evaluation_case_conflict()

        session.add(to_run_orm(run.run))
        session.flush()
        session.add_all(to_case_orm(case, run_id=run.run.run_id) for case in run.cases)
        session.flush()
        return run


__all__ = [
    "ProviderEvaluationSqlStoreMixin",
    "case_ids_for_run",
    "case_select",
    "cases_for_run",
    "optional_case_predicates",
    "optional_run_predicates",
    "preflight_case_observations",
    "run_select",
    "to_case_dto",
    "to_case_orm",
    "to_configuration_dto",
    "to_configuration_orm",
    "to_run_dto",
    "to_run_orm",
]
