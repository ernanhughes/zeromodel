"""In-memory Store methods for the provider-evaluation aggregate.

Mixed into `InMemoryVideoActionSetStore` (see `video_action_set_memory.py`);
kept in its own module so that legacy file already tracked against a fixed
line-count ceiling does not have to grow to host it.
"""

from __future__ import annotations

from collections.abc import Iterable

from zeromodel.video.domains.video_action_set.observation_dto import ObservationDTO
from zeromodel.video.domains.video_action_set.provider_evaluation_dto import (
    MaterializedProviderEvaluationRunDTO,
    ProviderConfigurationDTO,
    ProviderEvaluationCaseDTO,
    ProviderEvaluationRunDTO,
)
from zeromodel.video.domains.video_action_set.store import (
    raise_provider_evaluation_case_conflict,
    raise_provider_evaluation_configuration_conflict,
    raise_provider_evaluation_policy_mismatch,
    raise_provider_evaluation_run_conflict,
    raise_unknown_observation_for_provider_evaluation,
)


class _ProviderEvaluationMemoryStoreMixin:
    _provider_evaluation_configurations: dict[str, ProviderConfigurationDTO]
    _provider_evaluation_runs: dict[str, ProviderEvaluationRunDTO]
    _provider_evaluation_cases: dict[str, ProviderEvaluationCaseDTO]
    _provider_evaluation_run_case_ids: dict[str, tuple[str, ...]]
    _observations: dict[str, ObservationDTO]

    def save_provider_evaluation_run(
        self,
        run: MaterializedProviderEvaluationRunDTO,
    ) -> MaterializedProviderEvaluationRunDTO:
        existing_configuration = self._provider_evaluation_configurations.get(
            run.run.provider_configuration.provider_configuration_id
        )
        if (
            existing_configuration is not None
            and existing_configuration != run.run.provider_configuration
        ):
            raise_provider_evaluation_configuration_conflict()

        existing_run = self._provider_evaluation_runs.get(run.run.run_id)
        if existing_run is not None:
            if existing_run != run.run:
                raise_provider_evaluation_run_conflict()
            existing_case_ids = self._provider_evaluation_run_case_ids.get(
                run.run.run_id, ()
            )
            existing_cases = tuple(
                self._provider_evaluation_cases[case_id]
                for case_id in existing_case_ids
            )
            return MaterializedProviderEvaluationRunDTO(
                run=existing_run, cases=existing_cases, summary=existing_run.summary
            )

        for case in run.cases:
            if case.frame_id not in self._observations:
                raise_unknown_observation_for_provider_evaluation()
            if case.policy_artifact_id != run.run.policy_artifact_id:
                raise_provider_evaluation_policy_mismatch()
            # A case belongs to exactly one run (unlike matrix blobs or provider
            # configurations, which are legitimately content-addressed and shared).
            # Since case_ordinal is part of case_id's digest, a collision here can
            # only mean this case_id already belongs to a different, already-saved
            # run - always a conflict, never a dedup opportunity.
            if case.case_id in self._provider_evaluation_cases:
                raise_provider_evaluation_case_conflict()

        self._provider_evaluation_configurations.setdefault(
            run.run.provider_configuration.provider_configuration_id,
            run.run.provider_configuration,
        )
        for case in run.cases:
            self._provider_evaluation_cases.setdefault(case.case_id, case)
        self._provider_evaluation_runs[run.run.run_id] = run.run
        self._provider_evaluation_run_case_ids[run.run.run_id] = tuple(
            case.case_id for case in run.cases
        )
        return run

    def get_provider_evaluation_run(
        self,
        run_id: str,
    ) -> ProviderEvaluationRunDTO | None:
        return self._provider_evaluation_runs.get(run_id)

    def get_materialized_provider_evaluation_run(
        self,
        run_id: str,
    ) -> MaterializedProviderEvaluationRunDTO | None:
        run = self._provider_evaluation_runs.get(run_id)
        if run is None:
            return None
        case_ids = self._provider_evaluation_run_case_ids.get(run_id, ())
        cases = tuple(self._provider_evaluation_cases[case_id] for case_id in case_ids)
        return MaterializedProviderEvaluationRunDTO(
            run=run, cases=cases, summary=run.summary
        )

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
        return tuple(
            sorted(
                (
                    run
                    for run in self._provider_evaluation_runs.values()
                    if _matches_provider_evaluation_run(
                        run,
                        fixture_identity=fixture_identity,
                        provider_kind=provider_kind,
                        model_digest=model_digest,
                        policy_artifact_id=policy_artifact_id,
                        case_mode=case_mode,
                        representation_mode=representation_mode,
                    )
                ),
                key=lambda run: (run.fixture_identity, run.run_id),
            )
        )

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
        candidates: Iterable[ProviderEvaluationCaseDTO]
        if run_id is not None:
            candidate_ids = self._provider_evaluation_run_case_ids.get(run_id, ())
            candidates = (
                self._provider_evaluation_cases[case_id] for case_id in candidate_ids
            )
        else:
            candidates = self._provider_evaluation_cases.values()
        return tuple(
            sorted(
                (
                    case
                    for case in candidates
                    if _matches_provider_evaluation_case(
                        case,
                        outcome=outcome,
                        accepted=accepted,
                        exact_state_match=exact_state_match,
                        action_match=action_match,
                        frame_id=frame_id,
                    )
                ),
                key=lambda case: (case.case_ordinal, case.frame_id, case.case_id),
            )
        )


def _matches_provider_evaluation_run(
    run: ProviderEvaluationRunDTO,
    *,
    fixture_identity: str | None,
    provider_kind: str | None,
    model_digest: str | None,
    policy_artifact_id: str | None,
    case_mode: str | None,
    representation_mode: str | None,
) -> bool:
    if fixture_identity is not None and run.fixture_identity != fixture_identity:
        return False
    if (
        provider_kind is not None
        and run.provider_configuration.provider_kind != provider_kind
    ):
        return False
    if (
        model_digest is not None
        and run.provider_configuration.model_digest != model_digest
    ):
        return False
    if policy_artifact_id is not None and run.policy_artifact_id != policy_artifact_id:
        return False
    if case_mode is not None and run.case_mode != case_mode:
        return False
    if (
        representation_mode is not None
        and run.representation_mode != representation_mode
    ):
        return False
    return True


def _matches_provider_evaluation_case(
    case: ProviderEvaluationCaseDTO,
    *,
    outcome: str | None,
    accepted: bool | None,
    exact_state_match: bool | None,
    action_match: bool | None,
    frame_id: str | None,
) -> bool:
    if outcome is not None and case.outcome != outcome:
        return False
    if accepted is not None and case.accepted != accepted:
        return False
    if exact_state_match is not None and case.exact_state_match != exact_state_match:
        return False
    if action_match is not None and case.action_match != action_match:
        return False
    if frame_id is not None and case.frame_id != frame_id:
        return False
    return True


__all__ = ["_ProviderEvaluationMemoryStoreMixin"]
