from __future__ import annotations

from dataclasses import dataclass

from zeromodel.video.domains.video_action_set.provider_evaluation_dto import (
    MaterializedProviderEvaluationRunDTO,
    ProviderEvaluationCaseDTO,
    ProviderEvaluationRunDTO,
)
from zeromodel.video.domains.video_action_set.store import VideoActionSetStore


@dataclass(frozen=True, slots=True)
class ProviderEvaluationService:
    store: VideoActionSetStore

    def save_run(
        self,
        run: MaterializedProviderEvaluationRunDTO,
    ) -> MaterializedProviderEvaluationRunDTO:
        return self.store.save_provider_evaluation_run(run)

    def get_run(self, run_id: str) -> ProviderEvaluationRunDTO | None:
        return self.store.get_provider_evaluation_run(run_id)

    def get_materialized_run(
        self,
        run_id: str,
    ) -> MaterializedProviderEvaluationRunDTO | None:
        return self.store.get_materialized_provider_evaluation_run(run_id)

    def list_runs(
        self,
        *,
        fixture_identity: str | None = None,
        provider_kind: str | None = None,
        model_digest: str | None = None,
        policy_artifact_id: str | None = None,
        case_mode: str | None = None,
        representation_mode: str | None = None,
    ) -> tuple[ProviderEvaluationRunDTO, ...]:
        return self.store.list_provider_evaluation_runs(
            fixture_identity=fixture_identity,
            provider_kind=provider_kind,
            model_digest=model_digest,
            policy_artifact_id=policy_artifact_id,
            case_mode=case_mode,
            representation_mode=representation_mode,
        )

    def list_cases(
        self,
        *,
        run_id: str | None = None,
        outcome: str | None = None,
        accepted: bool | None = None,
        exact_state_match: bool | None = None,
        action_match: bool | None = None,
        frame_id: str | None = None,
    ) -> tuple[ProviderEvaluationCaseDTO, ...]:
        return self.store.list_provider_evaluation_cases(
            run_id=run_id,
            outcome=outcome,
            accepted=accepted,
            exact_state_match=exact_state_match,
            action_match=action_match,
            frame_id=frame_id,
        )


__all__ = ["ProviderEvaluationService"]
