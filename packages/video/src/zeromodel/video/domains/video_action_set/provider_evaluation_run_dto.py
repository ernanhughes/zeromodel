"""`ProviderEvaluationRunDTO` / `MaterializedProviderEvaluationRunDTO` - the aggregate root."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    PROVIDER_EVALUATION_RUN_VERSION,
)
from zeromodel.video.domains.video_action_set.dto import CanonicalJsonDTO
from zeromodel.video.domains.video_action_set.observation_common import (
    json_mapping,
    mapping,
    require_keys,
    sequence,
    sha256,
    string,
)
from zeromodel.video.domains.video_action_set.provider_evaluation_case_dto import (
    ProviderEvaluationCaseDTO,
)
from zeromodel.video.domains.video_action_set.provider_evaluation_common import (
    nonempty_str,
)
from zeromodel.video.domains.video_action_set.provider_evaluation_configuration_dto import (
    ProviderConfigurationDTO,
)
from zeromodel.video.domains.video_action_set.provider_evaluation_summary_dto import (
    ProviderEvaluationSummaryDTO,
)


RUN_KEYS = (
    "version",
    "run_id",
    "fixture_identity",
    "provider_configuration",
    "policy_artifact_id",
    "case_mode",
    "representation_mode",
    "case_ids",
    "summary",
    "metadata",
)


@dataclass(frozen=True, slots=True)
class ProviderEvaluationRunDTO:
    """Aggregate root for one provider-evaluation run.

    Does not duplicate pixel payloads or transformation provenance: those
    remain owned by ``ObservationDTO``/``MatrixBlob``/
    ``ObservationOperationChainDTO`` and are reached through the case list's
    ``frame_id`` references. Only validates its own shape; full aggregate
    closure against the actual cases lives on
    ``MaterializedProviderEvaluationRunDTO``.
    """

    version: str
    run_id: str
    fixture_identity: str
    provider_configuration: ProviderConfigurationDTO
    policy_artifact_id: str
    case_mode: str
    representation_mode: str
    case_ids: tuple[str, ...]
    summary: ProviderEvaluationSummaryDTO
    metadata: CanonicalJsonDTO

    def __post_init__(self) -> None:
        if self.version != PROVIDER_EVALUATION_RUN_VERSION:
            raise VPMValidationError("unsupported provider evaluation run version")
        nonempty_str(self.fixture_identity, "run fixture identity mismatch")
        sha256(self.policy_artifact_id, "run policy artifact id mismatch")
        nonempty_str(self.case_mode, "run case mode mismatch")
        nonempty_str(self.representation_mode, "run representation mode mismatch")
        if any(
            not isinstance(case_id, str) or not case_id for case_id in self.case_ids
        ):
            raise VPMValidationError("run case ids mismatch")
        if len(set(self.case_ids)) != len(self.case_ids):
            raise VPMValidationError("run case ids contain a duplicate")
        json_mapping(self.metadata, "run metadata mismatch")
        expected_id = canonical_sha256(_run_payload_without_id(self))
        if self.run_id != expected_id:
            raise VPMValidationError("run id mismatch")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProviderEvaluationRunDTO":
        require_keys(payload, RUN_KEYS, "provider evaluation run keys mismatch")
        case_ids_value = sequence(payload["case_ids"], "run case ids mismatch")
        return cls(
            version=string(
                payload, "version", "unsupported provider evaluation run version"
            ),
            run_id=string(payload, "run_id", "run id mismatch"),
            fixture_identity=string(
                payload, "fixture_identity", "run fixture identity mismatch"
            ),
            provider_configuration=ProviderConfigurationDTO.from_dict(
                mapping(
                    payload["provider_configuration"],
                    "run provider configuration mismatch",
                )
            ),
            policy_artifact_id=sha256(
                payload["policy_artifact_id"], "run policy artifact id mismatch"
            ),
            case_mode=string(payload, "case_mode", "run case mode mismatch"),
            representation_mode=string(
                payload, "representation_mode", "run representation mode mismatch"
            ),
            case_ids=tuple(str(case_id) for case_id in case_ids_value),
            summary=ProviderEvaluationSummaryDTO.from_dict(
                mapping(payload["summary"], "run summary mismatch")
            ),
            metadata=CanonicalJsonDTO.from_value(payload["metadata"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "fixture_identity": self.fixture_identity,
            "provider_configuration": self.provider_configuration.to_dict(),
            "policy_artifact_id": self.policy_artifact_id,
            "case_mode": self.case_mode,
            "representation_mode": self.representation_mode,
            "case_ids": list(self.case_ids),
            "summary": self.summary.to_dict(),
            "metadata": self.metadata.to_value(),
        }


def _run_payload_without_id(run: ProviderEvaluationRunDTO) -> dict[str, object]:
    payload = run.to_dict()
    payload.pop("run_id")
    return payload


@dataclass(frozen=True, slots=True)
class MaterializedProviderEvaluationRunDTO:
    """The complete provider-evaluation aggregate: run + ordered cases + summary.

    ``__post_init__`` is where every aggregate-closure invariant lives: case
    ordinals are contiguous and zero-based, every case's identity/policy/
    configuration agrees with the run, and the summary both equals
    ``run.summary`` and independently reconciles with the actual cases. It is
    therefore impossible to combine a valid run, cases from a different run,
    and an unrelated but individually valid summary.
    """

    run: ProviderEvaluationRunDTO
    cases: tuple[ProviderEvaluationCaseDTO, ...]
    summary: ProviderEvaluationSummaryDTO

    def __post_init__(self) -> None:
        if len(self.cases) != len(self.run.case_ids):
            raise VPMValidationError("provider evaluation run case count mismatch")
        for index, case in enumerate(self.cases):
            self._validate_case_membership(index, case)
        if self.summary != self.run.summary:
            raise VPMValidationError("provider evaluation run summary mismatch")
        if self.summary != ProviderEvaluationSummaryDTO.from_cases(self.cases):
            raise VPMValidationError(
                "provider evaluation run summary does not reconcile with cases"
            )

    def _validate_case_membership(
        self, index: int, case: ProviderEvaluationCaseDTO
    ) -> None:
        if case.case_ordinal != index:
            raise VPMValidationError(
                "provider evaluation case ordinals are not contiguous"
            )
        if case.case_id != self.run.case_ids[index]:
            raise VPMValidationError("provider evaluation run case identity mismatch")
        if case.policy_artifact_id != self.run.policy_artifact_id:
            raise VPMValidationError(
                "provider evaluation case policy artifact mismatch"
            )
        if (
            case.provider_configuration_id
            != self.run.provider_configuration.provider_configuration_id
        ):
            raise VPMValidationError(
                "provider evaluation case provider configuration mismatch"
            )


def build_provider_evaluation_run(
    *,
    fixture_identity: str,
    provider_configuration: ProviderConfigurationDTO,
    policy_artifact_id: str,
    case_mode: str,
    representation_mode: str,
    cases: Sequence[ProviderEvaluationCaseDTO],
    metadata: Mapping[str, object] | None = None,
) -> MaterializedProviderEvaluationRunDTO:
    """Build and closure-validate a complete provider-evaluation aggregate."""
    ordered_cases = tuple(sorted(cases, key=lambda case: case.case_ordinal))
    summary = ProviderEvaluationSummaryDTO.from_cases(ordered_cases)
    case_ids = tuple(case.case_id for case in ordered_cases)
    payload = {
        "version": PROVIDER_EVALUATION_RUN_VERSION,
        "fixture_identity": fixture_identity,
        "provider_configuration": provider_configuration.to_dict(),
        "policy_artifact_id": policy_artifact_id,
        "case_mode": case_mode,
        "representation_mode": representation_mode,
        "case_ids": list(case_ids),
        "summary": summary.to_dict(),
        "metadata": dict(metadata or {}),
    }
    run_id = canonical_sha256(payload)
    run = ProviderEvaluationRunDTO.from_dict(payload | {"run_id": run_id})
    return MaterializedProviderEvaluationRunDTO(
        run=run, cases=ordered_cases, summary=summary
    )


__all__ = [
    "MaterializedProviderEvaluationRunDTO",
    "ProviderEvaluationRunDTO",
    "RUN_KEYS",
    "build_provider_evaluation_run",
]
