from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from zeromodel.perception.transition_analysis import (
    TransitionActionDeclarationDTO,
    TransitionExpectationSetDTO,
    VisualTransitionAnalysisDTO,
    VisualTransitionReaderTraceDTO,
)
from zeromodel.perception.transition_conformance import (
    TransitionExpectationDTO,
    evaluate_transition_conformance,
)
from zeromodel.perception.transition_evidence import TransitionEvidenceVPMDTO
from zeromodel.video.arcade_policy import ShooterConfig, next_rows, parse_state_row_id

from visual_transition_benchmark import zeromodel_adapter as component_zm
from visual_transition_benchmark.adjudication._json import digest
from visual_transition_benchmark.value_contracts import (
    build_value_transition_evidence,
    decode_values,
    evaluate_contracts,
)

EvidenceMode = Literal["component", "value"]


@dataclass(frozen=True)
class CandidateTransitionContract:
    candidate_id: str
    row_id: str
    action: str
    evidence_mode: EvidenceMode
    expectation_signature: str
    expected_components: tuple[str, ...]
    expected_tank_delta: int | None
    expected_cooldown_level: str | None
    observable_fields: tuple[str, ...]
    source: str
    expectations: tuple[TransitionExpectationDTO, ...] = ()

    def to_runtime_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "row_id": self.row_id,
            "action": self.action,
            "evidence_mode": self.evidence_mode,
            "expectation_signature": self.expectation_signature,
            "expected_components": list(self.expected_components),
            "expected_tank_delta": self.expected_tank_delta,
            "expected_cooldown_level": self.expected_cooldown_level,
            "observable_fields": list(self.observable_fields),
            "source": self.source,
            "expectation_ids": [item.expectation_id for item in self.expectations],
        }


def build_candidate_contract(
    row_id: str,
    action: str,
    *,
    evidence_mode: EvidenceMode,
    config: ShooterConfig = ShooterConfig(),
) -> CandidateTransitionContract:
    next_row = next_rows(*parse_state_row_id(row_id), action, width=config.width)[0]
    tank_x, target_x, cooldown = parse_state_row_id(row_id)
    next_tank, next_target, next_cooldown = parse_state_row_id(next_row)
    expected_components = []
    if next_tank != tank_x:
        expected_components.append("tank")
    if next_target != target_x:
        expected_components.append("alien")
    if next_cooldown != cooldown:
        expected_components.append("cooldown")
    expected_components_tuple = tuple(sorted(expected_components))

    expected_delta = None
    expected_cooldown = None
    observable = component_zm.COMPONENT_NAMES
    if evidence_mode == "value":
        expected_delta = next_tank - tank_x
        expected_cooldown = "blocked" if next_cooldown else "ready"
        observable = (
            "tank",
            "alien",
            "cooldown",
            "background",
            "tank_delta",
            "cooldown_level",
        )

    payload = {
        "row_id": row_id,
        "action": action,
        "evidence_mode": evidence_mode,
        "next_row_projection": next_row,
        "expected_components": expected_components_tuple,
        "expected_tank_delta": expected_delta,
        "expected_cooldown_level": expected_cooldown,
        "observable_fields": observable,
        "source": "predeclared_tiny_arcade_transition_rules",
    }
    expectations = _component_expectations(expected_components_tuple)
    return CandidateTransitionContract(
        candidate_id=digest(payload),
        row_id=row_id,
        action=action,
        evidence_mode=evidence_mode,
        expectation_signature=digest(
            {
                k: payload[k]
                for k in sorted(payload)
                if k not in {"row_id", "next_row_projection"}
            }
        ),
        expected_components=expected_components_tuple,
        expected_tank_delta=expected_delta,
        expected_cooldown_level=expected_cooldown,
        observable_fields=tuple(observable),
        source="predeclared_tiny_arcade_transition_rules",
        expectations=expectations,
    )


def _component_expectations(
    expected_components: tuple[str, ...],
) -> tuple[TransitionExpectationDTO, ...]:
    expectations = []
    for name in component_zm.COMPONENT_NAMES:
        if name in expected_components:
            expectations.append(
                TransitionExpectationDTO.create(
                    field_schema_id=component_zm.FIELD_SCHEMA.field_schema_id,
                    annotation_ids=(component_zm.ANNOTATIONS[name].annotation_id,),
                    expected_change="change",
                    minimum_mean_absolute_change=component_zm.FIELD_MIN_MEAN_ABS,
                    minimum_changed_fraction=component_zm.FIELD_MIN_CHANGED_FRACTION,
                )
            )
        else:
            expectations.append(
                TransitionExpectationDTO.create(
                    field_schema_id=component_zm.FIELD_SCHEMA.field_schema_id,
                    annotation_ids=(component_zm.ANNOTATIONS[name].annotation_id,),
                    expected_change="stable",
                    maximum_mean_absolute_change=component_zm.STABLE_MAX_MEAN_ABS,
                    maximum_changed_fraction=component_zm.STABLE_MAX_FRACTION,
                )
            )
    return tuple(sorted(expectations, key=lambda item: item.expectation_id))


def observed_component_signature(
    frame_before: np.ndarray, frame_after: np.ndarray
) -> tuple[str, ...]:
    analysis = component_zm.ArcadeBandZeroModelAnalyzer().analyze(
        frame_before,
        frame_after,
        "STAY",
        component_zm.TransitionMetadata(
            transition_id="adjudication-observed", step_number=0
        ),
    )
    return tuple(sorted(analysis.predicted_components))


def observed_value_signature(
    frame_before: np.ndarray, frame_after: np.ndarray
) -> Mapping[str, object]:
    evidence = build_value_transition_evidence(frame_before, frame_after)
    values = decode_values(evidence)
    return {
        "components": observed_component_signature(frame_before, frame_after),
        "tank_delta": values.tank.delta_x,
        "cooldown_level": values.cooldown.after_level,
        "alien_after_x": values.alien.after_x,
        "alien_after_alive": values.alien.after_alive,
    }


def component_analysis_for_contract(
    *,
    contract: CandidateTransitionContract,
    transition_evidence: TransitionEvidenceVPMDTO,
    action: TransitionActionDeclarationDTO,
    reader_trace: VisualTransitionReaderTraceDTO,
) -> VisualTransitionAnalysisDTO:
    expectation_set = TransitionExpectationSetDTO.create(contract.expectations)
    report = evaluate_transition_conformance(
        transition_evidence,
        contract.expectations,
        component_zm.ANNOTATIONS_TUPLE,
        relations=(),
        minimum_unexplained_mean_absolute_change=component_zm.UNEXPLAINED_MIN_MEAN_ABS,
        minimum_unexplained_changed_fraction=component_zm.UNEXPLAINED_MIN_FRACTION,
    )
    return VisualTransitionAnalysisDTO.create(
        transition=transition_evidence,
        action=action,
        expectation_set=expectation_set,
        conformance_report=report,
        before_reader_trace=reader_trace,
    )


def contract_matches_observation(
    contract: CandidateTransitionContract,
    frame_before: np.ndarray,
    frame_after: np.ndarray,
    *,
    transition_evidence: TransitionEvidenceVPMDTO | None = None,
    action: TransitionActionDeclarationDTO | None = None,
    reader_trace: VisualTransitionReaderTraceDTO | None = None,
) -> tuple[bool, tuple[str, ...], str, str, str | None]:
    expected_signature = contract.expectation_signature
    if contract.evidence_mode == "component":
        if transition_evidence is not None and action is not None and reader_trace is not None:
            analysis = component_analysis_for_contract(
                contract=contract,
                transition_evidence=transition_evidence,
                action=action,
                reader_trace=reader_trace,
            )
            observed_signature = digest(
                {
                    "mode": "component",
                    "transition_evidence_id": transition_evidence.transition_evidence_id,
                }
            )
            ok = analysis.status == "conformant"
            reasons = () if ok else ("component_contract_mismatch",)
            return ok, reasons, expected_signature, observed_signature, analysis.analysis_id
        observed = observed_component_signature(frame_before, frame_after)
        observed_signature = digest({"mode": "component", "observed": observed})
        if tuple(sorted(contract.expected_components)) == observed:
            return True, (), expected_signature, observed_signature, None
        return (
            False,
            ("component_contract_mismatch",),
            expected_signature,
            observed_signature,
            None,
        )

    observed = observed_value_signature(frame_before, frame_after)
    value_evidence = build_value_transition_evidence(frame_before, frame_after)
    values = decode_values(value_evidence)
    verdict = evaluate_contracts(contract.action, values)
    reasons = []
    if tuple(sorted(contract.expected_components)) != tuple(observed["components"]):
        reasons.append("component_contract_mismatch")
    if contract.expected_tank_delta != observed["tank_delta"]:
        reasons.append("tank_value_mismatch")
    if contract.expected_cooldown_level != observed["cooldown_level"]:
        reasons.append("cooldown_value_mismatch")
    if verdict.relation_violations:
        reasons.extend(verdict.relation_violations)
    return (
        not reasons,
        tuple(sorted(reasons)),
        expected_signature,
        digest({"mode": "value", "observed": observed}),
        None,
    )
