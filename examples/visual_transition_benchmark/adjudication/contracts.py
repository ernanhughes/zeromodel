from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

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
    return CandidateTransitionContract(
        candidate_id=digest(payload),
        row_id=row_id,
        action=action,
        evidence_mode=evidence_mode,
        expectation_signature=digest(
            {k: payload[k] for k in sorted(payload) if k != "row_id"}
        ),
        expected_components=expected_components_tuple,
        expected_tank_delta=expected_delta,
        expected_cooldown_level=expected_cooldown,
        observable_fields=tuple(observable),
        source="predeclared_tiny_arcade_transition_rules",
    )


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


def contract_matches_observation(
    contract: CandidateTransitionContract,
    frame_before: np.ndarray,
    frame_after: np.ndarray,
) -> tuple[bool, tuple[str, ...], str]:
    if contract.evidence_mode == "component":
        observed = observed_component_signature(frame_before, frame_after)
        if tuple(sorted(contract.expected_components)) == observed:
            return True, (), digest({"mode": "component", "observed": observed})
        return (
            False,
            ("component_contract_mismatch",),
            digest({"mode": "component", "observed": observed}),
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
        digest({"mode": "value", "observed": observed}),
    )
