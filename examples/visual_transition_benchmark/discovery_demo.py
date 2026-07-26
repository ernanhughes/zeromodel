"""Secondary demonstration of P18C (recurrent unexplained-transition discovery).

Not part of the core per-transition metrics (section 6). This shows one extra
capability raw pixel differencing has no analogue for at all: recurrence
*across an episode's cohort of transitions*, rather than judging one transition
in isolation. It reuses the exact P18A/P18B artifacts System C already built
for each transition (``TransitionAnalysis.transition_evidence`` /
``.conformance_report``); it builds no new evidence.

Thresholds are loosened relative to ``TransitionDiscoveryPolicyDTO``'s defaults
because one benchmark episode is only 18 *heterogeneous* category-representative
transitions (not a long homogeneous action stream), so natural recurrence
fractions are low. This is a mechanism demonstration, not a calibrated
production policy.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from zeromodel.perception.transition_discovery import (
    TransitionDiscoveryObservationDTO,
    TransitionDiscoveryPolicyDTO,
    discover_recurrent_unexplained_transitions,
)

from visual_transition_benchmark import zeromodel_adapter as zm
from visual_transition_benchmark.dataset import TransitionRecord
from visual_transition_benchmark.zeromodel_adapter import TransitionAnalysis

DEMO_POLICY = TransitionDiscoveryPolicyDTO.create(
    minimum_observation_count=3,
    minimum_field_occurrence_count=2,
    minimum_field_recurrence_fraction=0.15,
    minimum_signature_occurrence_count=2,
    minimum_signature_recurrence_fraction=0.15,
)


def run_episode_discovery(
    episode_id: str,
    records: Sequence[TransitionRecord],
    outputs: Sequence[TransitionAnalysis],
) -> Mapping[str, object]:
    observations = []
    for record, output in zip(records, outputs):
        if record.episode_id != episode_id:
            continue
        observations.append(
            TransitionDiscoveryObservationDTO.create(
                interaction_id=record.transition_id,
                cohort_id=episode_id,
                transition=output.transition_evidence,
                conformance=output.conformance_report,
            )
        )
    if not observations:
        return {"episode_id": episode_id, "status": "no_observations"}

    report = discover_recurrent_unexplained_transitions(tuple(observations), DEMO_POLICY)
    candidates = []
    for candidate in report.candidates:
        bands = sorted({zm.FIELD_ID_TO_BAND[fid] for fid in candidate.field_ids})
        candidates.append(
            {
                "candidate_kind": candidate.candidate_kind,
                "bands": bands,
                "occurrence_count": candidate.occurrence_count,
                "observation_count": candidate.observation_count,
                "recurrence_fraction": candidate.recurrence_fraction,
                "proposed_expected_change": candidate.proposed_expected_change,
                "dominant_direction": candidate.dominant_direction,
            }
        )
    return {
        "episode_id": episode_id,
        "status": report.status,
        "n_observations": len(observations),
        "candidates": candidates,
    }
