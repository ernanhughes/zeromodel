from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CriticCounterexampleDTO:
    artifact_id: str
    ground_truth_label: float
    current_score: float
    candidate_score: float
    score_delta: float
    current_verdict: str | None
    candidate_verdict: str | None
    correct_critic: str | None


def find_counterexamples(
    artifact_ids: list[str],
    labels: list[float],
    current_scores: list[float],
    candidate_scores: list[float],
    *,
    threshold: float = 0.5,
    min_delta: float = 0.0,
) -> tuple[CriticCounterexampleDTO, ...]:
    rows = []
    for artifact_id, label, current, candidate in zip(
        artifact_ids, labels, current_scores, candidate_scores
    ):
        current_verdict = "ACCEPT" if current >= threshold else "REJECT"
        candidate_verdict = "ACCEPT" if candidate >= threshold else "REJECT"
        current_ok = (current >= threshold) == (label == 1.0)
        candidate_ok = (candidate >= threshold) == (label == 1.0)
        if (
            current_verdict != candidate_verdict
            or abs(candidate - current) >= min_delta
        ):
            correct = None
            if current_ok != candidate_ok:
                correct = "current" if current_ok else "candidate"
            rows.append(
                CriticCounterexampleDTO(
                    artifact_id=str(artifact_id),
                    ground_truth_label=float(label),
                    current_score=float(current),
                    candidate_score=float(candidate),
                    score_delta=float(candidate - current),
                    current_verdict=current_verdict,
                    candidate_verdict=candidate_verdict,
                    correct_critic=correct,
                )
            )
    return tuple(rows)
