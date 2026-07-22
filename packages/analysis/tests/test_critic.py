from __future__ import annotations

import pytest

from zeromodel.analysis.critic import (
    CriticObservation,
    build_critic_vpm,
    observations_from_critic_lines,
)
from zeromodel.core.artifact import VPMValidationError


def test_critic_vpm_places_highest_risk_first_and_warns() -> None:
    assessment = build_critic_vpm(
        [
            CriticObservation(
                item_id="supported-claim",
                critic_score=0.88,
                policy_fit=0.95,
                evidence_support=0.90,
                citation_match=0.92,
                semantic_drift=0.05,
            ),
            CriticObservation(
                item_id="risky-claim",
                critic_score=0.36,
                policy_fit=0.45,
                evidence_support=0.32,
                citation_match=0.40,
                semantic_drift=0.72,
                hallucination_energy=0.78,
                verifiability=0.39,
            ),
        ],
        risk_threshold=0.50,
        hallucination_energy_threshold=0.50,
        verifiability_threshold=0.60,
    )

    assert assessment.highest_risk_item_id == "risky-claim"
    assert assessment.risky_count == 1
    assert "risk_score_above_threshold" in assessment.warnings
    assert "hallucination_energy_above_threshold" in assessment.warnings
    assert "verifiability_below_threshold" in assessment.warnings
    assert assessment.artifact.provenance["highest_risk_item_id"] == "risky-claim"
    assert assessment.artifact.source.metric_ids == (
        "risk_score",
        "hallucination_energy",
        "semantic_drift",
        "critic_risk",
        "policy_gap",
        "evidence_gap",
        "citation_gap",
        "verifiability",
    )

    cell = assessment.artifact.cell(0, 0)
    assert cell.row_id == "risky-claim"
    assert cell.metric_id == "risk_score"


def test_critic_observation_computes_energy_and_verifiability() -> None:
    observation = CriticObservation(
        item_id="claim-a",
        critic_score=0.50,
        policy_fit=0.80,
        evidence_support=0.60,
        citation_match=0.40,
        semantic_drift=0.20,
    )

    assert observation.computed_hallucination_energy == pytest.approx(
        (0.20 + 0.40 + 0.60 + 0.20) / 4.0
    )
    assert observation.computed_verifiability == pytest.approx(
        (0.60 + 0.40 + 0.80) / 3.0
    )
    assert observation.critic_risk == pytest.approx(0.50)


def test_from_writer_style_critic_result_extracts_features() -> None:
    observation = CriticObservation.from_critic_result(
        {
            "index": 2,
            "text": "The model claims the source says something it does not say.",
            "score": 0.41,
            "label": 0,
            "threshold": 0.50,
            "verdict": "risky",
            "explanation": [{"feature": "semantic_drift", "contribution": 0.8}],
            "features": {
                "policy_score": 0.58,
                "support_score": 0.31,
                "citation_score": 0.44,
                "semantic_drift": 0.70,
            },
        },
        item_id="line_2",
    )

    assert observation.item_id == "line_2"
    assert observation.critic_score == pytest.approx(0.41)
    assert observation.policy_fit == pytest.approx(0.58)
    assert observation.evidence_support == pytest.approx(0.31)
    assert observation.citation_match == pytest.approx(0.44)
    assert observation.semantic_drift == pytest.approx(0.70)
    assert observation.metadata["threshold"] == 0.50
    assert observation.metadata["label"] == 0
    assert observation.explanation[0]["feature"] == "semantic_drift"


def test_observations_from_critic_lines_matches_writer_shape() -> None:
    observations = observations_from_critic_lines(
        {
            "ok": True,
            "count": 2,
            "risky_count": 1,
            "items": [
                {
                    "index": 0,
                    "text": "Grounded sentence.",
                    "score": 0.82,
                    "verdict": "good",
                    "features": {"support_score": 0.90},
                },
                {
                    "index": 1,
                    "text": "Ungrounded sentence.",
                    "score": 0.28,
                    "verdict": "bad",
                    "features": {
                        "support_score": 0.20,
                        "citation_score": 0.30,
                        "semantic_drift": 0.76,
                    },
                },
            ],
        }
    )

    assert [observation.item_id for observation in observations] == ["line_0", "line_1"]
    assessment = build_critic_vpm(observations)
    assert assessment.highest_risk_item_id == "line_1"


def test_critic_vpm_rejects_duplicate_items_and_invalid_values() -> None:
    with pytest.raises(VPMValidationError):
        CriticObservation(item_id="bad", critic_score=1.2)

    with pytest.raises(VPMValidationError):
        build_critic_vpm(
            [
                CriticObservation(item_id="same", critic_score=0.9),
                CriticObservation(item_id="same", critic_score=0.8),
            ]
        )
