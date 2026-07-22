"""Critic evidence artifacts for policy-bounded review traces.

This module is inspired by Writer's critic domain shape: upstream code extracts
features, a critic scores each item, and the result includes a score, verdict,
threshold, explanation, and input metadata. ZeroModel does not train or run that
critic. It turns critic/evidence outputs into deterministic VPM artifacts that
surface which items deserve inspection first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from zeromodel.core.artifact import (
    LayoutRecipe,
    ScoreTable,
    VPMArtifact,
    VPMValidationError,
    build_vpm,
)

CRITIC_METRICS: Tuple[str, ...] = (
    "risk_score",
    "hallucination_energy",
    "semantic_drift",
    "critic_risk",
    "policy_gap",
    "evidence_gap",
    "citation_gap",
    "verifiability",
)

_NUMERIC_FEATURE_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "policy_fit": ("policy_fit", "policy_score", "policy_alignment"),
    "evidence_support": (
        "evidence_support",
        "support_score",
        "source_support",
        "grounding_score",
    ),
    "citation_match": ("citation_match", "citation_score", "citation_support"),
    "semantic_drift": ("semantic_drift", "drift", "semantic_distance"),
    "hallucination_energy": ("hallucination_energy", "hallucination_risk", "he"),
    "verifiability": ("verifiability", "verifiability_score", "evidence_verifiability"),
}


def _bounded01(value: float, *, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise VPMValidationError("%s must be finite" % label)
    if number < 0.0 or number > 1.0:
        raise VPMValidationError("%s must be in [0, 1]" % label)
    return number


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _first_numeric(mapping: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key not in mapping:
            continue
        try:
            number = float(mapping[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return None


def _merge_features(result: Mapping[str, Any]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    raw_features = result.get("features")
    if isinstance(raw_features, Mapping):
        features.update(raw_features)
    for key, aliases in _NUMERIC_FEATURE_ALIASES.items():
        if key in result and key not in features:
            features[key] = result[key]
        for alias in aliases:
            if alias in result and alias not in features:
                features[alias] = result[alias]
    return features


@dataclass(frozen=True)
class CriticObservation:
    """One scored critic/evidence item.

    ``critic_score`` follows Writer's critic convention: higher means the item is
    better or safer. The VPM inverts it into ``critic_risk`` so the default layout
    can place higher-risk rows first.
    """

    item_id: str
    critic_score: float
    policy_fit: float = 1.0
    evidence_support: float = 1.0
    citation_match: float = 1.0
    semantic_drift: float = 0.0
    hallucination_energy: Optional[float] = None
    verifiability: Optional[float] = None
    verdict: Optional[str] = None
    explanation: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        item_id = str(self.item_id).strip()
        if not item_id:
            raise VPMValidationError("CriticObservation item_id must be non-empty")
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(
            self, "critic_score", _bounded01(self.critic_score, label="critic_score")
        )
        object.__setattr__(
            self, "policy_fit", _bounded01(self.policy_fit, label="policy_fit")
        )
        object.__setattr__(
            self,
            "evidence_support",
            _bounded01(self.evidence_support, label="evidence_support"),
        )
        object.__setattr__(
            self,
            "citation_match",
            _bounded01(self.citation_match, label="citation_match"),
        )
        object.__setattr__(
            self,
            "semantic_drift",
            _bounded01(self.semantic_drift, label="semantic_drift"),
        )
        if self.hallucination_energy is not None:
            object.__setattr__(
                self,
                "hallucination_energy",
                _bounded01(self.hallucination_energy, label="hallucination_energy"),
            )
        if self.verifiability is not None:
            object.__setattr__(
                self,
                "verifiability",
                _bounded01(self.verifiability, label="verifiability"),
            )
        object.__setattr__(
            self, "explanation", tuple(dict(item) for item in self.explanation)
        )

    @property
    def critic_risk(self) -> float:
        return 1.0 - self.critic_score

    @property
    def policy_gap(self) -> float:
        return 1.0 - self.policy_fit

    @property
    def evidence_gap(self) -> float:
        return 1.0 - self.evidence_support

    @property
    def citation_gap(self) -> float:
        return 1.0 - self.citation_match

    @property
    def computed_hallucination_energy(self) -> float:
        if self.hallucination_energy is not None:
            return float(self.hallucination_energy)
        return _clip01(
            (
                self.semantic_drift
                + self.evidence_gap
                + self.citation_gap
                + self.policy_gap
            )
            / 4.0
        )

    @property
    def computed_verifiability(self) -> float:
        if self.verifiability is not None:
            return float(self.verifiability)
        return _clip01(
            (self.evidence_support + self.citation_match + self.policy_fit) / 3.0
        )

    @property
    def risk_score(self) -> float:
        return _clip01(
            0.30 * self.computed_hallucination_energy
            + 0.20 * self.critic_risk
            + 0.15 * self.semantic_drift
            + 0.15 * self.policy_gap
            + 0.10 * self.evidence_gap
            + 0.10 * self.citation_gap
        )

    def metric_row(self) -> Tuple[float, ...]:
        return (
            self.risk_score,
            self.computed_hallucination_energy,
            self.semantic_drift,
            self.critic_risk,
            self.policy_gap,
            self.evidence_gap,
            self.citation_gap,
            self.computed_verifiability,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "critic_score": self.critic_score,
            "policy_fit": self.policy_fit,
            "evidence_support": self.evidence_support,
            "citation_match": self.citation_match,
            "semantic_drift": self.semantic_drift,
            "hallucination_energy": self.hallucination_energy,
            "computed_hallucination_energy": self.computed_hallucination_energy,
            "verifiability": self.verifiability,
            "computed_verifiability": self.computed_verifiability,
            "risk_score": self.risk_score,
            "verdict": self.verdict,
            "explanation": [dict(item) for item in self.explanation],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_critic_result(
        cls,
        result: Mapping[str, Any],
        *,
        item_id: Optional[str] = None,
        default_policy_fit: float = 1.0,
        default_evidence_support: float = 1.0,
        default_citation_match: float = 1.0,
        default_semantic_drift: float = 0.0,
    ) -> "CriticObservation":
        """Create an observation from a Writer-style critic result mapping."""
        features = _merge_features(result)
        resolved_item_id = (
            item_id or result.get("item_id") or result.get("id") or result.get("index")
        )
        if resolved_item_id is None:
            resolved_item_id = "critic_item"
        text = result.get("text") or result.get("claim")
        metadata = (
            dict(result.get("metadata") or {})
            if isinstance(result.get("metadata"), Mapping)
            else {}
        )
        if text is not None:
            metadata["text"] = str(text)
        if "threshold" in result:
            metadata["threshold"] = result["threshold"]
        if "label" in result:
            metadata["label"] = result["label"]

        return cls(
            item_id=str(resolved_item_id),
            critic_score=_bounded01(
                result.get("score", result.get("critic_score", 0.0)),
                label="critic_score",
            ),
            policy_fit=_clip01(
                _first_numeric(features, _NUMERIC_FEATURE_ALIASES["policy_fit"])
                or default_policy_fit
            ),
            evidence_support=_clip01(
                _first_numeric(features, _NUMERIC_FEATURE_ALIASES["evidence_support"])
                or default_evidence_support
            ),
            citation_match=_clip01(
                _first_numeric(features, _NUMERIC_FEATURE_ALIASES["citation_match"])
                or default_citation_match
            ),
            semantic_drift=_clip01(
                _first_numeric(features, _NUMERIC_FEATURE_ALIASES["semantic_drift"])
                or default_semantic_drift
            ),
            hallucination_energy=(
                _clip01(value)
                if (
                    value := _first_numeric(
                        features, _NUMERIC_FEATURE_ALIASES["hallucination_energy"]
                    )
                )
                is not None
                else None
            ),
            verifiability=(
                _clip01(value)
                if (
                    value := _first_numeric(
                        features, _NUMERIC_FEATURE_ALIASES["verifiability"]
                    )
                )
                is not None
                else None
            ),
            verdict=str(result.get("verdict"))
            if result.get("verdict") is not None
            else None,
            explanation=result.get("explanation") or (),
            metadata=metadata,
        )


@dataclass(frozen=True)
class CriticAssessment:
    """Risk-first VPM artifact and summary for critic observations."""

    artifact: VPMArtifact
    observations: Tuple[CriticObservation, ...]
    highest_risk_item_id: str
    highest_risk_score: float
    risky_count: int
    warnings: Tuple[str, ...]
    thresholds: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact.artifact_id,
            "highest_risk_item_id": self.highest_risk_item_id,
            "highest_risk_score": self.highest_risk_score,
            "risky_count": self.risky_count,
            "warnings": list(self.warnings),
            "thresholds": dict(self.thresholds),
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
        }


def critic_recipe() -> LayoutRecipe:
    """Default layout that places critic/policy risk first."""
    return LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "critic-risk-first",
            "row_order": {
                "kind": "lexicographic",
                "keys": [
                    {"metric_id": "risk_score", "direction": "desc"},
                    {"metric_id": "hallucination_energy", "direction": "desc"},
                    {"metric_id": "verifiability", "direction": "asc"},
                ],
                "tie_break": "row_id",
            },
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )


def observations_from_critic_lines(
    result: Mapping[str, Any],
) -> list[CriticObservation]:
    """Convert a Writer ``criticize_lines`` style result into observations."""
    items = result.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise VPMValidationError(
            "criticize_lines result must contain an items sequence"
        )
    observations: list[CriticObservation] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise VPMValidationError("criticize_lines items must be mappings")
        index = item.get("index", len(observations))
        observations.append(
            CriticObservation.from_critic_result(item, item_id="line_%s" % index)
        )
    return observations


def build_critic_vpm(
    observations: Sequence[CriticObservation | Mapping[str, Any]],
    *,
    recipe: Optional[LayoutRecipe] = None,
    risk_threshold: float = 0.50,
    hallucination_energy_threshold: float = 0.40,
    verifiability_threshold: float = 0.60,
    provenance: Optional[Mapping[str, Any]] = None,
) -> CriticAssessment:
    """Build a deterministic VPM for critic/evidence/policy observations."""
    normalized = tuple(
        item
        if isinstance(item, CriticObservation)
        else CriticObservation.from_critic_result(dict(item))
        for item in observations
    )
    if not normalized:
        raise VPMValidationError("build_critic_vpm requires at least one observation")
    row_ids = tuple(observation.item_id for observation in normalized)
    if len(set(row_ids)) != len(row_ids):
        raise VPMValidationError("Critic observations require unique item_id values")

    risk_threshold = _bounded01(risk_threshold, label="risk_threshold")
    hallucination_energy_threshold = _bounded01(
        hallucination_energy_threshold,
        label="hallucination_energy_threshold",
    )
    verifiability_threshold = _bounded01(
        verifiability_threshold, label="verifiability_threshold"
    )

    table = ScoreTable(
        values=[observation.metric_row() for observation in normalized],
        row_ids=row_ids,
        metric_ids=CRITIC_METRICS,
        metadata={
            "kind": "critic_evidence",
            "observations": [observation.to_dict() for observation in normalized],
        },
    )

    highest = max(
        normalized,
        key=lambda observation: (observation.risk_score, observation.item_id),
    )
    risky = [
        observation
        for observation in normalized
        if observation.risk_score >= risk_threshold
    ]
    warnings: list[str] = []
    if risky:
        warnings.append("risk_score_above_threshold")
    if any(
        observation.computed_hallucination_energy >= hallucination_energy_threshold
        for observation in normalized
    ):
        warnings.append("hallucination_energy_above_threshold")
    if any(
        observation.computed_verifiability < verifiability_threshold
        for observation in normalized
    ):
        warnings.append("verifiability_below_threshold")

    artifact = build_vpm(
        table,
        recipe or critic_recipe(),
        provenance={
            "kind": "critic_evidence",
            "parents": [],
            "highest_risk_item_id": highest.item_id,
            "highest_risk_score": highest.risk_score,
            "risky_count": len(risky),
            "warnings": warnings,
            **dict(provenance or {}),
        },
    )

    return CriticAssessment(
        artifact=artifact,
        observations=normalized,
        highest_risk_item_id=highest.item_id,
        highest_risk_score=highest.risk_score,
        risky_count=len(risky),
        warnings=tuple(warnings),
        thresholds={
            "risk_threshold": risk_threshold,
            "hallucination_energy_threshold": hallucination_energy_threshold,
            "verifiability_threshold": verifiability_threshold,
        },
    )
