"""Bertin-inspired pattern discovery over VPM score matrices.

This module searches for latent row structure inside a scored matrix instead of
only rendering a declared layout. The design contract:

1. Discovery may use numeric methods, but every published outcome is frozen:
   the discovered ordering compiles into an explicit ``row_order`` recipe, so
   the derived view artifact's identity depends on the outcome, never on
   re-running the procedure.
2. Confidence is calibrated against a selection-corrected null: the FULL
   pipeline (ordering + every objective) runs on each null sample (independent
   within-column permutations), and the family p-value compares the observed
   maximum standardized statistic against the null distribution of maxima.
3. Reports name structures numerically (orders, statistics, p-values). They do
   not assert domain meaning; geometry is not meaning.

Outputs are two linked artifacts:

- a pattern-report artifact (``relation: analyzes`` parent), mirroring the
  verification-report conventions in :mod:`zeromodel.policy_properties`;
- a reordered view artifact (``relation: derived_from`` the analyzed artifact,
  ``relation: ordered_by`` the report), built with the explicit row order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from zeromodel.core.artifact import (
    LayoutRecipe,
    ScoreTable,
    VPMArtifact,
    VPMValidationError,
    build_vpm,
)

PATTERN_CHECKER_VERSION = "bertin-pattern-detector/v1"
PATTERN_METHOD = "spectral_seriation/v1"
OBJECTIVE_IDS = ("adjacent_coherence", "anti_robinson")
REPORT_METRICS = (
    "observed",
    "null_mean",
    "null_std",
    "z_score",
    "p_value",
    "family_p_value",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class PatternAnalysisSpec:
    """Declares one deterministic pattern analysis.

    ``null_samples`` and ``seed`` control only the calibration; the discovered
    ordering itself is deterministic given the matrix and ``value_source``.
    ``alpha`` is part of the analytical declaration and therefore participates
    in the specification digest.
    """

    method: str = PATTERN_METHOD
    objectives: Tuple[str, ...] = OBJECTIVE_IDS
    value_source: str = "normalized"
    null_samples: int = 200
    seed: int = 0
    alpha: float = 0.05

    def validate(self) -> None:
        if self.method != PATTERN_METHOD:
            raise VPMValidationError("Unsupported pattern method: %r" % self.method)
        if self.value_source not in {"normalized", "raw"}:
            raise VPMValidationError(
                "value_source must be 'normalized' or 'raw', got %r"
                % self.value_source
            )
        if not self.objectives:
            raise VPMValidationError("At least one objective is required")
        for objective in self.objectives:
            if objective not in OBJECTIVE_IDS:
                raise VPMValidationError("Unknown objective: %r" % objective)
        if len(set(self.objectives)) != len(self.objectives):
            raise VPMValidationError("Objectives must be unique")
        if self.null_samples < 1:
            raise VPMValidationError("null_samples must be positive")
        if self.seed < 0:
            raise VPMValidationError("seed must be non-negative")
        if not np.isfinite(float(self.alpha)):
            raise VPMValidationError("alpha must be finite")
        if not (0.0 < float(self.alpha) < 1.0):
            raise VPMValidationError("alpha must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "objectives": list(self.objectives),
            "value_source": self.value_source,
            "null_samples": int(self.null_samples),
            "seed": int(self.seed),
            "alpha": float(self.alpha),
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()


@dataclass(frozen=True)
class ObjectiveResult:
    objective_id: str
    observed: float
    null_mean: float
    null_std: float
    z_score: float
    p_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "observed": self.observed,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "z_score": self.z_score,
            "p_value": self.p_value,
        }


@dataclass(frozen=True)
class PatternReport:
    """Frozen outcome of one pattern analysis over one artifact."""

    analyzed_artifact_id: str
    spec: PatternAnalysisSpec
    row_order: Tuple[str, ...]
    objective_results: Tuple[ObjectiveResult, ...]
    family_p_value: float
    degenerate: bool
    checker_version: str = PATTERN_CHECKER_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.spec.validate()
        if not self.analyzed_artifact_id:
            raise VPMValidationError("PatternReport requires analyzed_artifact_id")
        if not self.row_order:
            raise VPMValidationError("PatternReport requires a non-empty row_order")
        if len(set(self.row_order)) != len(self.row_order):
            raise VPMValidationError("PatternReport row_order must be unique")
        if not self.objective_results:
            raise VPMValidationError("PatternReport requires objective results")
        if not np.isfinite(float(self.family_p_value)):
            raise VPMValidationError("PatternReport family_p_value must be finite")
        if not (0.0 <= float(self.family_p_value) <= 1.0):
            raise VPMValidationError("PatternReport family_p_value must be in [0, 1]")
        _canonical_json_bytes(dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checker_version": self.checker_version,
            "analyzed_artifact_id": self.analyzed_artifact_id,
            "spec": self.spec.to_dict(),
            "spec_digest": self.spec.digest,
            "row_order": list(self.row_order),
            "objectives": [result.to_dict() for result in self.objective_results],
            "family_p_value": self.family_p_value,
            "degenerate": self.degenerate,
            "metadata": dict(self.metadata),
        }

    @property
    def significant(self) -> bool:
        """Whether the family-level null test rejects at the declared alpha."""

        return (not self.degenerate) and self.family_p_value <= self.spec.alpha

    @property
    def primary_objective(self) -> str:
        """Return the objective with the largest standardized observed effect."""

        return max(
            self.objective_results,
            key=lambda result: (float(result.z_score), result.objective_id),
        ).objective_id

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()

    def to_vpm(self) -> VPMArtifact:
        """Materialize this report as an identity-bearing pattern artifact."""

        values = [
            (
                result.observed,
                result.null_mean,
                result.null_std,
                result.z_score,
                result.p_value,
                self.family_p_value,
            )
            for result in self.objective_results
        ]
        table = ScoreTable(
            values=values,
            row_ids=[result.objective_id for result in self.objective_results],
            metric_ids=list(REPORT_METRICS),
            metadata={
                "kind": "pattern_report",
                "analyzed_artifact_id": self.analyzed_artifact_id,
                "checker_version": self.checker_version,
                "pattern_spec_digest": self.spec.digest,
                "report": self.to_dict(),
            },
        )
        recipe = LayoutRecipe.from_dict(
            {
                "version": "vpm-layout/0",
                "name": "pattern-report-source-order",
                "row_order": {"kind": "source", "tie_break": "row_id"},
                "column_order": {"kind": "source"},
                "normalization": {"kind": "per_metric_minmax", "clip": True},
            }
        )
        return build_vpm(
            table,
            recipe,
            provenance={
                "kind": "pattern_report",
                "checker": self.checker_version,
                "pattern_spec_digest": self.spec.digest,
                "parents": [
                    {
                        "artifact_id": self.analyzed_artifact_id,
                        "relation": "analyzes",
                    }
                ],
            },
        )


@dataclass(frozen=True)
class PatternDiscoveryArtifacts:
    """Complete materialized output of one pattern-discovery run.

    Returning the report, report artifact, and view artifact together prevents
    the convenience API from creating a view whose ``ordered_by`` parent is
    silently discarded by the caller.
    """

    report: PatternReport
    report_artifact: VPMArtifact
    view_artifact: VPMArtifact


# ---------------------------------------------------------------------------
# Deterministic spectral seriation
# ---------------------------------------------------------------------------

def _source_matrix(artifact: VPMArtifact, value_source: str) -> np.ndarray:
    values = np.asarray(artifact.source.values, dtype=np.float64)
    if value_source == "raw":
        return values
    lo = values.min(axis=0, keepdims=True)
    hi = values.max(axis=0, keepdims=True)
    span = np.where(hi > lo, hi - lo, 1.0)
    return np.clip((values - lo) / span, 0.0, 1.0)


def _pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    squared = np.sum(matrix**2, axis=1)
    gram = matrix @ matrix.T
    distances = squared[:, None] + squared[None, :] - 2.0 * gram
    np.maximum(distances, 0.0, out=distances)
    return np.sqrt(distances)


def _spectral_order(
    matrix: np.ndarray, row_ids: Sequence[str]
) -> Tuple[Tuple[int, ...], bool]:
    """Order rows by the Fiedler vector of the similarity Laplacian.

    Deterministic: symmetric ``eigh``, canonical sign fixing (first component
    above tolerance made positive), and ``row_id`` tie-breaking on plateaus.
    Returns (source-index order, degenerate flag).
    """

    n = matrix.shape[0]
    identity_order = tuple(range(n))
    if n <= 2:
        return identity_order, True
    distances = _pairwise_distances(matrix)
    max_distance = float(distances.max())
    if max_distance <= 0.0:
        return identity_order, True
    similarity = max_distance - distances
    np.fill_diagonal(similarity, 0.0)
    laplacian = np.diag(similarity.sum(axis=1)) - similarity
    _, eigenvectors = np.linalg.eigh(laplacian)
    fiedler = eigenvectors[:, 1]
    above_tolerance = np.flatnonzero(np.abs(fiedler) > 1e-12)
    if above_tolerance.size == 0:
        return identity_order, True
    if fiedler[above_tolerance[0]] < 0.0:
        fiedler = -fiedler
    order = tuple(
        sorted(range(n), key=lambda index: (float(fiedler[index]), row_ids[index]))
    )
    return order, False


# ---------------------------------------------------------------------------
# Structural objectives (higher = more structure)
# ---------------------------------------------------------------------------

def _adjacent_coherence(ordered_matrix: np.ndarray) -> float:
    """Negative mean squared difference between vertically adjacent cells."""

    if ordered_matrix.shape[0] < 2:
        return 0.0
    deltas = np.diff(ordered_matrix, axis=0)
    return -float(np.mean(deltas**2))


def _inversions(vector: np.ndarray) -> int:
    if vector.size < 2:
        return 0
    return int(np.sum(np.triu(vector[:, None] > vector[None, :], k=1)))


def _anti_robinson(ordered_distances: np.ndarray) -> float:
    """Negative count of anti-Robinson events in the ordered distance matrix."""

    n = ordered_distances.shape[0]
    events = 0
    for i in range(n - 2):
        events += _inversions(ordered_distances[i, i + 1 :])
    for k in range(2, n):
        events += _inversions(ordered_distances[:k, k][::-1])
    return -float(events)


def _score_arrangement(
    matrix: np.ndarray,
    order: Sequence[int],
    objectives: Sequence[str],
) -> Dict[str, float]:
    ordered = matrix[np.asarray(order, dtype=int)]
    scores: Dict[str, float] = {}
    if "adjacent_coherence" in objectives:
        scores["adjacent_coherence"] = _adjacent_coherence(ordered)
    if "anti_robinson" in objectives:
        scores["anti_robinson"] = _anti_robinson(_pairwise_distances(ordered))
    return scores


def _run_pipeline(
    matrix: np.ndarray,
    row_ids: Sequence[str],
    objectives: Sequence[str],
) -> Tuple[Tuple[int, ...], Dict[str, float], bool]:
    order, degenerate = _spectral_order(matrix, row_ids)
    return order, _score_arrangement(matrix, order, objectives), degenerate


# ---------------------------------------------------------------------------
# Selection-corrected null calibration
# ---------------------------------------------------------------------------

def detect_patterns(
    artifact: VPMArtifact,
    spec: Optional[PatternAnalysisSpec] = None,
) -> PatternReport:
    """Run pattern discovery with selection-corrected null calibration.

    Every null sample destroys joint row structure with independent
    within-column permutations while preserving column marginals, then runs
    the complete discovery pipeline, so the observed statistics are compared
    against nulls that received the same optimization opportunity.
    """

    spec = spec or PatternAnalysisSpec()
    spec.validate()
    artifact.validate()

    matrix = _source_matrix(artifact, spec.value_source)
    row_ids = artifact.source.row_ids
    order, observed_scores, degenerate = _run_pipeline(
        matrix, row_ids, spec.objectives
    )

    rng = np.random.default_rng(spec.seed)
    null_scores: Dict[str, list] = {objective: [] for objective in spec.objectives}
    null_row_ids = tuple("null:%06d" % index for index in range(matrix.shape[0]))
    for _ in range(spec.null_samples):
        null_matrix = np.empty_like(matrix)
        for column in range(matrix.shape[1]):
            null_matrix[:, column] = matrix[
                rng.permutation(matrix.shape[0]), column
            ]
        _, scores, _ = _run_pipeline(null_matrix, null_row_ids, spec.objectives)
        for objective, score in scores.items():
            null_scores[objective].append(score)

    results = []
    observed_z: Dict[str, float] = {}
    null_z_rows = np.zeros((spec.null_samples, len(spec.objectives)))
    for column_index, objective in enumerate(spec.objectives):
        null_values = np.asarray(null_scores[objective], dtype=np.float64)
        null_mean = float(null_values.mean())
        null_std = float(null_values.std())
        safe_std = null_std if null_std > 0.0 else 1.0
        observed = float(observed_scores[objective])
        z_score = (observed - null_mean) / safe_std
        exceed = int(np.sum(null_values >= observed))
        p_value = (1.0 + exceed) / (1.0 + spec.null_samples)
        observed_z[objective] = z_score
        null_z_rows[:, column_index] = (null_values - null_mean) / safe_std
        results.append(
            ObjectiveResult(
                objective_id=objective,
                observed=observed,
                null_mean=null_mean,
                null_std=null_std,
                z_score=z_score,
                p_value=p_value,
            )
        )

    observed_max_z = max(observed_z.values())
    null_max_z = null_z_rows.max(axis=1)
    family_exceed = int(np.sum(null_max_z >= observed_max_z))
    family_p = (1.0 + family_exceed) / (1.0 + spec.null_samples)

    return PatternReport(
        analyzed_artifact_id=artifact.artifact_id,
        spec=spec,
        row_order=tuple(row_ids[index] for index in order),
        objective_results=tuple(results),
        family_p_value=family_p,
        degenerate=degenerate,
    )


class MatrixPatternDetector:
    """Configured Bertin-inspired detector over immutable VPM artifacts.

    The detector is intentionally compile-time analysis. It does not change
    runtime policy semantics. ``detect`` returns a frozen report;
    ``materialize`` returns the complete report/report-artifact/view-artifact
    set so no lineage parent is dropped by the convenience path.
    """

    def __init__(self, spec: Optional[PatternAnalysisSpec] = None) -> None:
        self.spec = spec or PatternAnalysisSpec()
        self.spec.validate()

    def detect(self, artifact: VPMArtifact) -> PatternReport:
        return detect_patterns(artifact, self.spec)

    def build_view(
        self,
        artifact: VPMArtifact,
        report: PatternReport,
        report_artifact: VPMArtifact,
        *,
        name: str = "bertin-discovered-order",
    ) -> VPMArtifact:
        return build_discovered_view(
            artifact,
            report,
            report_artifact,
            name=name,
        )

    def materialize(
        self,
        artifact: VPMArtifact,
        *,
        name: str = "bertin-discovered-order",
    ) -> PatternDiscoveryArtifacts:
        """Build and return the complete linked discovery artifact set."""

        report = self.detect(artifact)
        report_artifact = report.to_vpm()
        view_artifact = self.build_view(
            artifact,
            report,
            report_artifact,
            name=name,
        )
        return PatternDiscoveryArtifacts(
            report=report,
            report_artifact=report_artifact,
            view_artifact=view_artifact,
        )


def build_discovered_view(
    artifact: VPMArtifact,
    report: PatternReport,
    report_artifact: VPMArtifact,
    *,
    name: str = "bertin-discovered-order",
) -> VPMArtifact:
    """Compile a materialized report's ordering into a frozen view artifact.

    Requiring the report artifact prevents the API from silently minting an
    ``ordered_by`` parent that the caller never receives.
    """

    if report.analyzed_artifact_id != artifact.artifact_id:
        raise VPMValidationError(
            "Report analyzes artifact %s, not %s"
            % (report.analyzed_artifact_id, artifact.artifact_id)
        )
    expected_report_artifact = report.to_vpm()
    if report_artifact.artifact_id != expected_report_artifact.artifact_id:
        raise VPMValidationError(
            "report_artifact does not materialize the supplied PatternReport"
        )
    report_parents = tuple(report_artifact.provenance.get("parents") or ())
    if not any(
        parent.get("relation") == "analyzes"
        and parent.get("artifact_id") == artifact.artifact_id
        for parent in report_parents
        if isinstance(parent, Mapping)
    ):
        raise VPMValidationError(
            "report_artifact must link to the analyzed artifact"
        )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": name,
            "row_order": {
                "kind": "explicit",
                "row_ids": list(report.row_order),
                "tie_break": "row_id",
            },
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(
        artifact.source,
        recipe,
        provenance={
            "kind": "pattern_discovered_view",
            "pattern_spec_digest": report.spec.digest,
            "pattern_report_digest": report.digest,
            "parents": [
                {"artifact_id": artifact.artifact_id, "relation": "derived_from"},
                {"artifact_id": report_artifact.artifact_id, "relation": "ordered_by"},
            ],
        },
    )
