"""Metric packing helpers for ZeroModel v2.

These helpers pull the useful Stephanie metric-normalization pattern into the
standalone package. They convert application-specific metric dictionaries into
stable numeric rows that can be used by ``ScoreTable``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from zeromodel.core.artifact import ScoreTable

CANONICAL_METRICS = (
    "overall",
    "coverage",
    "correctness",
    "coherence",
    "faithfulness",
    "citation_support",
    "entity_consistency",
    "readability",
    "novelty",
    "stickiness",
    "structure",
    "no_halluc",
    "figure_ground",
    "tests_pass_rate",
    "mutation_score",
    "complexity",
    "type_safe",
    "lint_clean",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def pack_metrics(
    data: Mapping[str, Any], metric_ids: Optional[Sequence[str]] = None
) -> Dict[str, float]:
    """Pack heterogeneous metric keys into a stable float dictionary.

    The function intentionally supports common aliases found in Stephanie:
    ``claim_coverage`` -> ``coverage``; ``hallucination_rate`` ->
    ``no_halluc``; ``figure_results.overall_figure_score`` ->
    ``figure_ground``.
    """
    src: Dict[str, Any] = dict(data or {})

    packed: Dict[str, float] = {}
    packed["overall"] = _as_float(src.get("overall"))
    packed["coverage"] = _as_float(src.get("coverage", src.get("claim_coverage")))
    packed["correctness"] = _as_float(src.get("correctness"))
    packed["coherence"] = _as_float(src.get("coherence"))
    packed["faithfulness"] = _as_float(src.get("faithfulness"))
    packed["citation_support"] = _as_float(
        src.get("citation_support", src.get("evidence_strength"))
    )
    packed["entity_consistency"] = _as_float(src.get("entity_consistency"))
    packed["readability"] = _as_float(src.get("readability"))
    packed["novelty"] = _as_float(src.get("novelty"))
    packed["stickiness"] = _as_float(src.get("stickiness"))
    packed["structure"] = _as_float(src.get("structure"))
    packed["tests_pass_rate"] = _as_float(src.get("tests_pass_rate"))
    packed["mutation_score"] = _as_float(src.get("mutation_score"))
    packed["complexity"] = _as_float(src.get("complexity", src.get("complexity_ok")))
    packed["type_safe"] = _as_float(src.get("type_safe"))
    packed["lint_clean"] = _as_float(src.get("lint_clean"))

    if "hallucination_rate" in src:
        packed["no_halluc"] = 1.0 - _as_float(src.get("hallucination_rate"), 1.0)
    else:
        packed["no_halluc"] = _as_float(src.get("no_halluc"))

    figure_results = src.get("figure_results", {})
    if isinstance(figure_results, Mapping):
        packed["figure_ground"] = _as_float(figure_results.get("overall_figure_score"))
    else:
        packed["figure_ground"] = _as_float(src.get("figure_ground"))

    # Preserve additional numeric metrics so applications can extend the schema.
    for key, value in src.items():
        if key not in packed and isinstance(value, (int, float)):
            packed[str(key)] = float(value)

    wanted = tuple(metric_ids or CANONICAL_METRICS)
    return {metric_id: float(packed.get(metric_id, 0.0)) for metric_id in wanted}


def metric_ids_for_rows(
    rows: Iterable[Mapping[str, Any]], preferred: Sequence[str] = CANONICAL_METRICS
) -> List[str]:
    """Return a stable metric id list for a collection of metric dictionaries."""
    seen = set()
    result: List[str] = []
    for metric_id in preferred:
        result.append(str(metric_id))
        seen.add(str(metric_id))
    for row in rows:
        for key, value in dict(row or {}).items():
            if isinstance(value, (int, float)) and key not in seen:
                result.append(str(key))
                seen.add(str(key))
    return result


def score_table_from_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    row_ids: Optional[Sequence[str]] = None,
    metric_ids: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ScoreTable:
    """Build a ``ScoreTable`` from application metric dictionaries."""
    ids = tuple(
        str(row_id)
        for row_id in (row_ids or ["row_%04d" % i for i in range(len(rows))])
    )
    metrics = tuple(
        str(metric_id) for metric_id in (metric_ids or metric_ids_for_rows(rows))
    )
    values = [
        [pack_metrics(row, metrics)[metric_id] for metric_id in metrics] for row in rows
    ]
    return ScoreTable(
        values=values, row_ids=ids, metric_ids=metrics, metadata=metadata or {}
    )
