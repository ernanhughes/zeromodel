#!/usr/bin/env python3
"""Comparison and classification logic for the controlled PNG representation
benchmark. Never added to the production `Store` protocol - this is a
presentation/decision layer over `MaterializedProviderEvaluationRunDTO`
summaries that the Stage 2D aggregate already owns as source truth.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from zeromodel.video.domains.video_action_set.provider_evaluation_dto import (
    MaterializedProviderEvaluationRunDTO,
    ProviderEvaluationRunDTO,
)


class IncompatibleRunsError(ValueError):
    """Raised when two runs cannot be legitimately compared."""


FIXED_IDENTITY_FIELDS = (
    "provider_configuration_id",
    "model_digest",
    "prompt_digest",
    "protocol_version",
    "policy_artifact_id",
    "fixture_identity",
    "case_mode",
)

GENERIC_TARGET_METRICS = ("exact_count", "rejected_count", "latency_median_us")
COOLDOWN_TARGET_METRICS = ("cooldown",)
LANE_TARGET_METRICS = ("tank_column", "target_column")
LATENCY_MATERIAL_IMPROVEMENT_RATIO = 0.10


@dataclass(frozen=True, slots=True)
class RepresentationComparisonRow:
    """One variant's summary counts, projected for comparison output.

    Counts are copied verbatim from `ProviderEvaluationSummaryDTO` (the
    identity-bearing source of truth); this row exists only to make
    presentation/report generation convenient, never as a second source of
    truth.
    """

    variant_id: str
    recipe_id: str
    run_id: str
    exact_count: int
    action_equivalent_count: int
    action_changing_count: int
    rejected_count: int
    action_correct_count: int
    factor_counts: Mapping[str, object]
    latency_median_us: int | None
    latency_p95_us: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "recipe_id": self.recipe_id,
            "run_id": self.run_id,
            "exact_count": self.exact_count,
            "action_equivalent_count": self.action_equivalent_count,
            "action_changing_count": self.action_changing_count,
            "rejected_count": self.rejected_count,
            "action_correct_count": self.action_correct_count,
            "factor_counts": dict(self.factor_counts),
            "latency_median_us": self.latency_median_us,
            "latency_p95_us": self.latency_p95_us,
        }


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    label: str
    reasoning: str


def _configuration_fingerprint(run: ProviderEvaluationRunDTO) -> dict[str, str]:
    cfg = run.provider_configuration
    return {
        "provider_configuration_id": cfg.provider_configuration_id,
        "model_digest": cfg.model_digest,
        "prompt_digest": cfg.prompt_digest,
        "protocol_version": cfg.protocol_version,
        "policy_artifact_id": run.policy_artifact_id,
        "fixture_identity": run.fixture_identity,
        "case_mode": run.case_mode,
    }


def validate_comparable_runs(
    runs: Sequence[MaterializedProviderEvaluationRunDTO],
) -> None:
    """Reject any comparison where a fixed-identity dimension differs.

    `representation_mode` and the recipe id are expected (and required) to
    differ - only they, and the resulting evidence, may vary between runs
    being compared.
    """
    if len(runs) < 2:
        return
    reference = runs[0].run
    reference_fingerprint = _configuration_fingerprint(reference)
    for materialized in runs[1:]:
        candidate = materialized.run
        candidate_fingerprint = _configuration_fingerprint(candidate)
        mismatches = [
            field
            for field in FIXED_IDENTITY_FIELDS
            if reference_fingerprint[field] != candidate_fingerprint[field]
        ]
        if mismatches:
            raise IncompatibleRunsError(
                f"runs {reference.run_id} and {candidate.run_id} are not "
                f"comparable: differing {', '.join(mismatches)}"
            )


def _latency_materially_improved(
    base_median: int | None, candidate_median: int | None
) -> bool:
    if base_median is None or candidate_median is None or base_median == 0:
        return False
    return (
        base_median - candidate_median
    ) / base_median >= LATENCY_MATERIAL_IMPROVEMENT_RATIO


def _declared_target_improvements(
    *,
    base_summary,
    candidate_summary,
    target_metrics: Sequence[str],
) -> list[str]:
    improved: list[str] = []
    if (
        "exact_count" in target_metrics
        and candidate_summary.exact_count > base_summary.exact_count
    ):
        improved.append("exact_count")
    if (
        "rejected_count" in target_metrics
        and candidate_summary.rejected_count < base_summary.rejected_count
    ):
        improved.append("rejected_count")
    if "latency_median_us" in target_metrics and _latency_materially_improved(
        base_summary.latency_median_us, candidate_summary.latency_median_us
    ):
        improved.append("latency_median_us")
    base_factor = base_summary.factor_correct_counts.to_value()
    candidate_factor = candidate_summary.factor_correct_counts.to_value()
    for key in target_metrics:
        if (
            key in base_factor
            and key in candidate_factor
            and candidate_factor[key] > base_factor[key]
        ):
            improved.append(key)
    return improved


def classify_variant(
    *,
    baseline: MaterializedProviderEvaluationRunDTO,
    candidate: MaterializedProviderEvaluationRunDTO,
    target_metrics: Sequence[str] = GENERIC_TARGET_METRICS,
) -> ClassificationResult:
    """Classify `candidate` against `baseline` as advance / no_material_change
    / regression / incompatible, per
    `docs/research/controlled-png-representation-benchmark.md`.
    """
    try:
        validate_comparable_runs((baseline, candidate))
    except IncompatibleRunsError as exc:
        return ClassificationResult(label="incompatible", reasoning=str(exc))

    base_summary = baseline.summary
    candidate_summary = candidate.summary

    if candidate_summary.action_changing_count > base_summary.action_changing_count:
        return ClassificationResult(
            "regression",
            "action_changing_count increased "
            f"({base_summary.action_changing_count} -> {candidate_summary.action_changing_count})",
        )
    if candidate_summary.rejected_count > base_summary.rejected_count:
        return ClassificationResult(
            "regression",
            "rejected_count increased "
            f"({base_summary.rejected_count} -> {candidate_summary.rejected_count})",
        )

    base_factor = base_summary.factor_correct_counts.to_value()
    candidate_factor = candidate_summary.factor_correct_counts.to_value()
    worsened = sorted(
        key
        for key in target_metrics
        if key in base_factor
        and key in candidate_factor
        and candidate_factor[key] < base_factor[key]
    )
    if worsened:
        return ClassificationResult(
            "regression", f"targeted factor(s) worsened: {', '.join(worsened)}"
        )

    improvements = _declared_target_improvements(
        base_summary=base_summary,
        candidate_summary=candidate_summary,
        target_metrics=target_metrics,
    )
    if improvements:
        return ClassificationResult(
            "advance", f"declared target metric(s) improved: {', '.join(improvements)}"
        )
    return ClassificationResult(
        "no_material_change",
        "compatible run with no regression and no declared target metric improvement",
    )


def build_comparison_rows(
    runs_by_variant: Mapping[str, MaterializedProviderEvaluationRunDTO],
    recipe_ids_by_variant: Mapping[str, str],
) -> list[RepresentationComparisonRow]:
    rows: list[RepresentationComparisonRow] = []
    for variant_id in sorted(runs_by_variant):
        run = runs_by_variant[variant_id]
        summary = run.summary
        rows.append(
            RepresentationComparisonRow(
                variant_id=variant_id,
                recipe_id=recipe_ids_by_variant[variant_id],
                run_id=run.run.run_id,
                exact_count=summary.exact_count,
                action_equivalent_count=summary.action_equivalent_count,
                action_changing_count=summary.action_changing_count,
                rejected_count=summary.rejected_count,
                action_correct_count=summary.action_correct_count,
                factor_counts=summary.factor_correct_counts.to_value(),
                latency_median_us=summary.latency_median_us,
                latency_p95_us=summary.latency_p95_us,
            )
        )
    return rows


def build_compatibility_statement(reference_run: ProviderEvaluationRunDTO) -> str:
    cfg = reference_run.provider_configuration
    return (
        "All compared runs share provider_configuration_id="
        f"{cfg.provider_configuration_id}, model_digest={cfg.model_digest}, "
        f"prompt_digest={cfg.prompt_digest}, policy_artifact_id="
        f"{reference_run.policy_artifact_id}, fixture_identity="
        f"{reference_run.fixture_identity}, case_mode={reference_run.case_mode}. "
        "Only representation_mode and recipe_id varied across the compared runs."
    )


def write_comparison_json(
    path: Path,
    *,
    rows: Sequence[RepresentationComparisonRow],
    classifications: Mapping[str, ClassificationResult],
    compatibility_statement: str,
) -> None:
    payload = {
        "compatibility_statement": compatibility_statement,
        "rows": [row.to_dict() for row in rows],
        "classifications": {
            variant_id: {"label": result.label, "reasoning": result.reasoning}
            for variant_id, result in classifications.items()
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_comparison_csv(
    path: Path, rows: Sequence[RepresentationComparisonRow]
) -> None:
    fieldnames = [
        "variant_id",
        "recipe_id",
        "run_id",
        "exact_count",
        "action_equivalent_count",
        "action_changing_count",
        "rejected_count",
        "action_correct_count",
        "latency_median_us",
        "latency_p95_us",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def write_comparison_md(
    path: Path,
    *,
    rows: Sequence[RepresentationComparisonRow],
    classifications: Mapping[str, ClassificationResult],
    compatibility_statement: str,
) -> None:
    lines = [
        "# Representation comparison",
        "",
        compatibility_statement,
        "",
        (
            "| Variant | Recipe | Exact | Action-equiv | Action-changing | Rejected | "
            "Action-correct | Latency median (us) | Latency p95 (us) | Classification |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        result = classifications.get(row.variant_id)
        label = result.label if result is not None else "n/a"
        lines.append(
            f"| {row.variant_id} | `{row.recipe_id[:16]}...` | {row.exact_count} | "
            f"{row.action_equivalent_count} | {row.action_changing_count} | "
            f"{row.rejected_count} | {row.action_correct_count} | "
            f"{row.latency_median_us if row.latency_median_us is not None else '-'} | "
            f"{row.latency_p95_us if row.latency_p95_us is not None else '-'} | {label} |"
        )
    lines.append("")
    lines.append("## Classification reasoning")
    lines.append("")
    for variant_id in sorted(classifications):
        result = classifications[variant_id]
        lines.append(f"- **{variant_id}**: {result.label} - {result.reasoning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "COOLDOWN_TARGET_METRICS",
    "FIXED_IDENTITY_FIELDS",
    "GENERIC_TARGET_METRICS",
    "LANE_TARGET_METRICS",
    "ClassificationResult",
    "IncompatibleRunsError",
    "RepresentationComparisonRow",
    "build_comparison_rows",
    "build_compatibility_statement",
    "classify_variant",
    "validate_comparable_runs",
    "write_comparison_csv",
    "write_comparison_json",
    "write_comparison_md",
]
