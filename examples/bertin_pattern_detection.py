"""Recover planted matrix structure and freeze the result as VPM artifacts.

This is a bounded research fixture for the Bertin-inspired pattern detector.
It does not assign domain meaning to the recovered geometry. It demonstrates:

- deterministic spectral seriation;
- selection-corrected permutation-null calibration;
- an identity-bearing pattern report;
- an explicit-order discovered view over the unchanged source table.

Run:

    python examples/bertin_pattern_detection.py
    python examples/bertin_pattern_detection.py --output-dir build/patterns
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from zeromodel import LayoutRecipe, ScoreTable, build_vpm, to_bundle, write_png
from zeromodel.patterns import MatrixPatternDetector, PatternAnalysisSpec


def planted_block_matrix(*, seed: int = 7, noise: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Return a shuffled three-block matrix and its hidden row labels."""

    rng = np.random.default_rng(seed)
    signatures = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.repeat(np.arange(3), 8)
    matrix = signatures[labels] + rng.normal(0.0, noise, size=(24, 6))
    shuffle = rng.permutation(matrix.shape[0])
    return matrix[shuffle], labels[shuffle]


def source_artifact(matrix: np.ndarray):
    table = ScoreTable(
        values=matrix,
        row_ids=["state:%03d" % index for index in range(matrix.shape[0])],
        metric_ids=["signal:%d" % index for index in range(matrix.shape[1])],
        metadata={"kind": "bertin-pattern-recovery-fixture"},
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "bertin-pattern-source-order",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(table, recipe, provenance={"kind": "pattern-source"})


def block_adjacency(row_order: tuple[str, ...], row_ids: tuple[str, ...], labels: np.ndarray) -> float:
    label_by_row = dict(zip(row_ids, labels))
    ordered = [label_by_row[row_id] for row_id in row_order]
    return sum(left == right for left, right in zip(ordered, ordered[1:])) / float(len(ordered) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--null-samples", type=int, default=199)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    matrix, labels = planted_block_matrix()
    source = source_artifact(matrix)
    detector = MatrixPatternDetector(
        PatternAnalysisSpec(
            null_samples=args.null_samples,
            seed=args.seed,
            alpha=args.alpha,
        )
    )
    result = detector.materialize(source)
    report = result.report
    report_artifact = result.report_artifact
    discovered_view = result.view_artifact
    adjacency = block_adjacency(report.row_order, source.source.row_ids, labels)

    output = {
        "source_artifact_id": source.artifact_id,
        "pattern_report_artifact_id": report_artifact.artifact_id,
        "discovered_view_artifact_id": discovered_view.artifact_id,
        "pattern_spec_digest": report.spec.digest,
        "primary_objective": report.primary_objective,
        "family_p_value": report.family_p_value,
        "alpha": report.spec.alpha,
        "significant": report.significant,
        "same_block_adjacency": adjacency,
        "row_count": matrix.shape[0],
        "metric_count": matrix.shape[1],
    }

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        to_bundle(source, args.output_dir / "pattern-source.vpm")
        to_bundle(report_artifact, args.output_dir / "pattern-report.vpm")
        to_bundle(discovered_view, args.output_dir / "pattern-view.vpm")
        write_png(discovered_view, args.output_dir / "pattern-view.png")
        output["output_dir"] = str(args.output_dir)

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
