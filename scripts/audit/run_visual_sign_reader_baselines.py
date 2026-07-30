from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.video.arcade_policy import (
    ACTIONS,
    ShooterConfig,
    compile_policy_artifact,
    enumerate_visual_frames,
    parse_state_row_id,
)
from zeromodel.vision.visual import extract_visual_features

from run_visual_sign_reader_perturbations import (
    RESULT_ROOT,
    arcade_visual_feature_spec,
    write_json,
)


def exact_codebook(
    row_ids: tuple[str, ...],
    features: np.ndarray,
    queries: np.ndarray,
) -> tuple[str | None, ...]:
    table = {
        np.asarray(vector, dtype=np.uint8).tobytes(order="C"): row_id
        for row_id, vector in zip(row_ids, features)
    }
    return tuple(
        table.get(np.asarray(query, dtype=np.uint8).tobytes(order="C"))
        for query in queries
    )


def nearest_neighbour(
    row_ids: tuple[str, ...],
    features: np.ndarray,
    queries: np.ndarray,
    *,
    threshold: float,
    required_margin: float,
) -> tuple[str | None, ...]:
    matrix = np.asarray(features, dtype=np.float64)
    output: list[str | None] = []
    for query in np.asarray(queries, dtype=np.float64):
        distances = np.sqrt(np.sum((matrix - query[None, :]) ** 2, axis=1))
        ranking = sorted(
            range(len(row_ids)),
            key=lambda index: (float(distances[index]), row_ids[index]),
        )
        first, second = ranking[0], ranking[1]
        if (
            float(distances[first]) <= threshold + 1e-12
            and float(distances[second] - distances[first]) + 1e-12 >= required_margin
        ):
            output.append(row_ids[first])
        else:
            output.append(None)
    return tuple(output)


def direct_symbolic(row_ids: tuple[str, ...]) -> tuple[str, ...]:
    for row_id in row_ids:
        parse_state_row_id(row_id)
    return row_ids


def summarize(
    expected: tuple[str, ...],
    observed: tuple[str | None, ...],
    lookup: VPMPolicyLookup,
) -> dict[str, Any]:
    accepted = [value is not None for value in observed]
    return {
        "row_count": len(expected),
        "accepted_count": sum(accepted),
        "rejected_count": len(expected) - sum(accepted),
        "correct_row_count": sum(
            int(value is not None and value == row_id)
            for row_id, value in zip(expected, observed)
        ),
        "incorrect_row_count": sum(
            int(value is not None and value != row_id)
            for row_id, value in zip(expected, observed)
        ),
        "correct_action_count": sum(
            int(value is not None and lookup.choose(value) == lookup.choose(row_id))
            for row_id, value in zip(expected, observed)
        ),
        "incorrect_action_count": sum(
            int(value is not None and lookup.choose(value) != lookup.choose(row_id))
            for row_id, value in zip(expected, observed)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()

    config = ShooterConfig()
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    frames = dict(enumerate_visual_frames(config))
    row_ids = tuple(policy.source.row_ids)
    spec = arcade_visual_feature_spec(config)
    features = np.asarray(
        [extract_visual_features(frames[row_id], spec) for row_id in row_ids],
        dtype=np.uint8,
    )
    pairwise = np.sqrt(
        np.maximum(
            np.sum(features.astype(np.float64) ** 2, axis=1)[:, None]
            + np.sum(features.astype(np.float64) ** 2, axis=1)[None, :]
            - 2.0 * (features.astype(np.float64) @ features.astype(np.float64).T),
            0.0,
        )
    )
    np.fill_diagonal(pairwise, np.inf)
    min_between = float(pairwise.min())
    threshold = min_between * 0.25
    required_margin = min_between * 0.75

    started = time.perf_counter()
    exact = exact_codebook(row_ids, features, features)
    exact_seconds = time.perf_counter() - started

    started = time.perf_counter()
    nearest = nearest_neighbour(
        row_ids,
        features,
        features,
        threshold=threshold,
        required_margin=required_margin,
    )
    nearest_seconds = time.perf_counter() - started

    started = time.perf_counter()
    symbolic = direct_symbolic(row_ids)
    symbolic_seconds = time.perf_counter() - started

    payload = {
        "baseline_a_exact_codebook": {
            **summarize(row_ids, exact, lookup),
            "implementation": "feature bytes dictionary -> row_id",
            "lookup_seconds": exact_seconds,
            "serialized_size_bytes": len(
                json.dumps(
                    {
                        row_id: vector.tolist()
                        for row_id, vector in zip(row_ids, features)
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ),
            "identity_digest": hashlib.sha256(features.tobytes(order="C")).hexdigest(),
        },
        "baseline_b_plain_nearest_neighbour": {
            **summarize(row_ids, nearest, lookup),
            "implementation": "NumPy exhaustive Euclidean distance with fixed threshold and margin",
            "lookup_seconds": nearest_seconds,
            "threshold": threshold,
            "required_margin": required_margin,
        },
        "baseline_c_direct_symbolic_state": {
            **summarize(row_ids, symbolic, lookup),
            "implementation": "arcade fixture direct symbolic row_id",
            "lookup_seconds": symbolic_seconds,
        },
        "zeromodel_adds": [
            "policy and visual index artifact identities",
            "policy/index mismatch rejection",
            "accepted and rejected JSON-safe traces",
            "bundle round-trip validation",
            "calibration metadata and provenance",
        ],
    }
    write_json(args.output_root / "baseline-comparison.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
