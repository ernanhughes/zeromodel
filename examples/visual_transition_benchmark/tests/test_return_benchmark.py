"""Unit tests for Return benchmark harness helpers and a mini end-to-end run."""

from __future__ import annotations

import json

import numpy as np

from visual_transition_benchmark import future_memory_benchmark as fmb
from visual_transition_benchmark.return_benchmark import (
    _aggregate_return_rows,
    build_coarse_arcade_setup,
    build_coarse_warehouse_setup,
    downsample_frame,
    main,
)
from zeromodel.video.arcade_policy.model import ShooterConfig


def test_downsample_halves_deterministically() -> None:
    frame = np.arange(16 * 28, dtype=np.uint8).reshape(16, 28)
    first = downsample_frame(frame)
    second = downsample_frame(frame)
    assert first.shape == (8, 14)
    assert np.array_equal(first, second)
    assert first.dtype == np.uint8
    assert int(first[0, 0]) == int(round((0 + 1 + 28 + 29) / 4))


def test_coarse_arcade_bands_partition_fields() -> None:
    setup = fmb._arcade_setup()
    records = fmb.generate_arcade_episodes(
        prefix="probe", episode_count=1, seed_offset=0, config=ShooterConfig()
    )
    coarse = build_coarse_arcade_setup(setup, records[0].frame_before)
    assert coarse.field_schema.width == 14
    assert coarse.field_schema.height == 8
    covered = [
        field_id
        for annotation in coarse.annotations
        for field_id in annotation.field_ids
    ]
    assert sorted(covered) == sorted(
        field.field_id for field in coarse.field_schema.fields
    )
    assert set(coarse.expectations_by_action) == set(setup.expectations_by_action)


def test_coarse_warehouse_schema_dimensions() -> None:
    setup = fmb._warehouse_setup()
    records = fmb.generate_warehouse_episodes(
        prefix="probe", episode_count=1, seed_offset=0, goal=(3, 3)
    )
    coarse = build_coarse_warehouse_setup(setup, records[0].frame_before)
    assert (coarse.field_schema.width, coarse.field_schema.height) == (15, 17)
    assert len(coarse.annotations) == 1


def test_mini_return_run_completes(tmp_path) -> None:
    out = tmp_path / "return"
    assert (
        main(
            [
                "--seeds",
                "0",
                "--train-episodes",
                "2",
                "--eval-episodes",
                "1",
                "--output-dir",
                str(out),
            ]
        )
        == 0
    )
    results = json.loads((out / "return-results.json").read_text(encoding="utf-8"))
    assert set(results["domains"]) == {"arcade", "warehouse"}
    for domain in results["domains"].values():
        assert "clean" in domain["splits"]
        arms = domain["splits"]["clean"]
        assert {"A gate-only", "B reobserve", "C fallback", "D combined"}.issubset(
            set(arms)
        )
        for name, cell in arms.items():
            if name == "regional":
                continue
            assert "return_net" in cell
            assert "halves" in cell


def _row(**overrides):
    row = {
        "episode_id": "ep",
        "step_number": 0,
        "quadrant": "top-left|MOVE_UP",
        "base_ok": False,
        "initial_ok": False,
        "initial_accepted": False,
        "final_ok": False,
        "accepted": False,
        "decided_by": "NONE",
        "disposition": None,
        "trigger": None,
        "invoked": False,
        "reobserved": False,
        "fallback_used": False,
        "support_restored": None,
        "fallback_accepted": None,
        "fallback_correct": None,
        "verdict": "confirmed",
        "matched": True,
        "mae": 0.01,
    }
    row.update(overrides)
    return row


def test_return_harm_counts_abstain_to_wrong_commit() -> None:
    """Frozen definition: abstain→wrong-commit is harm, not zero harm."""
    rows = [
        # Safe abstention converted into a wrong fallback commit.
        _row(
            initial_accepted=False,
            accepted=True,
            final_ok=False,
            decided_by="FALLBACK",
            disposition=None,
            trigger="CONTRADICTED",
            invoked=True,
            fallback_used=True,
            fallback_accepted=True,
            fallback_correct=False,
            verdict="future_projection_mismatch",
            matched=False,
        ),
        # Abstention recovered correctly.
        _row(
            initial_accepted=False,
            accepted=True,
            final_ok=True,
            decided_by="REOBSERVE",
            invoked=True,
            reobserved=True,
            support_restored=True,
        ),
        # Abstention left standing.
        _row(disposition="ABSTAIN"),
    ]
    metrics = _aggregate_return_rows(rows)
    assert metrics["return_gain"] == 1 / 3
    assert metrics["return_harm"] == 1 / 3
    assert metrics["return_net"] == 0.0
    assert metrics["coverage"] == 2 / 3


def test_return_harm_ignores_non_recovery_paths() -> None:
    rows = [
        _row(
            base_ok=True,
            initial_ok=True,
            initial_accepted=True,
            final_ok=True,
            accepted=True,
            decided_by="PRIMARY",
        ),
        _row(
            base_ok=True,
            initial_ok=True,
            initial_accepted=True,
            final_ok=False,
            accepted=True,
            decided_by="BASELINE",
        ),
    ]
    metrics = _aggregate_return_rows(rows)
    assert metrics["return_gain"] == 0.0
    assert metrics["return_harm"] == 0.0
    assert metrics["return_net"] == 0.0
