"""Regenerate the ZeroModel "signs, not directions" blog assets.

The script compiles the bounded arcade-shooter policy using the public ZeroModel
API, runs the complete default episode through ``VPMPolicyLookup``, and writes:

- ``zero_policy.vpm``: lossless policy artifact bundle
- ``zero_policy_results.json``: canonical run summary and decision trace
- ``zero_policy_vpm.png``: full 112 x 4 policy surface
- ``zero_money_shot.png``: one annotated FIRE decision
- ``zero_replay.gif``: the complete decision replay

Run from the repository root:

    python -m pip install matplotlib pillow
    python examples/render_signs_demo.py --output-dir docs/assets/signs-demo

The image formats are presentation outputs. The canonical reproducibility targets
are the VPM artifact identity and JSON trace; raster output can vary slightly with
font and matplotlib versions.
"""
from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from PIL import Image
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise SystemExit(
        "This example requires matplotlib and Pillow. Install them with: "
        "python -m pip install matplotlib pillow"
    ) from exc

from examples.arcade_shooter_policy import (  # noqa: E402
    ACTIONS,
    ShooterConfig,
    TinyArcadeShooter,
    compile_policy_artifact,
    random_baseline_average,
)
from zeromodel.core.bundle import to_bundle
from zeromodel.core.policy_lookup import VPMPolicyLookup

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "assets" / "signs-demo"
MOVEMENT_ACTIONS = frozenset({"LEFT", "RIGHT", "STAY"})
ACTION_MARKERS = {
    "LEFT": "<",
    "RIGHT": ">",
    "STAY": "o",
    "FIRE": "*",
}


def _run_trace(config: ShooterConfig) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    artifact = compile_policy_artifact(config)
    reader = VPMPolicyLookup(artifact, action_metric_ids=ACTIONS)
    game = TinyArcadeShooter(config)
    trace: list[dict[str, Any]] = []

    while not game.done:
        before = game.snapshot()
        decision = reader.read(game.row_id())
        game.step(decision.action)
        trace.append(
            {
                **before,
                **decision.to_dict(),
                "after_tank_x": game.tank_x,
                "after_cooldown": game.cooldown,
                "after_score": game.score,
                "after_remaining_aliens": list(game.aliens),
            }
        )

    summary = {
        "artifact_id": artifact.artifact_id,
        "state_count": len(artifact.source.row_ids),
        "action_cell_count": int(artifact.source.values.size),
        "policy_score": game.score,
        "policy_cleared": game.cleared,
        "policy_steps": game.steps,
        "random_average_score_10_seeds": random_baseline_average(config),
        "normalization_note": (
            "Brightness is normalized independently within each action column. "
            "In this demo every action column already spans 0..1, so the rendered "
            "intensities preserve the raw values."
        ),
        "trace": trace,
    }
    return artifact, trace, summary


def _draw_state(ax: Any, frame: dict[str, Any], width: int) -> None:
    ax.set_title("1  CURRENT STATE", loc="left", fontweight="bold", pad=12)
    ax.set_xlim(-0.6, width - 0.4)
    ax.set_ylim(-1.2, 1.25)
    ax.set_xticks(range(width))
    ax.set_yticks([])
    ax.set_xlabel("screen column")
    ax.hlines(0, 0, width - 1, linewidth=1.5)

    if frame["target_x"] is not None:
        ax.plot([frame["target_x"]], [0.46], marker="*", markersize=22, linewidth=0)
        ax.text(frame["target_x"], 0.85, "TARGET", ha="center", fontweight="bold")

    ax.plot([frame["tank_x"]], [-0.46], marker="^", markersize=18, linewidth=0)
    ax.text(frame["tank_x"], -0.88, "TANK", ha="center", fontweight="bold")
    ax.text(
        0.03,
        0.97,
        f"step {frame['step']}   score {frame['score']}/4   cooldown {frame['cooldown']}",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
    )
    ax.text(
        0.5,
        -0.17,
        frame["row_id"],
        transform=ax.transAxes,
        ha="center",
        family="monospace",
        fontsize=9,
        fontweight="bold",
    )


def _draw_vpm(ax: Any, artifact: Any, frame: dict[str, Any]) -> None:
    ax.set_title("2  ADDRESS THE VPM", loc="left", fontweight="bold", pad=12)
    ax.imshow(artifact.normalized_values, aspect="auto", interpolation="nearest", cmap="gray")
    ax.set_xticks(range(len(ACTIONS)), ACTIONS, rotation=35, ha="right")
    ax.set_ylabel(f"{len(artifact.source.row_ids)} state rows")
    ax.axhline(frame["view_row"], linewidth=2.5)
    ax.axvline(frame["view_column"], linewidth=2.5)
    ax.plot(
        [frame["view_column"]],
        [frame["view_row"]],
        marker="s",
        markersize=15,
        markerfacecolor="none",
        markeredgewidth=3,
        linewidth=0,
    )
    ax.set_ylim(len(artifact.source.row_ids) - 0.5, -0.5)


def _draw_action_values(ax: Any, frame: dict[str, Any]) -> None:
    ax.set_title("3  READ THE SIGN", loc="left", fontweight="bold", pad=12)
    values = [frame["candidates"][action] for action in ACTIONS]
    bars = ax.barh(ACTIONS, values)
    ax.invert_yaxis()  # Match the VPM's LEFT -> RIGHT -> STAY -> FIRE ordering.
    ax.set_xlim(0, max(1.05, max(values) * 1.05))
    ax.set_xlabel("raw action value")
    winner = frame["source_metric_index"]
    bars[winner].set_linewidth(3)
    bars[winner].set_hatch("//")


def _draw_proof(ax: Any, frame: dict[str, Any]) -> None:
    ax.set_title("4  ACTION + PROOF", loc="left", fontweight="bold", pad=12)
    ax.axis("off")
    ax.text(0.5, 0.82, frame["action"], ha="center", fontsize=30, fontweight="bold")
    ax.text(
        0.04,
        0.63,
        (
            f"selected value  {frame['value']:.1f}\n"
            f"source row      {frame['source_row_index']}\n"
            f"source action   {frame['metric_id']}\n"
            f"view row        {frame['view_row']}\n"
            f"view column     {frame['view_column']}\n\n"
            f"artifact id\n{frame['artifact_id'][:24]}..."
        ),
        family="monospace",
        fontsize=10,
        va="top",
    )


def _draw_trace(ax: Any, trace: Sequence[dict[str, Any]], current_step: int) -> None:
    ax.set_title("THE SAME ARTIFACT DRIVES THE COMPLETE TRACE", loc="left", fontweight="bold")
    ax.set_xlim(-0.6, len(trace) - 0.4)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([])
    ax.set_xticks(range(len(trace)))
    ax.set_xticklabels([str(item["step"]) for item in trace], fontsize=7)
    ax.set_xlabel("step")

    for index, item in enumerate(trace):
        action = item["action"]
        ax.plot(
            [index],
            [0],
            marker=ACTION_MARKERS[action],
            markersize=15 if action == "FIRE" else 9,
            color="black" if action in MOVEMENT_ACTIONS else "tab:red",
            linewidth=0,
        )
    ax.axvline(current_step, linewidth=2.5, color="black")


def _render_frame(
    artifact: Any,
    trace: Sequence[dict[str, Any]],
    frame: dict[str, Any],
    *,
    width: int,
    dpi: int,
) -> Image.Image:
    fig = plt.figure(figsize=(14, 7.875))
    grid = GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=(5.2, 1.0),
        width_ratios=(1.35, 0.68, 1.0, 0.88),
    )
    fig.subplots_adjust(left=0.045, right=0.975, bottom=0.12, top=0.79, wspace=0.42, hspace=0.5)
    fig.suptitle("SIGNS, NOT DIRECTIONS", fontsize=25, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.90,
        "game state -> VPM row -> winning action cell -> action + proof",
        ha="center",
        fontsize=12,
    )

    _draw_state(fig.add_subplot(grid[0, 0]), frame, width)
    _draw_vpm(fig.add_subplot(grid[0, 1]), artifact, frame)
    _draw_action_values(fig.add_subplot(grid[0, 2]), frame)
    _draw_proof(fig.add_subplot(grid[0, 3]), frame)
    _draw_trace(fig.add_subplot(grid[1, :]), trace, frame["step"])

    fig.text(
        0.5,
        0.045,
        (
            f"1 compiled VPM  |  {len(artifact.source.row_ids)} states  |  "
            f"{artifact.source.values.size} action cells  |  {len(trace)} decisions  |  wave cleared"
        ),
        ha="center",
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.018,
        (
            "Brightness is normalized per action column; all four columns span 0..1 "
            "in this demo, so rendered intensity preserves raw values."
        ),
        ha="center",
        fontsize=8,
    )

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _write_vpm_image(artifact: Any, path: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(5, 13), constrained_layout=True)
    ax.imshow(artifact.normalized_values, aspect="auto", interpolation="nearest", cmap="gray")
    ax.set_title(
        "ZeroModel Arcade-Shooter VPM\n"
        f"{len(artifact.source.row_ids)} states x {len(artifact.source.metric_ids)} actions"
    )
    ax.set_xticks(range(len(ACTIONS)), ACTIONS, rotation=35, ha="right")
    ax.set_ylabel("discretized state row")
    fig.text(0.5, 0.01, f"artifact {artifact.artifact_id[:24]}...", ha="center", family="monospace")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_gif(
    artifact: Any,
    trace: Sequence[dict[str, Any]],
    path: Path,
    *,
    width: int,
    dpi: int,
    duration_ms: int,
) -> None:
    frames = [
        _render_frame(artifact, trace, frame, width=width, dpi=dpi).convert(
            "P", palette=Image.Palette.ADAPTIVE
        )
        for frame in trace
    ]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--frame-duration-ms", type=int, default=480)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ShooterConfig()
    artifact, trace, summary = _run_trace(config)
    if not trace:
        raise RuntimeError("The shooter produced an empty trace")

    bundle_path = to_bundle(artifact, output_dir / "zero_policy.vpm")
    results_path = output_dir / "zero_policy_results.json"
    results_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    vpm_path = output_dir / "zero_policy_vpm.png"
    _write_vpm_image(artifact, vpm_path, dpi=args.dpi)

    fire_frame = next((frame for frame in trace if frame["action"] == "FIRE"), trace[0])
    money_path = output_dir / "zero_money_shot.png"
    _render_frame(artifact, trace, fire_frame, width=config.width, dpi=args.dpi).save(money_path)

    replay_path = output_dir / "zero_replay.gif"
    _write_gif(
        artifact,
        trace,
        replay_path,
        width=config.width,
        dpi=max(72, args.dpi - 20),
        duration_ms=args.frame_duration_ms,
    )

    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "policy_cleared": summary["policy_cleared"],
                "policy_score": summary["policy_score"],
                "policy_steps": summary["policy_steps"],
                "outputs": {
                    "bundle": str(bundle_path),
                    "results": str(results_path),
                    "vpm": str(vpm_path),
                    "money_shot": str(money_path),
                    "replay": str(replay_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
