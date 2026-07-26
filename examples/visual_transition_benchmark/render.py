"""Human-inspectable diagnostic panels (section 7) and an HTML index."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visual_transition_benchmark.baselines import SystemOutput
from visual_transition_benchmark.dataset import TransitionRecord

SCALE = 8
PANEL_GAP = 6
TEXT_LINE_HEIGHT = 14
_FONT = ImageFont.load_default()


def _upscale(gray: np.ndarray) -> Image.Image:
    image = Image.fromarray(gray, mode="L").convert("RGB")
    return image.resize((image.width * SCALE, image.height * SCALE), Image.NEAREST)


def _diff_heatmap(before: np.ndarray, after: np.ndarray) -> Image.Image:
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16)).astype(np.uint8)
    rgb = np.zeros((*diff.shape, 3), dtype=np.uint8)
    rgb[..., 0] = diff  # red channel intensity = magnitude of change
    return Image.fromarray(rgb, mode="RGB").resize(
        (diff.shape[1] * SCALE, diff.shape[0] * SCALE), Image.NEAREST
    )


def _mask_overlay(base: np.ndarray, mask: np.ndarray, color: tuple) -> Image.Image:
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    overlay = np.array(color, dtype=np.float32)
    alpha = 0.55
    rgb[mask] = rgb[mask] * (1 - alpha) + overlay * alpha
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    return image.resize((base.shape[1] * SCALE, base.shape[0] * SCALE), Image.NEAREST)


def _labeled(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + TEXT_LINE_HEIGHT + 2), (24, 24, 24))
    canvas.paste(image, (0, TEXT_LINE_HEIGHT + 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((2, 1), label, fill=(255, 255, 255), font=_FONT)
    return canvas


def _hstack(images: Sequence[Image.Image], gap: int = PANEL_GAP) -> Image.Image:
    height = max(image.height for image in images)
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), (16, 16, 16))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    return canvas


def render_transition_panel(
    record: TransitionRecord,
    privileged_output: SystemOutput,
    zeromodel_output: SystemOutput,
    *,
    output_path: Path,
) -> Path:
    panels = [
        _labeled(_upscale(record.frame_before), "1. frame_before"),
        _labeled(_upscale(record.frame_after), "2. frame_after"),
        _labeled(_diff_heatmap(record.frame_before, record.frame_after), "3. raw pixel diff"),
        _labeled(
            _mask_overlay(record.frame_after, privileged_output.predicted_region_mask, (0, 220, 0)),
            "4. privileged ground truth",
        ),
        _labeled(
            _mask_overlay(record.frame_after, zeromodel_output.predicted_region_mask, (0, 200, 255)),
            "5. ZeroModel predicted",
        ),
    ]
    strip = _hstack(panels)

    lines = [
        f"transition_id: {record.transition_id}    category: {record.category}    "
        f"action: {record.action}    fault_type: {record.fault_type}",
        f"expected_changed_components: {list(record.expected_changed_components)}",
        f"observed_changed_components: {list(record.observed_changed_components)}",
        f"ZeroModel: predicted={list(zeromodel_output.predicted_components)}  "
        f"missing={list(zeromodel_output.missing_components)}  "
        f"unexpected={list(zeromodel_output.unexpected_components)}",
        f"notes: {record.notes}",
    ]
    text_height = TEXT_LINE_HEIGHT * len(lines) + 8
    canvas = Image.new("RGB", (strip.width, strip.height + text_height), (16, 16, 16))
    canvas.paste(strip, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        draw.text((4, strip.height + 4 + index * TEXT_LINE_HEIGHT), line, fill=(230, 230, 230), font=_FONT)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return output_path


def build_html_index(
    rows: Iterable[dict],
    *,
    output_path: Path,
    title: str = "Visual Transition Debugging Benchmark",
) -> Path:
    """rows: dicts with transition_id, category, fault_type, is_faulty, verdict, artifact_path (relative)."""

    def esc(value: object) -> str:
        return html.escape(str(value))

    groups = {
        "failures (nonconformant/attention_required ZeroModel status)": [],
        "ZeroModel-only successes": [],
        "pixel-baseline-only successes": [],
        "false positives (ZeroModel flagged, nothing wrong)": [],
        "false negatives (ZeroModel silent, something was wrong)": [],
        "all rendered transitions": [],
    }
    for row in rows:
        groups["all rendered transitions"].append(row)
        if row.get("zeromodel_status") in {"nonconformant", "attention_required"}:
            groups["failures (nonconformant/attention_required ZeroModel status)"].append(row)
        if row.get("verdict") == "better":
            groups["ZeroModel-only successes"].append(row)
        elif row.get("verdict") == "worse":
            groups["pixel-baseline-only successes"].append(row)
        if row.get("false_positive"):
            groups["false positives (ZeroModel flagged, nothing wrong)"].append(row)
        if row.get("false_negative"):
            groups["false negatives (ZeroModel silent, something was wrong)"].append(row)

    parts = [f"<html><head><meta charset='utf-8'><title>{esc(title)}</title></head><body>"]
    parts.append(f"<h1>{esc(title)}</h1>")
    for group_name, group_rows in groups.items():
        parts.append(f"<h2>{esc(group_name)} ({len(group_rows)})</h2><ul>")
        for row in group_rows:
            parts.append(
                "<li><a href='%s'>%s</a> -- %s / fault=%s / verdict=%s</li>"
                % (
                    esc(row["artifact_path"]),
                    esc(row["transition_id"]),
                    esc(row["category"]),
                    esc(row.get("fault_type")),
                    esc(row.get("verdict")),
                )
            )
        parts.append("</ul>")
    parts.append("</body></html>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path
