"""Render deterministic fixed-camera status-panel fixtures.

The PNGs are canonical panel fixtures, not real camera captures. They use
explicit visual channels so a deterministic compiler can read evidence without
OCR: anchors, LED colour, marker positions, bar fill, geometry, and beacon fill.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
STATES_PATH = ROOT / "states.json"
POLICY_PATH = ROOT / "policy.json"
REGIONS_PATH = ROOT / "regions.json"
CALIBRATION_PATH = ROOT / "calibration.json"
CANONICAL = ROOT / "canonical"
CAPTURES = ROOT / "captures"
MANIFESTS = ROOT / "manifests"
SESSIONS = ("development", "calibration", "evaluation")
FIELDS = ("power", "mode", "temperature", "door", "alarm")
PANEL_SIZE = (720, 480)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE_FONT = _font(34)
LABEL_FONT = _font(20)
SMALL_FONT = _font(15)


REGIONS: dict[str, object] = {
    "panel_layout_id": "fixed-camera-status-panel-layout/v2",
    "width": PANEL_SIZE[0],
    "height": PANEL_SIZE[1],
    "anchors": {
        "top_left": [48, 48, 74, 74],
        "top_right": [646, 48, 672, 74],
        "bottom_left": [48, 406, 74, 432],
        "bottom_right": [646, 406, 672, 432],
    },
    "fields": {
        "power": {
            "region_id": "power-led",
            "allowed_values": ["green", "off", "red"],
            "led": [104, 143, 148, 187],
        },
        "mode": {
            "region_id": "mode-position",
            "allowed_values": ["auto", "maintenance", "manual"],
            "boxes": {
                "auto": [260, 134, 356, 190],
                "manual": [378, 134, 474, 190],
                "maintenance": [496, 134, 622, 190],
            },
        },
        "temperature": {
            "region_id": "temperature-bar",
            "allowed_values": ["critical", "elevated", "normal"],
            "segments": {
                "normal": [260, 226, 350, 260],
                "elevated": [366, 226, 456, 260],
                "critical": [472, 226, 562, 260],
            },
        },
        "door": {
            "region_id": "door-geometry",
            "allowed_values": ["closed", "open"],
            "closed_region": [270, 303, 340, 366],
            "open_region": [450, 303, 544, 366],
        },
        "alarm": {
            "region_id": "alarm-diamond",
            "allowed_values": ["active", "inactive"],
            "diamond": [592, 314, 642, 364],
        },
    },
}

CALIBRATION: dict[str, object] = {
    "calibration_id": "fixed-camera-status-panel-calibration/v1",
    "minimum_anchor_dark_fraction": 0.72,
    "minimum_valid_brightness": 30.0,
    "maximum_valid_brightness": 245.0,
    "minimum_colour_saturation": 0.28,
    "minimum_marker_occupancy": 0.18,
    "minimum_marker_margin": 0.14,
    "minimum_bar_occupancy": 0.35,
    "minimum_shape_occupancy": 0.12,
    "minimum_shape_margin": 0.08,
    "maximum_glare_fraction": 0.18,
}


def _write_static_json() -> None:
    REGIONS_PATH.write_text(json.dumps(REGIONS, indent=2, sort_keys=True) + "\n")
    CALIBRATION_PATH.write_text(
        json.dumps(CALIBRATION, indent=2, sort_keys=True) + "\n"
    )


def _digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _load_states() -> list[dict[str, str]]:
    return json.loads(STATES_PATH.read_text(encoding="utf-8"))


def _rect(region: object) -> tuple[int, int, int, int]:
    values = tuple(int(value) for value in region)  # type: ignore[arg-type]
    return values  # type: ignore[return-value]


def _center(region: object) -> tuple[int, int]:
    x0, y0, x1, y1 = _rect(region)
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def _led_colour(value: str) -> tuple[int, int, int]:
    if value == "green":
        return (28, 205, 88)
    if value == "red":
        return (232, 54, 48)
    return (42, 47, 54)


def _draw_anchor(draw: ImageDraw.ImageDraw, region: object) -> None:
    draw.rectangle(_rect(region), fill=(18, 22, 27))


def _draw_label(draw: ImageDraw.ImageDraw, label: str, y: int) -> None:
    draw.text((92, y), label, fill=(43, 51, 59), font=LABEL_FONT)


def _draw_mode(draw: ImageDraw.ImageDraw, mode: str) -> None:
    boxes = REGIONS["fields"]["mode"]["boxes"]  # type: ignore[index]
    for value, box in boxes.items():  # type: ignore[union-attr]
        rect = _rect(box)
        draw.rounded_rectangle(rect, radius=5, outline=(130, 140, 150), width=2)
        draw.text(
            (rect[0] + 8, rect[1] + 8),
            value.upper(),
            fill=(50, 58, 66),
            font=SMALL_FONT,
        )
        if value == mode:
            cx, cy = _center(box)
            draw.rounded_rectangle(
                (cx - 28, cy + 1, cx + 28, cy + 25),
                radius=8,
                fill=(22, 117, 240),
            )


def _draw_temperature(draw: ImageDraw.ImageDraw, temperature: str) -> None:
    segments = REGIONS["fields"]["temperature"]["segments"]  # type: ignore[index]
    active_count = {"normal": 1, "elevated": 2, "critical": 3}[temperature]
    for index, value in enumerate(("normal", "elevated", "critical"), start=1):
        rect = _rect(segments[value])  # type: ignore[index]
        active = index <= active_count
        fill = (238, 242, 244)
        if active:
            fill = (48, 180, 94) if index == 1 else (235, 169, 55)
            if index == 3:
                fill = (229, 60, 54)
        draw.rounded_rectangle(
            rect, radius=4, fill=fill, outline=(132, 142, 152), width=2
        )


def _draw_door(draw: ImageDraw.ImageDraw, door: str) -> None:
    closed = _rect(REGIONS["fields"]["door"]["closed_region"])  # type: ignore[index]
    open_ = _rect(REGIONS["fields"]["door"]["open_region"])  # type: ignore[index]
    draw.rectangle(closed, outline=(126, 136, 146), width=2)
    draw.polygon(
        [(open_[0], open_[3]), (open_[2], open_[1]), (open_[2], open_[3])],
        outline=(126, 136, 146),
    )
    if door == "closed":
        draw.rectangle(
            (closed[0] + 18, closed[1] + 6, closed[2] - 18, closed[3] - 6),
            fill=(42, 120, 210),
        )
    else:
        draw.polygon(
            [
                (open_[0] + 10, open_[3] - 4),
                (open_[2] - 12, open_[1] + 6),
                (open_[2] - 24, open_[1] + 22),
                (open_[0] + 22, open_[3] - 4),
            ],
            fill=(42, 120, 210),
        )


def _draw_alarm(draw: ImageDraw.ImageDraw, alarm: str) -> None:
    x0, y0, x1, y1 = _rect(REGIONS["fields"]["alarm"]["diamond"])  # type: ignore[index]
    points = [
        ((x0 + x1) // 2, y0),
        (x1, (y0 + y1) // 2),
        ((x0 + x1) // 2, y1),
        (x0, (y0 + y1) // 2),
    ]
    fill = (223, 50, 50) if alarm == "active" else (45, 51, 58)
    draw.polygon(points, fill=fill, outline=(120, 35, 35))
    if alarm == "inactive":
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        draw.polygon(
            [(cx, y0 + 10), (x1 - 10, cy), (cx, y1 - 10), (x0 + 10, cy)],
            fill=(238, 242, 244),
        )


def render_panel(state: Mapping[str, str], path: Path) -> None:
    image = Image.new("RGB", PANEL_SIZE, (245, 247, 248))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (38, 38, 682, 442),
        radius=8,
        fill=(236, 240, 242),
        outline=(22, 27, 32),
        width=4,
    )
    for region in REGIONS["anchors"].values():  # type: ignore[union-attr]
        _draw_anchor(draw, region)
    draw.text((92, 68), "MACHINE STATUS", fill=(20, 26, 32), font=TITLE_FONT)
    draw.line((92, 112, 628, 112), fill=(125, 135, 145), width=2)

    _draw_label(draw, "POWER", 148)
    x0, y0, x1, y1 = _rect(REGIONS["fields"]["power"]["led"])  # type: ignore[index]
    draw.ellipse(
        (x0, y0, x1, y1),
        fill=_led_colour(state["power"]),
        outline=(35, 40, 45),
        width=2,
    )
    draw.ellipse((x0 + 10, y0 + 8, x0 + 22, y0 + 20), fill=(255, 255, 255))

    _draw_label(draw, "MODE", 206)
    _draw_mode(draw, state["mode"])

    _draw_label(draw, "TEMPERATURE", 236)
    _draw_temperature(draw, state["temperature"])

    _draw_label(draw, "DOOR", 322)
    _draw_door(draw, state["door"])

    _draw_label(draw, "ALARM", 322)
    _draw_alarm(draw, state["alarm"])

    draw.text(
        (92, 397),
        f"{state['state_id'].upper()}  ACTION {state['action']}",
        fill=(82, 91, 101),
        font=SMALL_FONT,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


def _manifest_record(
    *, session: str, state: Mapping[str, str], image_path: Path
) -> dict[str, object]:
    relative = image_path.relative_to(ROOT).as_posix()
    return {
        "image": relative,
        "image_digest": _digest(image_path),
        "state_id": state["state_id"],
        "capture_session": session,
        "condition": "canonical-panel-fixture",
        "ground_truth": {field: state[field] for field in FIELDS},
        "expected_action": state["action"],
    }


def render_dataset(
    *, sessions: Iterable[str] = SESSIONS
) -> dict[str, list[dict[str, object]]]:
    _write_static_json()
    states = _load_states()
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    CANONICAL.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[dict[str, object]]] = {}
    for state in states:
        render_panel(state, CANONICAL / f"{state['state_id']}.png")
    for session in sessions:
        records: list[dict[str, object]] = []
        for state in states:
            image_path = (
                CAPTURES
                / session
                / state["state_id"]
                / "canonical-panel-fixture-01.png"
            )
            render_panel(state, image_path)
            records.append(
                _manifest_record(session=session, state=state, image_path=image_path)
            )
        manifest_path = MANIFESTS / f"{session}.jsonl"
        manifest_path.write_text(
            "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        manifests[session] = records
    return manifests


def main() -> None:
    manifests = render_dataset()
    count = sum(len(records) for records in manifests.values())
    print(f"rendered {count} canonical panel fixtures under {CAPTURES}")


if __name__ == "__main__":
    main()
