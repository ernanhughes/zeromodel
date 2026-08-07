"""Render deterministic fixed-camera status-panel fixtures.

The rendered PNGs are canonical computer screenshots, not real camera captures.
They provide exact state labels for the first state-claim compiler tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
STATES_PATH = ROOT / "states.json"
CAPTURES = ROOT / "captures"
MANIFESTS = ROOT / "manifests"
SESSIONS = ("development", "calibration", "evaluation")
FIELDS = ("power", "mode", "temperature", "door", "alarm")
PANEL_SIZE = (720, 480)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE_FONT = _font(36)
LABEL_FONT = _font(25)
VALUE_FONT = _font(27)


def _digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _load_states() -> list[dict[str, str]]:
    return json.loads(STATES_PATH.read_text(encoding="utf-8"))


def _led_colour(value: str) -> tuple[int, int, int]:
    if value == "green":
        return (31, 180, 90)
    if value == "red":
        return (219, 65, 58)
    if value == "off":
        return (72, 78, 86)
    if value == "active":
        return (229, 76, 65)
    return (88, 96, 105)


def render_panel(state: Mapping[str, str], path: Path) -> None:
    image = Image.new("RGB", PANEL_SIZE, (244, 247, 248))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((28, 26, 692, 454), radius=10, fill=(21, 26, 32))
    draw.rounded_rectangle((48, 48, 672, 434), radius=6, fill=(236, 240, 241))
    draw.text((80, 72), "MACHINE STATUS", fill=(18, 24, 31), font=TITLE_FONT)
    draw.line((80, 122, 640, 122), fill=(120, 130, 140), width=2)

    rows = (
        ("POWER", state["power"]),
        ("MODE", state["mode"]),
        ("TEMPERATURE", state["temperature"]),
        ("DOOR", state["door"]),
        ("ALARM", state["alarm"]),
    )
    y = 160
    for label, value in rows:
        draw.text((92, y), label, fill=(47, 56, 65), font=LABEL_FONT)
        if label in {"POWER", "ALARM"}:
            draw.ellipse((382, y + 2, 416, y + 36), fill=_led_colour(value))
            draw.ellipse((390, y + 8, 402, y + 20), fill=(255, 255, 255))
            draw.text((438, y - 1), value.upper(), fill=(18, 24, 31), font=VALUE_FONT)
        else:
            draw.rounded_rectangle(
                (376, y - 8, 622, y + 42),
                radius=5,
                fill=(250, 252, 252),
                outline=(165, 174, 184),
            )
            draw.text((398, y - 1), value.upper(), fill=(18, 24, 31), font=VALUE_FONT)
        y += 54

    draw.text(
        (80, 394),
        f"{state['state_id'].upper()}  ACTION {state['action']}",
        fill=(82, 91, 101),
        font=_font(18),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


def _manifest_record(
    *,
    session: str,
    state: Mapping[str, str],
    image_path: Path,
) -> dict[str, object]:
    relative = image_path.relative_to(ROOT).as_posix()
    return {
        "image": relative,
        "image_digest": _digest(image_path),
        "state_id": state["state_id"],
        "capture_session": session,
        "condition": "canonical-render",
        "ground_truth": {field: state[field] for field in FIELDS},
        "expected_action": state["action"],
    }


def render_dataset(
    *,
    sessions: Iterable[str] = SESSIONS,
) -> dict[str, list[dict[str, object]]]:
    states = _load_states()
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[dict[str, object]]] = {}
    for session in sessions:
        records: list[dict[str, object]] = []
        for state in states:
            image_path = (
                CAPTURES
                / session
                / state["state_id"]
                / "canonical-render-01.png"
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
    print(f"rendered {count} canonical panel images under {CAPTURES}")


if __name__ == "__main__":
    main()
