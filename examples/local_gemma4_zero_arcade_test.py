#!/usr/bin/env python3
"""Local-only Gemma 4 -> ZeroModel arcade perception smoke test.

Run from a ZeroModel checkout with Ollama serving a vision-capable Gemma 4 model.
The VLM reads state only; an independently compiled VPM policy chooses actions.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Protocol

try:
    from PIL import Image, ImageDraw
except ImportError as exc:
    raise SystemExit("Install Pillow: python -m pip install pillow") from exc

try:
    from zeromodel.core import LayoutRecipe, ScoreTable, VPMPolicyLookup, build_vpm
except ImportError as exc:
    raise SystemExit(
        "ZeroModel is not importable. From the repository root run:\n"
        "  python -m pip install -r requirements-dev.txt"
    ) from exc

WIDTH = 7
ACTIONS = ("LEFT", "RIGHT", "STAY", "FIRE")
SCHEMA_VERSION = "zeromodel-local-qwen3.5-arcade-state/v1"

STATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tank_column": {"type": "integer", "minimum": 0, "maximum": 6},
        "target_present": {"type": "boolean"},
        "target_column": {
            "type": "integer",
            "minimum": -1,
            "maximum": 6,
            "description": "-1 exactly when no target is visible",
        },
        "cooldown": {
            "type": "integer",
            "enum": [0, 1],
            "description": "0 means ready; 1 means blocked",
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": [
        "tank_column",
        "target_present",
        "target_column",
        "cooldown",
        "confidence",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class ArcadeState:
    tank_column: int
    target_column: int | None
    cooldown: int

    def __post_init__(self) -> None:
        if not 0 <= self.tank_column < WIDTH:
            raise ValueError("tank_column out of range")
        if self.target_column is not None and not 0 <= self.target_column < WIDTH:
            raise ValueError("target_column out of range")
        if self.cooldown not in (0, 1):
            raise ValueError("cooldown must be 0 or 1")

    @property
    def target_present(self) -> bool:
        return self.target_column is not None

    @property
    def row_id(self) -> str:
        target = "none" if self.target_column is None else str(self.target_column)
        return f"tank={self.tank_column}|target={target}|cooldown={self.cooldown}"

    def payload(self) -> dict[str, Any]:
        return {
            "tank_column": self.tank_column,
            "target_present": self.target_present,
            "target_column": -1 if self.target_column is None else self.target_column,
            "cooldown": self.cooldown,
            "row_id": self.row_id,
        }


@dataclass(frozen=True, slots=True)
class Prediction:
    tank_column: int
    target_present: bool
    target_column: int | None
    cooldown: int
    confidence: float

    @classmethod
    def parse(cls, value: dict[str, Any]) -> "Prediction":
        required = {
            "tank_column",
            "target_present",
            "target_column",
            "cooldown",
            "confidence",
        }
        if set(value) != required:
            raise ValueError(
                f"response fields must be exactly {sorted(required)}; got {sorted(value)}"
            )
        tank = strict_int(value["tank_column"], "tank_column")
        present = value["target_present"]
        target_raw = strict_int(value["target_column"], "target_column")
        cooldown = strict_int(value["cooldown"], "cooldown")
        confidence = strict_number(value["confidence"], "confidence")
        if not isinstance(present, bool):
            raise ValueError("target_present must be boolean")
        if tank not in range(WIDTH):
            raise ValueError("tank_column out of range")
        if target_raw not in (-1, *range(WIDTH)):
            raise ValueError("target_column must be -1 or 0..6")
        if cooldown not in (0, 1):
            raise ValueError("cooldown must be 0 or 1")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if present != (target_raw != -1):
            raise ValueError("target_present and target_column disagree")
        return cls(tank, present, None if target_raw == -1 else target_raw, cooldown, confidence)

    @property
    def row_id(self) -> str:
        target = "none" if self.target_column is None else str(self.target_column)
        return f"tank={self.tank_column}|target={target}|cooldown={self.cooldown}"

    def payload(self) -> dict[str, Any]:
        return {
            "tank_column": self.tank_column,
            "target_present": self.target_present,
            "target_column": -1 if self.target_column is None else self.target_column,
            "cooldown": self.cooldown,
            "confidence": self.confidence,
            "row_id": self.row_id,
        }


@dataclass(frozen=True, slots=True)
class ProviderReply:
    raw_text: str
    parsed: dict[str, Any]
    duration_ms: float
    metadata: dict[str, Any]


class Provider(Protocol):
    def predict(self, image: bytes, render_mode: str, truth: ArcadeState) -> ProviderReply: ...


class FakeProvider:
    def predict(self, image: bytes, render_mode: str, truth: ArcadeState) -> ProviderReply:
        del image, render_mode
        payload = {
            "tank_column": truth.tank_column,
            "target_present": truth.target_present,
            "target_column": -1 if truth.target_column is None else truth.target_column,
            "cooldown": truth.cooldown,
            "confidence": 1.0,
        }
        return ProviderReply(json.dumps(payload), payload, 0.0, {"backend": "fake"})


class OllamaProvider:
    def __init__(self, base_url: str, model: str, timeout: float, seed: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.seed = seed
        self.model_record = self._find_model()

    def _json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Ollama returned non-object JSON")
        return result

    def _find_model(self) -> dict[str, Any]:
        models = self._json("GET", "/api/tags").get("models", [])
        available: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "")
            if name:
                available.append(name)
            if name == self.model or str(item.get("model", "")) == self.model:
                return item
        raise RuntimeError(
            f"Ollama model {self.model!r} not found. Installed: "
            + (", ".join(available) or "<none>")
        )

    def predict(self, image: bytes, render_mode: str, truth: ArcadeState) -> ProviderReply:
        del truth
        payload = {
            "model": self.model,
            "prompt": prompt_for(render_mode),
            "images": [base64.b64encode(image).decode("ascii")],
            "format": STATE_SCHEMA,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.0, "seed": self.seed, "num_predict": 128},
        }
        started = time.perf_counter()
        response = self._json("POST", "/api/generate", payload)
        wall_ms = (time.perf_counter() - started) * 1000
        raw = response.get("response")
        if not isinstance(raw, str):
            raise RuntimeError("Ollama response has no string 'response'")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("model response must decode to a JSON object")
        duration = response.get("total_duration")
        duration_ms = float(duration) / 1_000_000 if isinstance(duration, (int, float)) else wall_ms
        return ProviderReply(
            raw,
            parsed,
            duration_ms,
            {
                "backend": "ollama",
                "model": self.model,
                "model_record": self.model_record,
                "wall_ms": wall_ms,
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
                "done_reason": response.get("done_reason"),
            },
        )


def strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be integer")
    return value


def strict_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def prompt_for(render_mode: str) -> str:
    lanes = (
        "The lanes are visibly labelled 0 through 6 from left to right."
        if render_mode == "labelled"
        else "The image has seven equal lanes; count 0 at far left through 6 at far right."
    )
    return f"""
You are a perception-only component in a bounded arcade experiment.
Do not choose LEFT, RIGHT, STAY, or FIRE. Do not explain strategy.

Read exactly these visible variables:
- tank_column: lane containing the cyan tank at the bottom
- target_present: whether a magenta alien is visible near the top
- target_column: alien lane, or -1 if no alien is visible
- cooldown: 0 when the top-left indicator is green/READY; 1 when red/BLOCKED
- confidence: confidence in the complete state from 0.0 to 1.0

{lanes}
Return only the JSON object required by the supplied schema.
""".strip()


def action_values(state: ArcadeState) -> tuple[float, float, float, float]:
    target = state.target_column
    if target is None:
        return (0.0, 0.0, 1.0, 0.0)
    if state.cooldown == 0 and state.tank_column == target:
        return (0.0, 0.0, 0.0, 1.0)
    if state.tank_column > target:
        return (1.0, 0.0, 0.1, 0.0)
    if state.tank_column < target:
        return (0.0, 1.0, 0.1, 0.0)
    return (0.0, 0.0, 1.0, 0.0)


def all_states() -> list[ArcadeState]:
    return [
        ArcadeState(tank, target, cooldown)
        for tank in range(WIDTH)
        for target in (None, *range(WIDTH))
        for cooldown in (0, 1)
    ]


def smoke_states() -> list[ArcadeState]:
    return [
        ArcadeState(0, None, 0),
        ArcadeState(6, None, 1),
        ArcadeState(0, 0, 0),
        ArcadeState(3, 3, 1),
        ArcadeState(6, 0, 0),
        ArcadeState(0, 6, 0),
        ArcadeState(3, 1, 1),
        ArcadeState(2, 5, 1),
    ]


def policy_reader() -> tuple[VPMPolicyLookup, str]:
    states = all_states()
    table = ScoreTable(
        values=[action_values(state) for state in states],
        row_ids=[state.row_id for state in states],
        metric_ids=ACTIONS,
        metadata={"kind": "local_qwen3.5_arcade_policy", "schema": SCHEMA_VERSION},
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "local-qwen3.5-source-order",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    artifact = build_vpm(
        table,
        recipe,
        provenance={"kind": "local_only_experiment", "provider": "local_vlm"},
    )
    return VPMPolicyLookup(artifact, action_metric_ids=ACTIONS), artifact.artifact_id


def render(state: ArcadeState, mode: str, width: int = 896, height: int = 512) -> Image.Image:
    bg, grid = (13, 18, 30), (53, 66, 92)
    cyan, magenta = (65, 220, 255), (255, 74, 171)
    green, red, white = (70, 225, 125), (242, 74, 74), (224, 232, 245)
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    hud, left, right = 72, 42, 42
    bottom = height - (48 if mode == "labelled" else 24)
    lane_w = (width - left - right) / WIDTH

    indicator = green if state.cooldown == 0 else red
    draw.rounded_rectangle((20, 18, 56, 54), radius=8, fill=indicator, outline=white, width=2)
    if mode == "labelled":
        draw.text((68, 27), "READY (cooldown 0)" if state.cooldown == 0 else "BLOCKED (cooldown 1)", fill=white)

    for i in range(WIDTH + 1):
        x = round(left + i * lane_w)
        draw.line((x, hud, x, bottom), fill=grid, width=2)
    draw.line((left, bottom, width - right, bottom), fill=grid, width=3)

    if state.target_column is not None:
        cx = left + (state.target_column + 0.5) * lane_w
        cy, body_w, body_h = hud + 92, lane_w * 0.46, 52
        draw.rounded_rectangle(
            (cx - body_w / 2, cy - body_h / 2, cx + body_w / 2, cy + body_h / 2),
            radius=12,
            fill=magenta,
            outline=(255, 220, 240),
            width=3,
        )
        for eye_x in (cx - body_w * 0.18, cx + body_w * 0.18):
            draw.ellipse((eye_x - 5, cy - 12, eye_x + 5, cy - 2), fill=bg)
        draw.line((cx - body_w * 0.28, cy + 12, cx + body_w * 0.28, cy + 12), fill=bg, width=4)

    cx = left + (state.tank_column + 0.5) * lane_w
    y, body_w = bottom - 45, lane_w * 0.58
    draw.rounded_rectangle((cx - body_w / 2, y - 13, cx + body_w / 2, y + 17), radius=8, fill=cyan, outline=(215, 250, 255), width=3)
    draw.rectangle((cx - 7, y - 37, cx + 7, y - 10), fill=cyan)
    draw.polygon([(cx - body_w * 0.42, y + 17), (cx + body_w * 0.42, y + 17), (cx + body_w * 0.30, y + 33), (cx - body_w * 0.30, y + 33)], fill=cyan)

    if mode == "labelled":
        for i in range(WIDTH):
            cx = left + (i + 0.5) * lane_w
            draw.text((cx - 4, height - 32), str(i), fill=white)
    return image


def png_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def safe_name(text: str) -> str:
    return text.replace("|", "__").replace("=", "-")


def count_values(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def run(args: argparse.Namespace) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or Path("local-results") / f"{args.model}-zero-arcade-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    images = output / "images"
    images.mkdir()

    reader, artifact_id = policy_reader()
    provider: Provider = (
        FakeProvider()
        if args.backend == "fake"
        else OllamaProvider(args.ollama_url, args.model, args.timeout, args.seed)
    )
    states = smoke_states() if args.mode == "smoke" else all_states()
    if args.max_cases:
        states = states[: args.max_cases]

    records: list[dict[str, Any]] = []
    cases_path = output / "cases.jsonl"
    print(f"Running {len(states)} case(s) -> {output}")

    with cases_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, truth in enumerate(states, 1):
            raw_image = png_bytes(render(truth, args.render))
            image_path = images / f"{index:03d}-{safe_name(truth.row_id)}.png"
            image_path.write_bytes(raw_image)
            truth_action = reader.read(truth.row_id).action
            record: dict[str, Any] = {
                "index": index,
                "image_path": image_path.as_posix(),
                "image_sha256": "sha256:" + hashlib.sha256(raw_image).hexdigest(),
                "truth": truth.payload(),
                "truth_action": truth_action,
                "accepted": False,
                "prediction": None,
                "predicted_action": None,
                "exact_state_match": False,
                "action_match": False,
                "rejection_reason": None,
                "provider": None,
            }
            try:
                reply = provider.predict(raw_image, args.render, truth)
                prediction = Prediction.parse(reply.parsed)
                record["provider"] = {
                    "raw_text": reply.raw_text,
                    "parsed": reply.parsed,
                    "duration_ms": reply.duration_ms,
                    "metadata": reply.metadata,
                }
                record["prediction"] = prediction.payload()
                if prediction.confidence < args.confidence_threshold:
                    record["rejection_reason"] = "confidence_below_threshold"
                else:
                    decision = reader.read(prediction.row_id)
                    record["accepted"] = True
                    record["predicted_action"] = decision.action
                    record["exact_state_match"] = prediction.row_id == truth.row_id
                    record["action_match"] = decision.action == truth_action
            except Exception as exc:
                record["rejection_reason"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            print(
                f"[{index:03d}/{len(states):03d}] "
                f"{'ACCEPT' if record['accepted'] else 'REJECT':<6} "
                f"{truth.row_id:<37} "
                f"{'exact' if record['exact_state_match'] else 'not-exact':<9} "
                f"{'action-ok' if record['action_match'] else 'action-wrong'}"
            )

    accepted = [r for r in records if r["accepted"]]
    durations = [float(r["provider"]["duration_ms"]) for r in records if r.get("provider")]
    factor_names = ("tank_column", "target_present", "target_column", "cooldown")
    factor_accuracy = {}
    for name in factor_names:
        correct = sum(r["truth"][name] == r["prediction"][name] for r in accepted)
        factor_accuracy[name] = correct / len(accepted) if accepted else None
    exact = sum(bool(r["exact_state_match"]) for r in accepted)
    action = sum(bool(r["action_match"]) for r in accepted)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "backend": args.backend,
        "model": args.model if args.backend == "ollama" else "fake",
        "render_mode": args.render,
        "case_mode": args.mode,
        "confidence_threshold": args.confidence_threshold,
        "policy_artifact_id": artifact_id,
        "attempted": len(records),
        "accepted": len(accepted),
        "rejected": len(records) - len(accepted),
        "exact_state_correct": exact,
        "exact_state_accuracy_over_attempted": exact / len(records) if records else None,
        "action_correct": action,
        "action_accuracy_over_attempted": action / len(records) if records else None,
        "factor_accuracy_over_accepted": factor_accuracy,
        "latency_ms": {
            "mean": statistics.fmean(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "p95": p95(durations),
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
        },
        "rejection_reasons": count_values(str(r.get("rejection_reason") or "accepted") for r in records),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "python": sys.version,
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "summary": summary_path.as_posix(),
                "cases": cases_path.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("\n" + json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSend back:\n  {summary_path}\n  {cases_path}")
    return 0 if not summary["rejected"] else 2


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ollama", "fake"), default="ollama")
    parser.add_argument("--model", default="qwen3.5")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--mode", choices=("smoke", "all"), default="smoke")
    parser.add_argument("--render", choices=("labelled", "unlabelled"), default="labelled")
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be in [0,1]")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    if args.output_dir and args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    return args


if __name__ == "__main__":
    raise SystemExit(run(arguments()))
