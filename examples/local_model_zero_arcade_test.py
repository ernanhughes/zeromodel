#!/usr/bin/env python3
"""Local vision model -> ZeroModel arcade perception experiment.

A local vision-language model observes a rendered arcade frame and returns a
small text-protocol state description. ZeroModel validates that description,
addresses an independently compiled VPM policy, and selects the action.

Every case is now recorded as a first-class `ProviderEvaluationCaseDTO`
through the video action-set RMDTO aggregate (see
`docs/architecture/provider-evaluation-rmdto.md`): observation-correctness
(exact state) and application-behaviour-correctness (policy action) are kept
as separate, database-backed evidence rather than being collapsed into one
accuracy number in an ad hoc JSON file.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
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
    import numpy as np
except ImportError as exc:
    raise SystemExit("Install NumPy: python -m pip install numpy") from exc

try:
    from zeromodel.core import LayoutRecipe, ScoreTable, VPMPolicyLookup, build_vpm
    from zeromodel.core.matrix_blob import MatrixBlob
    from zeromodel.observation.visual_address import ImageObservation
    from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
    from zeromodel.video.domains.video_action_set.contracts import (
        BENCHMARK_VERSION,
        EPISODE_PLAN_VERSION,
        GENERATOR_VERSION,
        OBSERVATION_OPERATION_CHAIN_VERSION,
        SEED_DERIVATION_VERSION,
    )
    from zeromodel.video.domains.video_action_set.dto import (
        BenchmarkIdentityDTO,
        CanonicalJsonDTO,
        EpisodePlanDTO,
    )
    from zeromodel.video.domains.video_action_set.observation_dto import (
        MaterializedObservationDTO,
        ObservationDTO,
    )
    from zeromodel.video.domains.video_action_set.observation_provenance_dto import (
        ObservationOperationChainDTO,
    )
    from zeromodel.video.domains.video_action_set.provider_evaluation_dto import (
        MaterializedProviderEvaluationRunDTO,
        ProviderConfigurationDTO,
        ProviderEvaluationCaseContext,
        ProviderEvaluationCaseDTO,
        ProviderResponseEvidence,
        build_provider_evaluation_run,
        confidence_to_basis_points,
    )
    from zeromodel.video.domains.video_action_set.provider_observation_dto import (
        ProviderObservationDescriptorDTO,
    )
    from zeromodel.video.runtime import ZeroModelRuntime, build_runtime
except ImportError as exc:
    raise SystemExit(
        "ZeroModel is not importable. From the repository root run:\n"
        "  python -m pip install -r requirements-dev.txt"
    ) from exc

WIDTH = 7
ACTIONS = ("LEFT", "RIGHT", "STAY", "FIRE")
SCHEMA_VERSION = "zeromodel-local-model-arcade-text-state/v2"


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
        return cls(
            tank,
            present,
            None if target_raw == -1 else target_raw,
            cooldown,
            confidence,
        )

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
    """The provider boundary. `predict()` receives only observable input -
    the rendered image and the render mode - never ground truth. Expected
    state, expected row, and expected action remain owned exclusively by the
    evaluation harness that calls this protocol, never by the provider it
    calls.
    """

    def predict(self, image: bytes, render_mode: str) -> ProviderReply: ...


class ScriptedProvider:
    """A scripted wiring-test provider, not an oracle.

    `predict()` itself sees only image bytes and render mode - exactly what
    `Provider` declares - and looks up a pre-scripted reply by the image's
    content digest. The mapping from image digest to reply is built once,
    before any `predict()` call, by `_build_scripted_replies()`, which *does*
    know the deterministic fixture's ground truth (that is what "scripting"
    the replies means). This keeps the truth/prediction split real: fixture
    construction knows the truth, the provider call boundary does not.
    """

    def __init__(self, replies_by_image_digest: Mapping[str, ProviderReply]) -> None:
        self._replies_by_image_digest = dict(replies_by_image_digest)

    def predict(self, image: bytes, render_mode: str) -> ProviderReply:
        del render_mode
        digest = _image_digest(image)
        try:
            return self._replies_by_image_digest[digest]
        except KeyError as exc:
            raise RuntimeError(
                f"no scripted provider reply for image digest {digest}"
            ) from exc


class OllamaProvider:
    def __init__(self, base_url: str, model: str, timeout: float, seed: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.seed = seed
        print(
            f"[INFO] OllamaProvider: connecting to {self.base_url}, model={self.model}"
        )
        self.model_record = self._find_model()

    def _json(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
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
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}: {exc}"
            ) from exc
        if not isinstance(result, dict):
            raise RuntimeError("Ollama returned non-object JSON")
        return result

    def _find_model(self) -> dict[str, Any]:
        print("[INFO] Checking available Ollama models...")
        models = self._json("GET", "/api/tags").get("models", [])
        available: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "")
            if name:
                available.append(name)
            if name == self.model or str(item.get("model", "")) == self.model:
                print(f"[INFO] Model {self.model!r} found.")
                return item
        raise RuntimeError(
            f"Ollama model {self.model!r} not found. Installed: "
            + (", ".join(available) or "<none>")
        )

    def predict(self, image: bytes, render_mode: str) -> ProviderReply:
        prompt = prompt_for(render_mode)
        base64_image = base64.b64encode(image).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64_image],
                }
            ],
            # Qwen 3.5 is a thinking model. Disable thinking at the top-level
            # Ollama request field so the answer budget is used for visible text.
            "think": False,
            "stream": False,
            # Do not use Ollama's server-side JSON/schema mode here. Local VLMs
            # are more reliable with a tiny text protocol parsed client-side.
            "options": {"temperature": 0.0, "seed": self.seed, "num_predict": 128},
        }
        print(
            f"[INFO] Sending chat request (image size: {len(image)} bytes, prompt length: {len(prompt)} chars)"
        )
        started = time.perf_counter()
        try:
            response = self._json("POST", "/api/chat", payload)
        except Exception as e:
            print(f"[ERROR] Ollama request failed: {e}")
            raise
        wall_ms = (time.perf_counter() - started) * 1000

        message = response.get("message")
        if not isinstance(message, dict):
            print(
                f"[ERROR] Ollama response missing 'message' dict, got: {type(message)}"
            )
            raise RuntimeError("Ollama response has no 'message' object")

        raw = message.get("content")
        thinking = message.get("thinking")
        if not isinstance(raw, str):
            print(f"[ERROR] Ollama message missing 'content' string, got: {type(raw)}")
            raise RuntimeError("Ollama response has no string 'content' in message")

        thinking_length = len(thinking) if isinstance(thinking, str) else 0
        print(
            "[INFO] Ollama response: "
            f"content_chars={len(raw)}, thinking_chars={thinking_length}, "
            f"done_reason={response.get('done_reason')!r}"
        )
        if not raw.strip():
            raise RuntimeError(
                "Ollama returned empty message.content "
                f"(thinking_chars={thinking_length}, "
                f"eval_count={response.get('eval_count')}, "
                f"done_reason={response.get('done_reason')!r})"
            )

        print(f"[INFO] Ollama raw response (first 500 chars): {raw[:500]}...")
        parsed = parse_state_text(raw)

        duration = response.get("total_duration")
        duration_ms = (
            float(duration) / 1_000_000
            if isinstance(duration, (int, float))
            else wall_ms
        )
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
                "thinking_length": thinking_length,
                "response_keys": sorted(response.keys()),
                "message_keys": sorted(message.keys()),
            },
        )


def parse_state_text(raw: str) -> dict[str, Any]:
    """Parse a small labelled text protocol into the existing state payload.

    The parser tolerates whitespace, hyphens, equals signs, and harmless prose,
    but it requires one unambiguous value for each declared field.
    """
    import re

    def one(pattern: str, name: str) -> str:
        matches = re.findall(pattern, raw, flags=re.IGNORECASE | re.MULTILINE)
        values = [str(value).strip() for value in matches]
        unique = list(dict.fromkeys(value.lower() for value in values))
        if not unique:
            raise ValueError(f"model text is missing {name}: {raw!r}")
        if len(unique) != 1:
            raise ValueError(f"model text has conflicting {name} values: {values!r}")
        return values[0]

    tank_text = one(
        r"(?:tank(?:[_ -]?column)?|cyan(?:[_ -]?tank)?(?:[_ -]?lane)?)\s*[:=]\s*([0-6])\b",
        "tank column",
    )
    target_text = one(
        r"(?:target(?:[_ -]?column)?|magenta(?:[_ -]?(?:target|alien))?(?:[_ -]?lane)?)"
        r"\s*[:=]\s*(none|absent|no[_ -]?target|-1|[0-6])\b",
        "target column",
    )
    cooldown_text = one(
        r"(?:cooldown|status(?:[_ -]?(?:light|indicator))?)\s*[:=]\s*"
        r"(ready|green|blocked|red|0|1)\b",
        "cooldown",
    )
    confidence_text = one(
        r"confidence\s*[:=]\s*(100|[0-9]{1,2})(?:\s*%)?\b",
        "confidence",
    )

    tank = int(tank_text)
    target_missing = target_text.strip().lower() in {
        "none",
        "absent",
        "no_target",
        "no-target",
        "no target",
        "-1",
    }
    target = -1 if target_missing else int(target_text)

    cooldown_token = cooldown_text.strip().lower()
    cooldown = 0 if cooldown_token in {"ready", "green", "0"} else 1
    confidence = int(confidence_text) / 100.0

    return {
        "tank_column": tank,
        "target_present": target != -1,
        "target_column": target,
        "cooldown": cooldown,
        "confidence": confidence,
    }


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
- tank column: lane containing the cyan tank at the bottom
- target column: magenta alien lane, or NONE when no alien is visible
- cooldown: READY when the top-left indicator is green; BLOCKED when red
- confidence: whole-number confidence percentage from 0 to 100

{lanes}

Return exactly these four labelled lines. Do not return JSON and do not add prose:
TANK_COLUMN: <0-6>
TARGET_COLUMN: <NONE or 0-6>
COOLDOWN: <READY or BLOCKED>
CONFIDENCE: <0-100>
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
    print("[INFO] Building VPM policy table...")
    states = all_states()
    table = ScoreTable(
        values=[action_values(state) for state in states],
        row_ids=[state.row_id for state in states],
        metric_ids=ACTIONS,
        metadata={"kind": "local_model_arcade_policy", "schema": SCHEMA_VERSION},
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "local-model-arcade-source-order",
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
    reader = VPMPolicyLookup(artifact, action_metric_ids=ACTIONS)
    print(f"[INFO] Policy built, artifact_id={artifact.artifact_id}")
    return reader, artifact.artifact_id


def render(
    state: ArcadeState, mode: str, width: int = 896, height: int = 512
) -> Image.Image:
    bg, grid = (13, 18, 30), (53, 66, 92)
    cyan, magenta = (65, 220, 255), (255, 74, 171)
    green, red, white = (70, 225, 125), (242, 74, 74), (224, 232, 245)
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    hud, left, right = 72, 42, 42
    bottom = height - (48 if mode == "labelled" else 24)
    lane_w = (width - left - right) / WIDTH

    indicator = green if state.cooldown == 0 else red
    draw.rounded_rectangle(
        (20, 18, 56, 54), radius=8, fill=indicator, outline=white, width=2
    )
    if mode == "labelled":
        draw.text(
            (68, 27),
            "READY (cooldown 0)" if state.cooldown == 0 else "BLOCKED (cooldown 1)",
            fill=white,
        )

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
        draw.line(
            (cx - body_w * 0.28, cy + 12, cx + body_w * 0.28, cy + 12), fill=bg, width=4
        )

    cx = left + (state.tank_column + 0.5) * lane_w
    y, body_w = bottom - 45, lane_w * 0.58
    draw.rounded_rectangle(
        (cx - body_w / 2, y - 13, cx + body_w / 2, y + 17),
        radius=8,
        fill=cyan,
        outline=(215, 250, 255),
        width=3,
    )
    draw.rectangle((cx - 7, y - 37, cx + 7, y - 10), fill=cyan)
    draw.polygon(
        [
            (cx - body_w * 0.42, y + 17),
            (cx + body_w * 0.42, y + 17),
            (cx + body_w * 0.30, y + 33),
            (cx - body_w * 0.30, y + 33),
        ],
        fill=cyan,
    )

    if mode == "labelled":
        for i in range(WIDTH):
            cx = left + (i + 0.5) * lane_w
            draw.text((cx - 4, height - 32), str(i), fill=white)
    return image


def png_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def _image_digest(data: bytes) -> str:
    """The one canonical content-digest convention for full-resolution PNG
    bytes in this example - used for the scripted-provider reply key, the
    per-case `image_sha256` record field, and the observation's
    `full_resolution_image_sha256` metadata, so there is exactly one digest
    convention for "this image's bytes," not several."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _prerender_states(
    states: Sequence[ArcadeState], render_mode: str
) -> list[tuple[Image.Image, bytes]]:
    """Render every state's frame once, up front. Reused both to avoid
    re-rendering per case and so `_build_scripted_replies` can compute each
    scripted reply's image-digest key from the exact bytes `predict()` will
    later receive."""
    rendered: list[tuple[Image.Image, bytes]] = []
    for truth in states:
        image = render(truth, render_mode)
        rendered.append((image, png_bytes(image)))
    return rendered


def _build_scripted_replies(
    states: Sequence[ArcadeState],
    prerendered: Sequence[tuple[Image.Image, bytes]],
) -> dict[str, ProviderReply]:
    """Build the fake backend's scripted reply-by-image-digest mapping.

    This function knows the ground truth (`states`) - that is what
    "scripting" the replies means. `ScriptedProvider.predict()` itself never
    sees `states` or any `ArcadeState`; it only ever sees the image bytes and
    render mode, then looks up the reply this function already prepared.
    """
    replies: dict[str, ProviderReply] = {}
    for truth, (_image, raw_image) in zip(states, prerendered, strict=True):
        payload = {
            "tank_column": truth.tank_column,
            "target_present": truth.target_present,
            "target_column": -1 if truth.target_column is None else truth.target_column,
            "cooldown": truth.cooldown,
            "confidence": 1.0,
        }
        replies[_image_digest(raw_image)] = ProviderReply(
            json.dumps(payload), payload, 0.0, {"backend": "fake"}
        )
    return replies


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


def _build_benchmark_identity(
    *, model: str, artifact_id: str, stamp: str
) -> BenchmarkIdentityDTO:
    seed_material = f"local-model-arcade:{model}:{stamp}"
    seed_digest = "sha256:" + hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    return BenchmarkIdentityDTO(
        contract_commit="local-model-zero-arcade-example",
        seed_material=seed_material,
        seed_digest=seed_digest,
        policy_artifact_id=artifact_id,
        parent_audit_sha="n/a",
        parent_v3_sha="n/a",
    )


def _build_episode_plan(
    *, identity: BenchmarkIdentityDTO, episode_id: str, frame_count: int
) -> EpisodePlanDTO:
    """Build the minimal `EpisodePlanDTO` this synthetic arcade fixture needs
    to own its observations. `ObservationDTO` requires a real owning episode
    plan (see `docs/architecture/video-action-set-rmdto.md`); this mirrors the
    RMDTO test suite's own minimal-fixture builder pattern
    (`sample_identity()`/`sample_plan_payload()` in
    `tests/test_video_episode_plan_rmdto.py`) rather than inventing a second
    convention.
    """
    split = "development"

    def _seed_node(
        namespace: str, parents: tuple[tuple[str, str], ...] = ()
    ) -> dict[str, Any]:
        payload = {
            "version": SEED_DERIVATION_VERSION,
            "root_seed_digest": identity.seed_digest,
            "split": split,
            "episode_ordinal": 0,
            "namespace": namespace,
            "parent_identities": [
                {"name": name, "identity": value} for name, value in parents
            ],
        }
        digest = canonical_sha256(payload)
        seed_int64 = int(digest.removeprefix("sha256:")[:16], 16)
        return payload | {"seed_digest": digest, "seed_int64": seed_int64}

    root_node = _seed_node("root")
    concrete_node = _seed_node(
        "concrete_episode_seed",
        parents=(("root", root_node["seed_digest"]),),
    )
    seed_lineage = {"root": root_node, "concrete_episode_seed": concrete_node}
    frame_plans = [{"frame_index": index} for index in range(frame_count)]
    payload = {
        "version": EPISODE_PLAN_VERSION,
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "episode_id": episode_id,
        "split": split,
        "ordinal": 0,
        "family_label": "valid",
        "family_ordinal": 0,
        "episode_family": "valid",
        "episode_disposition": "valid_episode",
        "denominator_class": "valid",
        "final_observation_provenance": {
            "materialization_status": "materialized",
            "observation_payload_included": True,
            "provenance": "in_memory_generation",
        },
        "mutation_kind": None,
        "source_row_id": "local-model-arcade:source",
        "secondary_row_id": None,
        "family_contract": {},
        "family_intervention": {},
        "derived_seed_identity": concrete_node["seed_digest"],
        "episode_seed": concrete_node["seed_int64"],
        "frame_count": frame_count,
        "seed_lineage": seed_lineage,
        "frame_plans": frame_plans,
    }
    return EpisodePlanDTO.from_dict(
        payload | {"plan_digest": canonical_sha256(payload)}
    )


def _build_observation(
    *,
    identity: BenchmarkIdentityDTO,
    plan: EpisodePlanDTO,
    frame_index: int,
    image: Image.Image,
    raw_image: bytes,
    render_mode: str,
    truth_row_id: str,
    truth_action: str,
) -> MaterializedObservationDTO:
    """Materialize one rendered arcade frame as an `ObservationDTO`.

    `ObservationDTO`'s existing pixel-materialization contract
    (`validate_observation_matrix_blob`) is hard-coded to the reachability
    benchmark's fixed 16x28 uint8 frame shape - it is not a general-purpose
    image store, and the brief explicitly rules out building one. Rather than
    weakening that existing, preserved contract, this computes one
    deterministic 16x28 grayscale thumbnail of the real rendered PNG to
    satisfy it, and links the two by digest in metadata. The full-resolution
    PNG on disk - not the thumbnail - remains the actual frame sent to the
    provider and referenced by the evaluation case.
    """
    frame_id = f"{plan.split}:{plan.episode_id}:frame-{frame_index:02d}"
    thumbnail = np.asarray(image.convert("L").resize((28, 16)), dtype=np.uint8)
    pixel_digest = (
        "sha256:"
        + hashlib.sha256(np.ascontiguousarray(thumbnail).tobytes(order="C")).hexdigest()
    )
    blob = MatrixBlob.from_array(
        thumbnail,
        dtype="uint8",
        metadata={
            "kind": "video_action_set_frame_pixels",
            "pixel_digest": pixel_digest,
        },
    )
    descriptor = ProviderObservationDescriptorDTO.from_dict(
        ImageObservation(thumbnail, source_id=frame_id).to_descriptor()
    )
    full_resolution_digest = _image_digest(raw_image)
    operation_payload: dict[str, Any] = {
        "index": 0,
        "operation": "render_frame",
        "operation_version": SCHEMA_VERSION,
        "input_digests": [None],
        "parameters": {
            "render_mode": render_mode,
            "full_resolution_png_sha256": full_resolution_digest,
        },
        "output_digest": pixel_digest,
    }
    operation_payload["parameter_digest"] = canonical_sha256(
        operation_payload["parameters"]
    )
    operation_payload = operation_payload | {
        "operation_digest": canonical_sha256(operation_payload)
    }
    chain_payload = {
        "version": OBSERVATION_OPERATION_CHAIN_VERSION,
        "operations": [operation_payload],
        "final_emitted_digest": pixel_digest,
    }
    chain = ObservationOperationChainDTO.from_dict(
        chain_payload | {"operation_chain_digest": canonical_sha256(chain_payload)}
    )
    observation = ObservationDTO(
        benchmark_version=BENCHMARK_VERSION,
        generator_version=GENERATOR_VERSION,
        benchmark_seed_digest=identity.seed_digest,
        episode_plan_digest=plan.plan_digest,
        split=plan.split,
        episode_id=plan.episode_id,
        clip_id=f"{plan.split}:{plan.episode_id}:clip",
        frame_id=frame_id,
        sequence_number=frame_index,
        event_type="frame",
        family="local_model_arcade",
        expected_disposition="valid",
        episode_family=plan.episode_family,
        episode_disposition=plan.episode_disposition,
        frame_disposition="valid_frame_payload",
        denominator_class=plan.denominator_class,
        expected_row=truth_row_id,
        expected_action=truth_action,
        actual_executed_action=truth_action,
        action_known=True,
        gap_declaration=None,
        observation_pixel_digest=pixel_digest,
        matrix_blob_id=blob.blob_id,
        provider_observation_descriptor=descriptor,
        provider_observation_digest=descriptor.descriptor_digest,
        operation_chain=chain,
        metadata=CanonicalJsonDTO.from_value(
            {
                "full_resolution_image_sha256": full_resolution_digest,
                "render_mode": render_mode,
                "pixel_note": (
                    "matrix_blob/provider descriptor are a deterministic 16x28 "
                    "grayscale thumbnail of the full-resolution PNG referenced "
                    "by full_resolution_image_sha256; the full-resolution PNG "
                    "on disk is the actual frame the provider evaluated."
                ),
            }
        ),
        final_access_id=None,
    )
    return MaterializedObservationDTO(observation, blob)


def _state_factors(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"row_id", "confidence"}
    }


def _decision_payload(decision: Any, *, policy_artifact_id: str) -> dict[str, Any]:
    """Normalize a `PolicyLookupDecision` into the video domain's sha256:-prefixed
    identity convention. `VPMArtifact.artifact_id` (and therefore
    `PolicyLookupDecision.artifact_id`) is a bare hex digest - Core's own
    convention - while every identity field in the provider-evaluation
    aggregate uses the `sha256:`-prefixed convention used throughout
    `video_action_set`. This is the one place that boundary is crossed.
    """
    return dict(decision.to_dict()) | {"artifact_id": policy_artifact_id}


def _build_runtime(args: argparse.Namespace) -> ZeroModelRuntime:
    if args.store == "sqlite":
        from zeromodel.persistence.sqlalchemy.db.runtime import build_sqlite_runtime

        return build_sqlite_runtime(
            f"sqlite:///{args.sqlite_path}", initialize_schema=True
        )
    return build_runtime()


def run(args: argparse.Namespace) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = args.model.replace(".", "_").replace(":", "_")
    output = (
        args.output_dir or Path("./local-results") / f"{model_path}-zero-arcade-{stamp}"
    )
    output.mkdir(parents=True, exist_ok=False)
    images = output / "images"
    images.mkdir()
    print(f"[INFO] Results will be written to {output}")

    reader, artifact_id = policy_reader()
    policy_artifact_id = f"sha256:{artifact_id}"
    states = smoke_states() if args.mode == "smoke" else all_states()
    if args.max_cases:
        states = states[: args.max_cases]
    prerendered = _prerender_states(states, args.render)

    provider: Provider
    if args.backend == "fake":
        # Fixture construction (this branch) knows the ground truth; the
        # provider call boundary (`ScriptedProvider.predict`) does not - see
        # `Provider`'s docstring.
        provider = ScriptedProvider(_build_scripted_replies(states, prerendered))
    else:
        provider = OllamaProvider(args.ollama_url, args.model, args.timeout, args.seed)

    runtime = _build_runtime(args)
    facade = runtime.video_action_set

    identity = _build_benchmark_identity(
        model=args.model, artifact_id=policy_artifact_id, stamp=stamp
    )
    facade.save_identity(identity)
    episode_id = f"development:local-model-arcade-{stamp}"
    plan = _build_episode_plan(
        identity=identity, episode_id=episode_id, frame_count=len(states)
    )
    facade.save_episode_plan(plan)

    provider_kind = "fake" if args.backend == "fake" else "ollama"
    model_name = args.model if args.backend == "ollama" else "fake"
    model_digest = "sha256:" + hashlib.sha256(model_name.encode("utf-8")).hexdigest()
    prompt_digest = (
        "sha256:" + hashlib.sha256(prompt_for(args.render).encode("utf-8")).hexdigest()
    )
    provider_configuration = ProviderConfigurationDTO.build(
        provider_kind=provider_kind,
        model_name=model_name,
        model_digest=model_digest,
        runtime_name="ollama" if args.backend == "ollama" else "in-process-fake",
        protocol_version=SCHEMA_VERSION,
        prompt_digest=prompt_digest,
        seed=args.seed,
        inference_options=(
            {"temperature": 0.0, "num_predict": 128} if args.backend == "ollama" else {}
        ),
        metadata={"backend": args.backend},
    )

    records: list[dict[str, Any]] = []
    cases: list[ProviderEvaluationCaseDTO] = []
    cases_path = output / "cases.jsonl"
    print(f"[INFO] Running {len(states)} case(s) -> {output}")

    with cases_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, truth in enumerate(states, 1):
            frame_index = index - 1
            print(f"\n[INFO] === Case {index}/{len(states)}: {truth.row_id} ===")
            image, raw_image = prerendered[frame_index]
            image_path = images / f"{index:03d}-{safe_name(truth.row_id)}.png"
            image_path.write_bytes(raw_image)
            print(f"[INFO] Image saved to {image_path}")

            truth_decision = reader.read(truth.row_id)
            truth_action = truth_decision.action

            materialized_observation = _build_observation(
                identity=identity,
                plan=plan,
                frame_index=frame_index,
                image=image,
                raw_image=raw_image,
                render_mode=args.render,
                truth_row_id=truth.row_id,
                truth_action=truth_action,
            )
            saved_observation = facade.save_materialized_observation(
                materialized_observation
            )
            frame_id = saved_observation.frame_id

            record: dict[str, Any] = {
                "index": index,
                "frame_id": frame_id,
                "image_path": image_path.as_posix(),
                "image_sha256": _image_digest(raw_image),
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
            case_context = ProviderEvaluationCaseContext(
                policy_artifact_id=policy_artifact_id,
                provider_configuration_id=provider_configuration.provider_configuration_id,
            )
            case_kwargs: dict[str, Any] = dict(
                case_ordinal=frame_index,
                frame_id=frame_id,
                context=case_context,
                expected_state=_state_factors(truth.payload()),
                expected_decision=_decision_payload(
                    truth_decision, policy_artifact_id=policy_artifact_id
                ),
            )
            try:
                print("[INFO] Calling provider.predict()...")
                reply = provider.predict(raw_image, args.render)
                print(f"[INFO] Provider replied in {reply.duration_ms:.2f} ms")
                print(f"[INFO] Parsed state text: {reply.parsed}")
                prediction = Prediction.parse(reply.parsed)
                print(f"[INFO] Parsed prediction: {prediction}")
                record["provider"] = {
                    "raw_text": reply.raw_text,
                    "parsed": reply.parsed,
                    "duration_ms": reply.duration_ms,
                    "metadata": reply.metadata,
                }
                record["prediction"] = prediction.payload()
                latency_us = int(round(reply.duration_ms * 1000))
                confidence_basis_points = confidence_to_basis_points(
                    prediction.confidence
                )
                if prediction.confidence < args.confidence_threshold:
                    record["rejection_reason"] = "confidence_below_threshold"
                    print(
                        f"[INFO] Rejected: confidence {prediction.confidence} < threshold {args.confidence_threshold}"
                    )
                    case = ProviderEvaluationCaseDTO.build(
                        **case_kwargs,
                        accepted=False,
                        evidence=ProviderResponseEvidence(
                            rejection_reason="confidence_below_threshold",
                            provider_confidence_basis_points=confidence_basis_points,
                            provider_latency_us=latency_us,
                            provider_raw_response_text=reply.raw_text,
                            provider_response_metadata=reply.metadata,
                        ),
                    )
                else:
                    decision = reader.read(prediction.row_id)
                    record["accepted"] = True
                    record["predicted_action"] = decision.action
                    record["exact_state_match"] = prediction.row_id == truth.row_id
                    record["action_match"] = decision.action == truth_action
                    print(
                        f"[INFO] Accepted. Predicted state: {prediction.row_id}, action: {decision.action}"
                    )
                    print(
                        f"[INFO] Exact state match: {record['exact_state_match']}, action match: {record['action_match']}"
                    )
                    case = ProviderEvaluationCaseDTO.build(
                        **case_kwargs,
                        accepted=True,
                        predicted_state=_state_factors(prediction.payload()),
                        predicted_decision=_decision_payload(
                            decision, policy_artifact_id=policy_artifact_id
                        ),
                        evidence=ProviderResponseEvidence(
                            provider_confidence_basis_points=confidence_basis_points,
                            provider_latency_us=latency_us,
                            provider_raw_response_text=reply.raw_text,
                            provider_response_metadata=reply.metadata,
                        ),
                    )
            except Exception as exc:
                print(
                    f"[ERROR] Exception during processing: {type(exc).__name__}: {exc}"
                )
                import traceback

                traceback.print_exc()
                record["rejection_reason"] = f"{type(exc).__name__}: {exc}"
                case = ProviderEvaluationCaseDTO.build(
                    **case_kwargs,
                    accepted=False,
                    evidence=ProviderResponseEvidence(
                        rejection_reason=record["rejection_reason"]
                    ),
                )
            # provider_confidence_basis_points is the persisted case's source
            # truth; provider_confidence is a derived presentation float and
            # must not be treated as identity-bearing.
            record["provider_confidence_basis_points"] = (
                case.provider_confidence_basis_points
            )
            record["provider_confidence"] = case.provider_confidence
            cases.append(case)
            records.append(record)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            print(
                f"[STATUS] {index:03d}/{len(states):03d} "
                f"{'ACCEPT' if record['accepted'] else 'REJECT':<6} "
                f"{truth.row_id:<37} "
                f"{'exact' if record['exact_state_match'] else 'not-exact':<9} "
                f"{'action-ok' if record['action_match'] else 'action-wrong'}"
            )

    materialized_run = build_provider_evaluation_run(
        fixture_identity=f"local-model-arcade:{args.mode}:{args.render}",
        provider_configuration=provider_configuration,
        policy_artifact_id=policy_artifact_id,
        case_mode=args.mode,
        representation_mode=args.render,
        cases=cases,
        metadata={"model": args.model, "backend": args.backend},
    )
    saved_run = facade.save_provider_evaluation_run(materialized_run)
    reloaded_run = facade.get_materialized_provider_evaluation_run(saved_run.run.run_id)
    if reloaded_run != materialized_run:
        raise RuntimeError(
            "provider evaluation aggregate failed to reload identically after save"
        )
    print(
        f"[INFO] Provider evaluation run saved and reload-verified: {saved_run.run.run_id}"
    )

    summary_dto = saved_run.summary
    factor_correct = summary_dto.factor_correct_counts.to_value()
    factor_denominators = summary_dto.factor_denominators.to_value()
    factor_accuracy = {
        key: (factor_correct[key] / factor_denominators[key])
        for key in factor_denominators
        if factor_denominators[key]
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "backend": args.backend,
        "model": args.model if args.backend == "ollama" else "fake",
        "render_mode": args.render,
        "case_mode": args.mode,
        "confidence_threshold": args.confidence_threshold,
        "policy_artifact_id": artifact_id,
        "run_id": saved_run.run.run_id,
        "attempted": summary_dto.attempted_count,
        "accepted": summary_dto.accepted_count,
        "rejected": summary_dto.rejected_count,
        "exact_state_correct": summary_dto.exact_count,
        "exact_state_accuracy_over_attempted": (
            summary_dto.exact_count / summary_dto.attempted_count
            if summary_dto.attempted_count
            else None
        ),
        "action_correct": summary_dto.action_correct_count,
        "action_accuracy_over_attempted": (
            summary_dto.action_correct_count / summary_dto.attempted_count
            if summary_dto.attempted_count
            else None
        ),
        "action_equivalent_count": summary_dto.action_equivalent_count,
        "action_changing_count": summary_dto.action_changing_count,
        "factor_accuracy_over_accepted": factor_accuracy,
        "latency_ms": {
            "mean": (
                summary_dto.latency_total_us / summary_dto.latency_sample_count / 1000.0
                if summary_dto.latency_sample_count
                else None
            ),
            "median": (
                summary_dto.latency_median_us / 1000.0
                if summary_dto.latency_median_us is not None
                else None
            ),
            "p95": (
                summary_dto.latency_p95_us / 1000.0
                if summary_dto.latency_p95_us is not None
                else None
            ),
            "min": (
                summary_dto.latency_min_us / 1000.0
                if summary_dto.latency_min_us is not None
                else None
            ),
            "max": (
                summary_dto.latency_max_us / 1000.0
                if summary_dto.latency_max_us is not None
                else None
            ),
        },
        "rejection_reasons": dict(summary_dto.rejection_reason_counts.to_value()),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report_path: Path | None = None
    if args.compile_report:
        report_path = _compile_report(saved_run, output=output)

    (output / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "python": sys.version,
                "run_id": saved_run.run.run_id,
                "arguments": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "summary": summary_path.as_posix(),
                "cases": cases_path.as_posix(),
                "report": None if report_path is None else report_path.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("\n" + "=" * 60)
    print("[INFO] Summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\n[INFO] Files written:\n  {summary_path}\n  {cases_path}")
    return 0 if not summary["rejected"] else 2


def _compile_report(
    saved_run: MaterializedProviderEvaluationRunDTO, *, output: Path
) -> Path:
    from provider_evaluation_report_adapter import compile_provider_evaluation_report
    from zeromodel.artifacts import (
        InMemoryArtifactStore,
        load_compiled_report_aggregate,
    )

    layout_recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "provider-evaluation-priority-order",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    artifact_store = InMemoryArtifactStore()
    compiled = compile_provider_evaluation_report(
        saved_run, layout_recipe=layout_recipe, store=artifact_store
    )
    aggregate = load_compiled_report_aggregate(
        ref=compiled.artifact_ref, resolver=artifact_store
    )
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "compiled_report_artifact_id": compiled.artifact_ref.artifact_id,
                "adapted_report_id": aggregate.adapted_report.adapted_report_id,
                "vpm_artifact_id": aggregate.vpm_artifact.artifact_id,
                "subject_count": len(aggregate.adapted_report.subjects),
                "dimension_count": len(aggregate.adapted_report.dimensions),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"[INFO] Compiled report written to {report_path}")
    return report_path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ollama", "fake"), default="ollama")
    parser.add_argument("--model", default="qwen3.5")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--mode", choices=("smoke", "all"), default="smoke")
    parser.add_argument(
        "--render", choices=("labelled", "unlabelled"), default="labelled"
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--store", choices=("memory", "sqlite"), default="memory")
    parser.add_argument("--sqlite-path", type=Path)
    parser.add_argument(
        "--compile-report",
        action="store_true",
        help="Compile the evaluation run into a VPM report via the external adapter.",
    )
    args = parser.parse_args()
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be in [0,1]")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    if args.output_dir and args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    if args.store == "sqlite" and args.sqlite_path is None:
        parser.error("--sqlite-path is required when --store sqlite")
    return args


if __name__ == "__main__":
    raise SystemExit(run(arguments()))
