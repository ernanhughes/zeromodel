"""Provider-isolation regression coverage for the local-model arcade example.

The provider call boundary (`Provider.predict`) must receive only observable
input - image bytes and render mode - never ground truth. Expected state,
expected row, expected action, and `ArcadeState` itself remain owned
exclusively by the evaluation harness. This is a runtime behavioral test
(a spy provider records what `run()` actually passes at the call site), not
just a static inspection of parameter names, plus a lightweight signature
assertion as a secondary guard.

Only the `--backend fake` path is exercised here - Ollama execution stays
external/manual and is never part of the fast suite.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import examples.local_model_zero_arcade_test as arcade_example


def _fake_run_arguments(output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        backend="fake",
        model="qwen3.5",
        ollama_url="http://localhost:11434",
        mode="smoke",
        render="labelled",
        confidence_threshold=0.0,
        timeout=180.0,
        seed=0,
        max_cases=None,
        output_dir=output_dir,
        store="memory",
        sqlite_path=None,
        compile_report=False,
    )


def test_provider_protocol_signature_excludes_ground_truth() -> None:
    """Lightweight secondary guard: the declared protocol/implementations
    accept exactly `(self, image, render_mode)` - no `truth`/`ArcadeState`
    parameter."""
    for predict in (
        arcade_example.Provider.predict,
        arcade_example.ScriptedProvider.predict,
        arcade_example.OllamaProvider.predict,
    ):
        parameters = list(inspect.signature(predict).parameters)
        assert parameters == ["self", "image", "render_mode"], predict


def test_scripted_provider_call_boundary_receives_no_ground_truth(
    tmp_path: Path,
) -> None:
    """Runtime behavioral proof: run the smoke harness end-to-end with a spy
    that records every argument `run()` actually passes to `predict()`."""
    recorded_calls: list[tuple[Any, Any]] = []

    class _SpyScriptedProvider(arcade_example.ScriptedProvider):
        def predict(
            self, image: bytes, render_mode: str
        ) -> arcade_example.ProviderReply:  # type: ignore[override]
            recorded_calls.append((image, render_mode))
            return super().predict(image, render_mode)

    original_scripted_provider = arcade_example.ScriptedProvider
    arcade_example.ScriptedProvider = _SpyScriptedProvider  # type: ignore[misc]
    try:
        exit_code = arcade_example.run(
            _fake_run_arguments(tmp_path / "provider-isolation-run")
        )
    finally:
        arcade_example.ScriptedProvider = original_scripted_provider  # type: ignore[misc]

    assert exit_code == 0
    assert len(recorded_calls) == 8  # smoke fixture size

    for image, render_mode in recorded_calls:
        # Exactly the observable input the Provider protocol declares - a
        # raw byte string and a plain string render mode.
        assert isinstance(image, bytes)
        assert isinstance(render_mode, str)
        # No ArcadeState (or anything resembling ground truth) ever reaches
        # the provider call boundary.
        assert not isinstance(image, arcade_example.ArcadeState)
        assert not hasattr(image, "tank_column")
        assert not hasattr(image, "target_column")
        assert not hasattr(image, "row_id")

    # The harness still computes comparisons after the provider returns -
    # the fixture is the exact-match smoke fixture, so every case should be
    # accepted and exact.
    output = tmp_path / "provider-isolation-run"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["attempted"] == 8
    assert summary["accepted"] == 8
    assert summary["exact_state_correct"] == 8
    assert summary["action_correct"] == 8

    cases = [
        json.loads(line)
        for line in (output / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(cases) == 8
    assert all(case["exact_state_match"] for case in cases)
    assert all(case["action_match"] for case in cases)


def test_scripted_provider_raises_on_unscripted_image_digest() -> None:
    """A `ScriptedProvider` with no reply for a given image digest fails
    loudly rather than silently fabricating a response - it is a scripted
    wiring provider, not an oracle that can answer for arbitrary input."""
    provider = arcade_example.ScriptedProvider({})
    try:
        provider.predict(b"not a scripted image", "labelled")
    except RuntimeError as exc:
        assert "no scripted provider reply" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for an unscripted image digest")
