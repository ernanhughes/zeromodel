from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from zeromodel import VPMPolicyLookup


# ---------------------------------------------------------------------------
# Helper: load the arcade shooter demo module (same as in your tests)
# ---------------------------------------------------------------------------
def _load_demo():
    path = Path(__file__).resolve().parents[1] / "examples" / "arcade_shooter_policy.py"
    spec = importlib.util.spec_from_file_location("arcade_shooter_policy_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# A plain dict‑based policy with canonical SHA‑256 identity.
# ---------------------------------------------------------------------------
@dataclass
class DictPolicyDecision:
    action: str
    candidates: Dict[str, float]
    artifact_id: str
    row_id: str
    value: float


class DictPolicy:
    """A minimal policy table stored as a dict, with SHA‑256 identity."""

    def __init__(self, state_action_map: Dict[str, Tuple[float, ...]], action_names: Tuple[str, ...]):
        self._map = state_action_map
        self.action_names = action_names
        self.artifact_id = self._compute_artifact_id()

    def _compute_artifact_id(self) -> str:
        # Sort by row_id for canonical ordering.
        items = sorted(self._map.items())
        serialized = json.dumps(items, sort_keys=True).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def read(self, row_id: str) -> DictPolicyDecision:
        values = self._map[row_id]
        winner_idx = max(range(len(self.action_names)),
                         key=lambda i: (values[i], -i))
        return DictPolicyDecision(
            action=self.action_names[winner_idx],
            candidates=dict(zip(self.action_names, values)),
            artifact_id=self.artifact_id,
            row_id=row_id,
            value=values[winner_idx],
        )


def _build_dict_policy(demo) -> DictPolicy:
    """Build a DictPolicy that reproduces the exact same source policy as the demo."""
    config = demo.ShooterConfig()
    targets: tuple[int | None, ...] = (None,) + tuple(range(config.width))
    mapping: Dict[str, Tuple[float, ...]] = {}
    for tank_x in range(config.width):
        for target_x in targets:
            for cooldown in (0, 1):
                row_id = demo.state_row_id(tank_x, target_x, cooldown)
                values = demo._action_values(tank_x, target_x, cooldown)
                mapping[row_id] = values
    return DictPolicy(mapping, demo.ACTIONS)


# ---------------------------------------------------------------------------
# Tests: same logical validation as the VPM suite, plus performance.
# ---------------------------------------------------------------------------
def test_dict_policy_exhaustive_fidelity() -> None:
    """DictPolicy matches the source policy on all 112 states."""
    demo = _load_demo()
    dict_policy = _build_dict_policy(demo)
    targets = (None, *range(7))
    checked_states = 0
    for tank_x in range(7):
        for target_x in targets:
            for cooldown in (0, 1):
                expected_values = demo._action_values(tank_x, target_x, cooldown)
                row_id = demo.state_row_id(tank_x, target_x, cooldown)
                decision = dict_policy.read(row_id)
                actual_values = tuple(decision.candidates[a] for a in demo.ACTIONS)
                assert actual_values == pytest.approx(expected_values)
                assert decision.action == demo.ACTIONS[
                    max(range(len(demo.ACTIONS)),
                        key=lambda i: (expected_values[i], -i))
                ]
                checked_states += 1
    assert checked_states == 112


def test_dict_policy_clears_all_waves() -> None:
    """DictPolicy clears all 2,401 ordered four‑target waves (identical to source)."""
    from itertools import product

    demo = _load_demo()
    dict_policy = _build_dict_policy(demo)
    for wave in product(range(7), repeat=4):
        config = demo.ShooterConfig(width=7, wave=wave, max_steps=32)
        game = demo.TinyArcadeShooter(config)
        while not game.done:
            row_id = game.row_id()
            decision = dict_policy.read(row_id)
            game.step(decision.action)
        assert game.cleared is True
        assert game.score == len(wave)


def test_dict_policy_mutation_changes_identity() -> None:
    """A one‑cell change alters the dict policy's SHA‑256 identity."""
    demo = _load_demo()
    original = _build_dict_policy(demo)

    # Behavioural mutation: change STAY in aligned state to 1.0
    mut_map = dict(original._map)  # shallow copy, values are tuples -> OK
    state_key = demo.state_row_id(3, 3, 0)
    values = list(mut_map[state_key])
    values[demo.ACTIONS.index("STAY")] = 1.0
    mut_map[state_key] = tuple(values)
    mutated = DictPolicy(mut_map, demo.ACTIONS)

    assert mutated.artifact_id != original.artifact_id
    assert original.read(state_key).action == "FIRE"
    assert mutated.read(state_key).action == "STAY"


def test_dict_policy_pickle_roundtrip(tmp_path: Path) -> None:
    """Pickle preserves identity and behaviour."""
    demo = _load_demo()
    policy = _build_dict_policy(demo)
    path = tmp_path / "dict_policy.pkl"
    with open(path, "wb") as f:
        pickle.dump(policy, f)
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded.artifact_id == policy.artifact_id
    # spot‑check a few states
    for row_id in ["tank=3|target=0|cooldown=0", "tank=0|target=0|cooldown=0"]:
        assert loaded.read(row_id).action == policy.read(row_id).action


def test_performance_comparison() -> None:
    """Report wall‑time and memory for VPM vs. plain dict."""
    demo = _load_demo()
    vpm_artifact = demo.compile_policy_artifact()
    vpm_reader = VPMPolicyLookup(vpm_artifact, action_metric_ids=demo.ACTIONS)
    dict_policy = _build_dict_policy(demo)

    # All 112 state row_ids
    row_ids = [
        demo.state_row_id(tank_x, target_x, cooldown)
        for tank_x in range(7)
        for target_x in (None, *range(7))
        for cooldown in (0, 1)
    ]
    N = 10_000  # number of full passes

    # Warm up
    for _ in range(100):
        for rid in row_ids:
            vpm_reader.read(rid)
            dict_policy.read(rid)

    # Time VPM
    start = time.perf_counter()
    for _ in range(N):
        for rid in row_ids:
            vpm_reader.read(rid)
    vpm_time = time.perf_counter() - start

    # Time Dict
    start = time.perf_counter()
    for _ in range(N):
        for rid in row_ids:
            dict_policy.read(rid)
    dict_time = time.perf_counter() - start

    # Memory (rough, only a hint)
    vpm_size = sys.getsizeof(vpm_artifact) + sys.getsizeof(vpm_reader)
    dict_size = sys.getsizeof(dict_policy) + sys.getsizeof(dict_policy._map)

    print("\n--- Performance comparison (112‑state shooter) ---")
    print(f"Total lookups: {N * len(row_ids):,}")
    print(f"VPM total time:        {vpm_time:.4f} s")
    print(f"DictPolicy total time: {dict_time:.4f} s")
    print(f"Ratio (VPM/Dict):      {vpm_time / dict_time:.3f}")
    print(f"Approx VPM memory (bytes):   {vpm_size}")
    print(f"Approx Dict memory (bytes):  {dict_size}")
    print("--------------------------------------------------\n")

    # Optional: assert that VPM isn't absurdly slower (e.g., < 10x)
    # Adjust factor as you see fit; the point is that overhead is small.
    assert vpm_time / dict_time < 20, "VPM overhead is unexpectedly high"