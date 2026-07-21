from pathlib import Path
from types import SimpleNamespace
from typing import Any

from zeromodel.domains.video_action_set import build_orchestration as build


def test_freeze_benchmark_preserves_compile_save_seal_write_order(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    class FakeDTO:
        def __init__(self, payload: dict[str, Any]):
            self.payload = payload

        @classmethod
        def from_dict(cls, payload: dict[str, Any]):
            return cls(payload)

        def to_dict(self) -> dict[str, Any]:
            return self.payload

    identity = SimpleNamespace(seed_digest="seed", seed_material="material")

    class VideoService:
        def load_identity(self, _repo_root: Path):
            calls.append("identity")
            return identity

        def save_episode_plans(self, plans):
            calls.append("save")
            return plans

        def seal_final_split(self, **_kwargs):
            calls.append("seal")
            return FakeDTO({"sealed_plan_digest": "sealed"})

    monkeypatch.setattr(
        build,
        "_build_durable_runtime",
        lambda _path: (
            calls.append("runtime") or SimpleNamespace(video_action_set=VideoService())
        ),
    )
    monkeypatch.setattr(
        build,
        "compile_policy_artifact",
        lambda: (
            calls.append("policy")
            or SimpleNamespace(
                artifact_id="policy", source=SimpleNamespace(row_ids=("r0",))
            )
        ),
    )
    monkeypatch.setattr(
        build,
        "VPMPolicyLookup",
        lambda *_args, **_kwargs: SimpleNamespace(choose=lambda _row: "LEFT"),
    )
    monkeypatch.setattr(build, "EpisodePlanDTO", FakeDTO)
    monkeypatch.setattr(
        build,
        "_episode_plans_for_split",
        lambda _identity, split, _rows, _actions: (
            calls.append(f"plan:{split}") or [{"split": split}]
        ),
    )
    monkeypatch.setattr(
        build,
        "_validate_episode_plan_collection",
        lambda *_args: calls.append("validate"),
    )
    monkeypatch.setattr(
        build,
        "_write_frozen_contract_artifacts",
        lambda *_args: calls.append("write:contract"),
    )
    monkeypatch.setattr(
        build, "_write_frozen_plan_artifacts", lambda *_args: calls.append("write:plan")
    )

    payload = build.freeze_benchmark(tmp_path, tmp_path)

    assert calls == [
        "runtime",
        "identity",
        "policy",
        "plan:development",
        "plan:calibration",
        "plan:selection",
        "plan:final",
        "validate",
        "save",
        "save",
        "save",
        "save",
        "seal",
        "write:contract",
        "write:plan",
    ]
    assert payload["final_total_expected_frame_count"] == 1008


def test_profile_runtime_preserves_provider_and_write_order(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(build, "canonical_prototypes", lambda: {})
    monkeypatch.setattr(
        build, "compile_policy_artifact", lambda: SimpleNamespace(artifact_id="policy")
    )
    monkeypatch.setattr(
        build, "_profiling_records", lambda *_args: [{"frame_id": "f0"}]
    )
    monkeypatch.setattr(
        build,
        "_profile_provider",
        lambda *, provider_id, implementation, **_kwargs: (
            calls.append(f"score:{implementation}:{provider_id}")
            or {
                "provider_id": provider_id,
                "mean_seconds_per_frame": 1.0,
                "frame_count": 1,
            }
        ),
    )
    monkeypatch.setattr(
        build, "runtime_profile_payload", lambda **_kwargs: {"payload": True}
    )
    monkeypatch.setattr(
        build, "_write_json", lambda path, _payload: calls.append(f"json:{path.name}")
    )
    monkeypatch.setattr(
        build, "_write_text", lambda path, _text: calls.append(f"text:{path.name}")
    )

    assert build.profile_runtime(tmp_path, tmp_path) == {"payload": True}
    assert calls == [
        "score:reference:P1",
        "score:optimized:P1",
        "score:reference:P2",
        "score:optimized:P2",
        "score:reference:P3",
        "score:optimized:P3",
        "json:runtime-profile-reference.json",
        "json:runtime-profile-optimized.json",
        "json:runtime-comparison.json",
        "text:runtime-profile-reference.md",
        "text:runtime-profile-optimized.md",
    ]
