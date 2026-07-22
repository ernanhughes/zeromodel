from pathlib import Path

from zeromodel.video.domains.video_action_set import (
    verification_orchestration as verification,
)


def test_reference_verification_preserves_gate_order(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        verification, "_reference_context", lambda _root: {"context": True}
    )
    monkeypatch.setattr(
        verification, "_measured_phase_access_counts", lambda _root: {"counts": True}
    )
    monkeypatch.setattr(
        verification, "_build_reference_verification_payload", lambda **kwargs: kwargs
    )

    names = (
        "structural_identity",
        "semantic_outcome",
        "seed_and_plan",
        "episode_regeneration",
        "family_contract",
        "reachability",
        "completeness_orphan",
        "access_prohibition",
    )
    for name in names:
        monkeypatch.setattr(
            verification,
            f"_{name}_gate",
            lambda *_args, _name=name, **_kwargs: (
                calls.append(_name) or {"gate": _name, "status": "passed"}
            ),
        )

    payload = verification.verify_reference_instrument(tmp_path, tmp_path)

    assert calls == list(names)
    assert [gate["gate"] for gate in payload["gates"]] == list(names)
    assert payload["phase_counts"] == {"counts": True}


def test_stop_after_first_failure_short_circuits_later_gates(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(verification, "_reference_context", lambda _root: {})
    monkeypatch.setattr(verification, "_measured_phase_access_counts", lambda _root: {})
    monkeypatch.setattr(
        verification, "_build_reference_verification_payload", lambda **kwargs: kwargs
    )
    monkeypatch.setattr(
        verification,
        "_structural_identity_gate",
        lambda *_args, **_kwargs: calls.append("structural") or {"status": "failed"},
    )
    monkeypatch.setattr(
        verification,
        "_semantic_outcome_gate",
        lambda *_args, **_kwargs: calls.append("semantic"),
    )

    verification.verify_reference_instrument(
        tmp_path, tmp_path, stop_after_first_failure=True
    )
    assert calls == ["structural"]
