from pathlib import Path

from research.video_action_set import mutation_orchestration as mutation


CASE = {
    "mutation_id": "policy_alter_artifact_identity",
    "validation_metadata": {"gate_scope": ["structural_identity"]},
}


def test_baseline_failure_short_circuits_mutation_execution(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mutation, "mutation_catalogue", lambda: (CASE,))
    monkeypatch.setattr(mutation, "validate_mutation_catalogue", lambda: [])
    monkeypatch.setattr(
        mutation,
        "verify_reference_instrument",
        lambda *_args, **_kwargs: {
            "verified": False,
            "primary_failure_code": "base_failed",
        },
    )
    monkeypatch.setattr(
        mutation,
        "_directory_snapshot",
        lambda _path: (_ for _ in ()).throw(AssertionError("snapshot must not run")),
    )
    monkeypatch.setattr(
        mutation, "_build_mutation_audit_payload", lambda **kwargs: kwargs
    )

    payload = mutation.run_reference_mutation_audit(tmp_path, tmp_path)
    assert payload["base_verified"] is False
    assert payload["results"] == ()


def test_application_error_uses_historical_broad_boundary(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "baseline.txt").write_text("baseline", encoding="utf-8")
    monkeypatch.setattr(mutation, "mutation_catalogue", lambda: (CASE,))
    monkeypatch.setattr(mutation, "validate_mutation_catalogue", lambda: [])
    monkeypatch.setattr(
        mutation,
        "verify_reference_instrument",
        lambda *_args, **_kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        mutation,
        "_apply_reference_mutation",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )
    monkeypatch.setattr(
        mutation,
        "_evaluate_mutation_case",
        lambda **kwargs: {"application_error": kwargs["application_error"]},
    )
    monkeypatch.setattr(
        mutation, "_build_mutation_audit_payload", lambda **kwargs: kwargs
    )

    payload = mutation.run_reference_mutation_audit(tmp_path, tmp_path)
    assert payload["results"] == [{"application_error": "RuntimeError"}]
