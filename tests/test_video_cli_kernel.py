import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import research.video_action_set.video_action_set_cli as cli


REPO_ROOT = Path(__file__).resolve().parents[1]


def _source_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        str(REPO_ROOT / path)
        for path in (
            "packages/core/src",
            "packages/observation/src",
            "packages/video/src",
        )
    )
    return env


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("--freeze-benchmark", ["freeze"]),
        ("--build-development", ["freeze", "build:development"]),
        ("--build-calibration", ["freeze", "build:calibration"]),
        ("--build-selection", ["freeze", "build:selection"]),
        ("--audit-evidence-completeness", ["evidence"]),
        ("--audit-canonical-providers", ["canonical"]),
        ("--profile-runtime", ["profile:all:8"]),
        ("--verify-provider-runtime-equivalence", ["equivalence"]),
        ("--verify-prospective-instrument", ["verify"]),
    ],
)
def test_cli_dispatches_each_historical_flag(
    monkeypatch, capsys, flag: str, expected: list[str]
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "freeze_benchmark",
        lambda *_args: calls.append("freeze") or {"command": "freeze"},
    )
    monkeypatch.setattr(
        cli,
        "build_split",
        lambda split, *_args, **_kwargs: (
            calls.append(f"build:{split}") or {"command": split}
        ),
    )
    monkeypatch.setattr(
        cli,
        "audit_evidence_completeness",
        lambda *_args: calls.append("evidence") or {"command": "evidence"},
    )
    monkeypatch.setattr(
        cli,
        "audit_canonical_providers",
        lambda *_args: calls.append("canonical") or {"command": "canonical"},
    )
    monkeypatch.setattr(
        cli,
        "profile_runtime",
        lambda *_args, provider, frame_count: (
            calls.append(f"profile:{provider}:{frame_count}") or {"command": "profile"}
        ),
    )
    monkeypatch.setattr(
        cli,
        "verify_provider_runtime_equivalence",
        lambda *_args: calls.append("equivalence") or {"command": "equivalence"},
    )
    monkeypatch.setattr(
        cli,
        "verify_instrument",
        lambda *_args: calls.append("verify") or {"command": "verify"},
    )
    monkeypatch.setattr(sys, "argv", ["instrument", flag])

    cli.main()

    assert calls == expected
    assert json.loads(capsys.readouterr().out)["command"]


def test_cli_rejects_no_command_and_unknown_options(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["instrument"])
    with pytest.raises(SystemExit, match="exactly one command is required"):
        cli.main()
    monkeypatch.setattr(sys, "argv", ["instrument", "--unknown-command"])
    with pytest.raises(SystemExit) as unknown:
        cli.main()
    assert unknown.value.code == 2
    assert "unrecognized arguments: --unknown-command" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag",
    ["--build-development", "--build-calibration", "--build-selection"],
)
def test_cli_accepts_progress_for_split_builds_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flag: str,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "freeze_benchmark",
        lambda *_args: calls.append("freeze") or {"command": "freeze"},
    )

    def fake_build_split(split: str, *_args, progress_observer=None) -> dict[str, str]:
        assert progress_observer is not None
        calls.append(f"build:{split}")
        return {"command": split}

    monkeypatch.setattr(cli, "build_split", fake_build_split)
    monkeypatch.setattr(sys, "argv", ["instrument", flag, "--progress"])

    cli.main()

    assert calls == ["freeze", f"build:{flag.removeprefix('--build-')}"]
    assert json.loads(capsys.readouterr().out)["command"]


@pytest.mark.parametrize(
    "flag",
    [
        "--freeze-benchmark",
        "--profile-runtime",
        "--audit-evidence-completeness",
        "--audit-canonical-providers",
    ],
)
def test_cli_rejects_progress_for_non_build_operations(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["instrument", flag, "--progress"])

    with pytest.raises(SystemExit, match="--progress is only valid"):
        cli.main()


def test_cli_rejects_progress_without_explicit_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["instrument", "--progress"])

    with pytest.raises(SystemExit, match="--progress is only valid"):
        cli.main()


def test_removed_root_benchmark_module_is_not_a_compatibility_surface(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "must-not-exist"
    invocations = (
        [sys.executable, "-m", "zeromodel.video_action_set_benchmark"],
        [
            sys.executable,
            "-m",
            "zeromodel.video_action_set_benchmark",
            "--output-dir",
            str(output_dir),
            "--freeze-benchmark",
        ],
    )
    for invocation in invocations:
        result = subprocess.run(
            invocation,
            cwd=REPO_ROOT,
            env=_source_env(),
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert result.stdout == ""
        assert "No module named zeromodel.video_action_set_benchmark" in result.stderr
    assert not output_dir.exists()


def test_research_cli_module_exposes_operational_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "research.video_action_set.video_action_set_cli", "--help"],
        cwd=REPO_ROOT,
        env=_source_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--freeze-benchmark" in result.stdout
    assert "--build-development" in result.stdout
    assert result.stderr == ""


def test_stage8_runbook_freezes_once_then_uses_direct_build_calls() -> None:
    runbook = (REPO_ROOT / "docs" / "video-action-set-stage8-runbook.md").read_text(
        encoding="utf-8"
    )
    controlled = runbook.split("### Controlled Stage 8 Sequence", 1)[1].split(
        "### Historical One-Shot Build Flags", 1
    )[0]

    assert controlled.count("--freeze-benchmark") == 1
    assert "--build-development" not in controlled
    assert "--build-calibration" not in controlled
    assert "--build-selection" not in controlled
    for split in ("development", "calibration", "selection"):
        assert f'build_split("{split}", out, repo)' in controlled
