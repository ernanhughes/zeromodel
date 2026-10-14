import json

import pytest

from examples.hand_anatomy.cli import build_parser, main


def test_help_exits_zero_and_lists_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("build-library", "inspect", "benchmark", "replay", "validate-report"):
        assert cmd in out
    assert build_parser().parse_args(["inspect", "--mask", "m.png", "--output", "o"]).mask == "m.png"


def test_library_fixture_inspect_validate_replay_flow(tmp_path, capsys):
    lib, fx, out = tmp_path / "lib", tmp_path / "fx", tmp_path / "out"
    assert main(["build-library", "--output", str(lib)]) == 0
    assert main(["verify-library", "--library", str(lib)]) == 0
    assert main(["inspect-library", "--library", str(lib)]) == 0
    assert main(["make-fixtures", "--output", str(fx)]) == 0
    mask = str(fx / "six_finger_hand-mask.png")
    assert main(["inspect", "--mask", mask, "--library", str(lib), "--output", str(out)]) == 0
    report = out / "report.json"
    assert json.loads(report.read_text())["decision"] in ("suspected_anomaly", "insufficient_evidence", "supported_plausible")
    assert main(["validate-report", "--report", str(report)]) == 0
    assert main(["replay", "--report", str(report), "--mask", mask, "--library", str(lib)]) == 0
    other = str(fx / "plausible_open_hand-mask.png")
    assert main(["replay", "--report", str(report), "--mask", other, "--library", str(lib)]) == 1
    assert (out / "08_vpm_annotated.png").is_file()


def test_exit_codes_for_bad_input(tmp_path):
    assert main(["inspect", "--mask", str(tmp_path / "nope.png"), "--output", str(tmp_path / "o")]) == 2
    assert main(["validate-report", "--report", str(tmp_path / "nope.json")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "hand-anatomy-report/1", "decision": "supported_plausible"}))
    assert main(["validate-report", "--report", str(bad)]) == 1
