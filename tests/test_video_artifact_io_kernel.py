import json
import os
from pathlib import Path

import pytest

from zeromodel.video.domains.video_action_set import artifact_io


def test_json_jsonl_and_csv_bytes_match_pre_extraction_goldens(tmp_path: Path) -> None:
    payload = {
        "none": None,
        "boolean": True,
        "integer": 7,
        "float": 1.25,
        "mapping": {"z": 2, "a": 1},
        "list": [3, 2, 1],
        "tuple": ("x", False),
    }
    json_path = tmp_path / "representative.json"
    jsonl_path = tmp_path / "multiple.jsonl"
    csv_path = tmp_path / "representative.csv"

    artifact_io._write_json(json_path, payload)
    artifact_io._write_jsonl(jsonl_path, [{"b": 2, "a": 1}, {"nested": {"z": [1, 2]}}])
    artifact_io._write_csv(
        csv_path,
        [
            {"z": None, "a": True, "nested": {"b": 2, "a": 1}},
            {"a": False, "middle": "comma,value", "nested": [1, 2]},
        ],
    )

    line_ending = os.linesep.encode()
    expected_json = b'{\n  "boolean": true,\n  "float": 1.25,\n  "integer": 7,\n  "list": [\n    3,\n    2,\n    1\n  ],\n  "mapping": {\n    "a": 1,\n    "z": 2\n  },\n  "none": null,\n  "tuple": [\n    "x",\n    false\n  ]\n}\n'
    expected_jsonl = b'{"a": 1, "b": 2}\n{"nested": {"z": [1, 2]}}\n'
    assert json_path.read_bytes() == expected_json.replace(b"\n", line_ending)
    assert jsonl_path.read_bytes() == expected_jsonl.replace(b"\n", line_ending)
    assert (
        csv_path.read_bytes()
        == b'a,middle,nested,z\r\nTrue,,"{""a"": 1, ""b"": 2}",\r\nFalse,"comma,value","[1, 2]",\r\n'
    )


def test_empty_and_malformed_read_behavior_is_unchanged(tmp_path: Path) -> None:
    empty_jsonl = tmp_path / "empty.jsonl"
    empty_csv = tmp_path / "empty.csv"
    malformed = tmp_path / "malformed.jsonl"
    artifact_io._write_jsonl(empty_jsonl, [])
    artifact_io._write_csv(empty_csv, [])
    malformed.write_text('{"a": 1}\nnot-json\n{"b": 2}\n', encoding="utf-8")

    assert empty_jsonl.read_bytes() == b""
    assert empty_csv.read_bytes() == b"\r\n"
    assert artifact_io._read_jsonl(tmp_path / "missing.jsonl") == []
    with pytest.raises(FileNotFoundError) as missing:
        artifact_io._read_json(tmp_path / "missing.json")
    assert missing.value.args[:2] == (2, "No such file or directory")
    with pytest.raises(json.JSONDecodeError) as malformed_error:
        artifact_io._read_jsonl(malformed)
    assert malformed_error.value.args == ("Expecting value: line 1 column 1 (char 0)",)
