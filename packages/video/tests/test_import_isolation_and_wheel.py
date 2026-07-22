from __future__ import annotations

import os
import subprocess
import sys
import zipfile


FORBIDDEN = {
    "zeromodel.analysis",
    "zeromodel.vision",
    "zeromodel.persistence",
    "sqlalchemy",
    "torch",
    "torchvision",
    "transformers",
    "PIL",
    "research",
}


def test_video_import_avoids_forbidden_dependencies() -> None:
    code = """
import sys
import zeromodel.video
for name in sorted(sys.modules):
    if name in %r or any(name.startswith(item + ".") for item in %r):
        raise SystemExit(name)
print("ok")
""" % (FORBIDDEN, FORBIDDEN)
    env = os.environ.copy()
    wheel = os.environ.get("VIDEO_WHEEL_PATH")
    if not wheel:
        env["PYTHONPATH"] = os.pathsep.join(
            [
                "packages/video/src",
                "packages/observation/src",
                "packages/core/src",
            ]
        )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    assert completed.stdout.strip() == "ok"


def test_video_wheel_contains_only_video_namespace_when_path_is_provided() -> None:
    wheel = os.environ.get("VIDEO_WHEEL_PATH")
    if not wheel:
        return
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    assert "zeromodel/video/__init__.py" in names
    assert "zeromodel/__init__.py" not in names
    assert all(
        name.startswith("zeromodel/video/")
        or name.startswith("zeromodel_video-1.0.13.dist-info/")
        for name in names
    )
    assert not any(
        name.startswith(prefix)
        for name in names
        for prefix in (
            "zeromodel/core/",
            "zeromodel/analysis/",
            "zeromodel/observation/",
            "zeromodel/vision/",
            "zeromodel/persistence/",
            "tests/",
            "research/",
            "examples/",
            "docs/",
            "scripts/",
        )
    )
