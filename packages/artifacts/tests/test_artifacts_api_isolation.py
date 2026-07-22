from __future__ import annotations

import os
import subprocess
import sys

import zeromodel.artifacts as artifacts


def test_artifacts_public_api_is_deliberate() -> None:
    assert hasattr(artifacts, "__all__")
    assert "ArtifactRef" in artifacts.__all__
    assert "ArtifactStore" in artifacts.__all__
    assert "InMemoryArtifactStore" in artifacts.__all__
    assert "ReportAdapter" in artifacts.__all__
    assert "ReportAdapterContractDTO" in artifacts.__all__
    assert "AdaptedReportDTO" in artifacts.__all__
    assert "CompiledReportArtifactDTO" in artifacts.__all__
    assert "compile_report" in artifacts.__all__
    assert "load_compiled_report_artifact" in artifacts.__all__
    assert "ScoreTable" not in artifacts.__all__
    assert "VideoPolicyReader" not in artifacts.__all__
    # No domain-specific adapter may ever be defined in this package - those
    # belong to the external application (see packages/artifacts/README.md).
    assert "AIArtifactReportAdapter" not in artifacts.__all__
    assert "WriterSentenceQualityReportAdapter" not in artifacts.__all__


def test_artifacts_import_loads_core_but_no_forbidden_siblings() -> None:
    script = r"""
import json
import sys

import zeromodel.artifacts

forbidden = [
    "zeromodel.analysis",
    "zeromodel.observation",
    "zeromodel.vision",
    "zeromodel.video",
    "zeromodel.trust",
    "zeromodel.navigation",
    "zeromodel.persistence",
    "sqlalchemy",
    "torch",
    "torchvision",
    "transformers",
    "PIL",
]

loaded_forbidden = [
    name
    for name in forbidden
    if name in sys.modules
    or any(module.startswith(name + ".") for module in sys.modules)
]

core_loaded = (
    "zeromodel.core" in sys.modules
    or any(module.startswith("zeromodel.core.") for module in sys.modules)
)

print(json.dumps({"core_loaded": core_loaded, "forbidden": loaded_forbidden}))
raise SystemExit(1 if loaded_forbidden or not core_loaded else 0)
"""
    args = [sys.executable, "-c", script]
    if os.environ.get("ARTIFACTS_WHEEL_PATH"):
        args.insert(1, "-I")
    result = subprocess.run(args, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_artifacts_wheel_contains_only_artifacts_namespace_when_path_is_provided() -> (
    None
):
    wheel_path = os.environ.get("ARTIFACTS_WHEEL_PATH")
    if not wheel_path:
        return
    import zipfile

    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
    for name in names:
        if (
            name.endswith(".dist-info/")
            or "/.dist-info/" in name
            or name.endswith(
                (
                    ".dist-info/METADATA",
                    ".dist-info/RECORD",
                    ".dist-info/WHEEL",
                    ".dist-info/top_level.txt",
                )
            )
        ):
            continue
        assert name.startswith("zeromodel/artifacts/"), name
