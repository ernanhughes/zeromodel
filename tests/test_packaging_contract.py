from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path("scripts/validate_packaging_contract.py")
SPEC = importlib.util.spec_from_file_location("validate_packaging_contract", SCRIPT)
assert SPEC is not None
packaging = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = packaging
SPEC.loader.exec_module(packaging)


def test_manifest_defines_runtime_packages_and_metadata_only_umbrella() -> None:
    manifest = packaging.load_manifest()
    packages = manifest["packages"]

    assert packages["core"]["distribution"] == "zeromodel-core"
    assert packages["core"]["namespace"] == "zeromodel.core"
    assert packages["meta"]["kind"] == "meta"
    assert packages["meta"]["distribution"] == "zeromodel"
    assert packages["meta"]["namespace"] == ""
    assert packages["meta"]["owned_prefixes"] == []
    assert set(packages["meta"]["depends_on"]) == set(
        packaging.runtime_package_keys(manifest)
    )
    assert "meta" not in packages["meta"]["depends_on"]


def test_implementation_packages_do_not_depend_on_umbrella() -> None:
    manifest = packaging.load_manifest()
    version = packaging.read_version()

    for key in packaging.runtime_package_keys(manifest):
        pyproject = packaging.tomllib.loads(
            (packaging.package_root(key, manifest) / "pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        dependencies = set(pyproject["project"].get("dependencies", []))
        if key == "core":
            assert f"zeromodel-core=={version}" not in dependencies
        else:
            assert f"zeromodel=={version}" not in dependencies
            assert f"zeromodel-core=={version}" in dependencies


def test_expected_internal_requirements_are_exact_coordinated_pins() -> None:
    manifest = packaging.load_manifest()
    version = packaging.read_version()
    distributions = {
        key: config["distribution"] for key, config in manifest["packages"].items()
    }

    for key in packaging.publishable_package_keys(manifest):
        expected = packaging.expected_internal_requirement_set(key, manifest)
        assert all(requirement.endswith(f"=={version}") for requirement in expected)
        assert expected == {
            f"{distributions[dependency]}=={version}"
            for dependency in manifest["packages"][key].get("depends_on", [])
        }


def test_runtime_dependency_graph_is_acyclic() -> None:
    manifest = packaging.load_manifest()
    runtime = set(packaging.runtime_package_keys(manifest))
    graph = {
        key: set(manifest["packages"][key].get("depends_on", [])) & runtime
        for key in runtime
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        assert node not in visiting, f"cycle at {node}"
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(runtime):
        visit(node)
