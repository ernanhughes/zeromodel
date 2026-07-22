from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARIES = REPO_ROOT / "package-boundaries.toml"


@dataclass(frozen=True)
class ModuleRecord:
    module: str
    path: Path
    package: str
    source_root: Path


@dataclass(frozen=True)
class ImportEdge:
    importer: str
    imported: str
    line: int
    kind: str


def load_manifest() -> dict[str, object]:
    return tomllib.loads(BOUNDARIES.read_text(encoding="utf-8"))


def module_name(path: Path, source_root: Path) -> str:
    parts = list(path.relative_to(source_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def package_name(module: str, path: Path) -> str:
    if path.name == "__init__.py":
        return module
    return module.rsplit(".", 1)[0]


def discover_modules(manifest: dict[str, object]) -> dict[str, ModuleRecord]:
    modules: dict[str, ModuleRecord] = {}
    packages = manifest["packages"]
    for package, config in packages.items():
        root = REPO_ROOT / config["source_root"]
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            module = module_name(path, root)
            if module in modules:
                raise SystemExit(
                    f"Duplicate module {module}: {path} and {modules[module].path}"
                )
            modules[module] = ModuleRecord(module, path, package, root)
    return modules


def resolve_relative(record: ModuleRecord, level: int, imported: str | None) -> str:
    parts = package_name(record.module, record.path).split(".")
    base = parts[: max(0, len(parts) - level + 1)]
    if imported:
        base.extend(imported.split("."))
    return ".".join(base)


def best_known(candidate: str, known: set[str]) -> str | None:
    parts = candidate.split(".")
    while parts:
        current = ".".join(parts)
        if current in known:
            return current
        parts.pop()
    return None


def import_kind(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    kind = "runtime"
    current: ast.AST | None = node
    while current in parents:
        current = parents[current]
        if (
            isinstance(current, ast.If)
            and isinstance(current.test, ast.Name)
            and current.test.id == "TYPE_CHECKING"
        ):
            return "type-checking"
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "deferred"
        if isinstance(current, ast.Try):
            kind = "optional"
    return kind


def collect_edges(record: ModuleRecord, known: set[str]) -> list[ImportEdge]:
    tree = ast.parse(record.path.read_text(encoding="utf-8"), filename=str(record.path))
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    edges: list[ImportEdge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if target := best_known(alias.name, known):
                    edges.append(
                        ImportEdge(
                            record.module,
                            target,
                            node.lineno,
                            import_kind(node, parents),
                        )
                    )
                elif alias.name == "research" or alias.name.startswith("research."):
                    edges.append(
                        ImportEdge(
                            record.module,
                            alias.name,
                            node.lineno,
                            import_kind(node, parents),
                        )
                    )
                elif alias.name == "tests" or alias.name.startswith("tests."):
                    edges.append(
                        ImportEdge(
                            record.module,
                            alias.name,
                            node.lineno,
                            import_kind(node, parents),
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            base = (
                resolve_relative(record, node.level, node.module)
                if node.level
                else (node.module or "")
            )
            candidates = (
                [base]
                if any(alias.name == "*" for alias in node.names)
                else [f"{base}.{alias.name}" for alias in node.names]
            )
            for candidate in candidates:
                if target := best_known(candidate, known):
                    edges.append(
                        ImportEdge(
                            record.module,
                            target,
                            node.lineno,
                            import_kind(node, parents),
                        )
                    )
                elif base == "research" or base.startswith("research."):
                    edges.append(
                        ImportEdge(
                            record.module, base, node.lineno, import_kind(node, parents)
                        )
                    )
                elif base == "tests" or base.startswith("tests."):
                    edges.append(
                        ImportEdge(
                            record.module, base, node.lineno, import_kind(node, parents)
                        )
                    )
    return sorted(set(edges), key=lambda e: (e.importer, e.imported, e.line, e.kind))


def dependency_cycles(packages: dict[str, object]) -> list[list[str]]:
    graph = {
        name: set(config.get("depends_on", [])) for name, config in packages.items()
    }
    cycles: list[list[str]] = []

    def visit(start: str, current: str, path: list[str]) -> None:
        for dep in sorted(graph[current]):
            if dep == start:
                cycles.append([*path, dep])
            elif dep not in path:
                visit(start, dep, [*path, dep])

    for package in sorted(graph):
        visit(package, package, [package])
    return cycles


def validate_metadata(manifest: dict[str, object]) -> list[str]:
    errors: list[str] = []
    version = manifest["release_version"]
    for name, config in manifest["packages"].items():
        pyproject = REPO_ROOT / name.replace("sqlalchemy", "sqlalchemy")
        pyproject = REPO_ROOT / "packages" / name / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        if data["project"]["name"] != config["distribution"]:
            errors.append(f"{pyproject}: distribution name mismatch")
        if data["project"]["version"] != version:
            errors.append(f"{pyproject}: version is not {version}")
    if (REPO_ROOT / "zeromodel" / "__init__.py").exists():
        errors.append("old root zeromodel/__init__.py still exists")
    if (
        any((REPO_ROOT / "zeromodel").rglob("*.py"))
        if (REPO_ROOT / "zeromodel").exists()
        else False
    ):
        errors.append("old root zeromodel tree still contains Python modules")
    return errors


def main() -> int:
    manifest = load_manifest()
    modules = discover_modules(manifest)
    known = set(modules)
    allowed = {
        name: set(config.get("depends_on", []))
        for name, config in manifest["packages"].items()
    }
    errors = validate_metadata(manifest)
    for cycle in dependency_cycles(manifest["packages"]):
        errors.append("package dependency cycle: " + " -> ".join(cycle))

    for record in modules.values():
        for edge in collect_edges(record, known):
            if edge.imported.startswith("research") or edge.imported.startswith(
                "tests"
            ):
                errors.append(
                    f"{record.path.relative_to(REPO_ROOT).as_posix()}:{edge.line}: production import of {edge.imported}"
                )
                continue
            imported_record = modules.get(edge.imported)
            if imported_record and imported_record.package != record.package:
                if imported_record.package not in allowed.get(record.package, set()):
                    errors.append(
                        f"{record.path.relative_to(REPO_ROOT).as_posix()}:{edge.line}: "
                        f"forbidden {record.package} -> {imported_record.package} import "
                        f"({edge.importer} -> {edge.imported})"
                    )
    if errors:
        print("Package boundary check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Package boundary check passed: {len(modules)} production modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
