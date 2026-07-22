from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from architecture_rules import TRACKED_EXTERNAL_MODULES, print_violations

REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARIES = REPO_ROOT / "package-boundaries.toml"


@dataclass(frozen=True)
class ImportEdge:
    importer: str
    imported: str
    line: int


@dataclass(frozen=True)
class Violation:
    rule: str
    importer: str
    imported: str
    detail: str


def relative_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def module_name_for_path(path: Path, source_root: Path) -> str:
    parts = list(path.relative_to(source_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def package_name_for_module(module_name: str, path: Path) -> str:
    if path.name == "__init__.py":
        return module_name
    return module_name.rsplit(".", 1)[0]


def workspace_source_roots() -> dict[str, Path]:
    """Every production distribution's source root, keyed by package name.

    Read from package-boundaries.toml so this stays the single shared authority
    for "what is a production source root" (see scripts/check_package_boundaries.py,
    which reads the same manifest).
    """
    manifest = tomllib.loads(BOUNDARIES.read_text(encoding="utf-8"))
    return {
        name: REPO_ROOT / config["source_root"]
        for name, config in manifest["packages"].items()
    }


def discover_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for source_root in workspace_source_roots().values():
        for path in sorted(source_root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            modules[module_name_for_path(path, source_root)] = path
    return dict(sorted(modules.items()))


def resolve_relative_import(
    module_name: str,
    path: Path,
    level: int,
    imported: str | None,
) -> str:
    package = package_name_for_module(module_name, path)
    package_parts = package.split(".") if package else []
    if level > len(package_parts):
        return imported or ""
    base_parts = package_parts[: len(package_parts) - level + 1]
    if imported:
        base_parts.extend(imported.split("."))
    return ".".join(part for part in base_parts if part)


def best_known_module(candidate: str, known_modules: set[str]) -> str | None:
    parts = candidate.split(".")
    while parts:
        current = ".".join(parts)
        if current in known_modules:
            return current
        parts.pop()
    return None


def tracked_external_module(candidate: str) -> str | None:
    return next(
        (
            module
            for module in sorted(TRACKED_EXTERNAL_MODULES)
            if candidate == module or candidate.startswith(f"{module}.")
        ),
        None,
    )


def import_from_candidates(
    node: ast.ImportFrom, module_name: str, path: Path
) -> list[str]:
    if node.level:
        base = resolve_relative_import(module_name, path, node.level, node.module)
    else:
        base = node.module or ""
    if not base:
        return []
    if (
        base == "zeromodel"
        or base.startswith("zeromodel.")
        or base.startswith("tests")
        or tracked_external_module(base) is not None
    ):
        return (
            [base]
            if any(alias.name == "*" for alias in node.names)
            else [f"{base}.{alias.name}" for alias in node.names]
        )
    return []


def collect_import_edges(
    module_name: str, path: Path, known_modules: set[str]
) -> list[ImportEdge]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path(path))
    edges: list[ImportEdge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = best_known_module(alias.name, known_modules)
                tracked = tracked_external_module(alias.name)
                if imported is not None or alias.name.startswith("tests") or tracked:
                    edges.append(
                        ImportEdge(
                            importer=module_name,
                            imported=imported or tracked or alias.name,
                            line=node.lineno,
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            for candidate in import_from_candidates(node, module_name, path):
                imported = best_known_module(candidate, known_modules)
                tracked = tracked_external_module(candidate)
                if imported is not None or candidate.startswith("tests") or tracked:
                    edges.append(
                        ImportEdge(
                            importer=module_name,
                            imported=imported or tracked or candidate,
                            line=node.lineno,
                        )
                    )
    return sorted(edges, key=lambda edge: (edge.imported, edge.line))


def build_import_graph(
    modules: dict[str, Path],
) -> tuple[dict[str, set[str]], list[ImportEdge]]:
    known_modules = set(modules)
    all_edges: list[ImportEdge] = []
    graph: dict[str, set[str]] = {module: set() for module in modules}
    for module_name, path in modules.items():
        edges = collect_import_edges(module_name, path, known_modules)
        all_edges.extend(edges)
        for edge in edges:
            if edge.imported in known_modules:
                graph[module_name].add(edge.imported)
    return graph, sorted(
        all_edges, key=lambda edge: (edge.importer, edge.imported, edge.line)
    )


def strongly_connected_components(graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(module: str) -> None:
        nonlocal index
        indices[module] = index
        lowlinks[module] = index
        index += 1
        stack.append(module)
        on_stack.add(module)

        for imported in sorted(graph[module]):
            if imported not in indices:
                visit(imported)
                lowlinks[module] = min(lowlinks[module], lowlinks[imported])
            elif imported in on_stack:
                lowlinks[module] = min(lowlinks[module], indices[imported])

        if lowlinks[module] == indices[module]:
            component: list[str] = []
            while True:
                imported = stack.pop()
                on_stack.remove(imported)
                component.append(imported)
                if imported == module:
                    break
            components.append(sorted(component))

    for module in sorted(graph):
        if module not in indices:
            visit(module)
    return components


def cycle_path(component: list[str], graph: dict[str, set[str]]) -> list[str]:
    component_set = set(component)

    def search(
        start: str, current: str, path: list[str], visited: set[str]
    ) -> list[str] | None:
        for imported in sorted(graph[current] & component_set):
            if imported == start:
                return [*path, start]
            if imported not in visited:
                found = search(start, imported, [*path, imported], visited | {imported})
                if found is not None:
                    return found
        return None

    for module in sorted(component):
        found = search(module, module, [module], {module})
        if found is not None:
            return found
    return [*sorted(component), sorted(component)[0]]


def cycle_violations(graph: dict[str, set[str]]) -> list[Violation]:
    violations: list[Violation] = []
    for module, imports in sorted(graph.items()):
        if module in imports:
            violations.append(
                Violation(
                    rule="local import cycle",
                    importer=module,
                    imported=module,
                    detail=f"cycle: {module} -> {module}",
                )
            )

    for component in strongly_connected_components(graph):
        if len(component) <= 1:
            continue
        path = cycle_path(component, graph)
        violations.append(
            Violation(
                rule="local import cycle",
                importer=path[0],
                imported=path[1],
                detail="cycle: " + " -> ".join(path),
            )
        )
    return violations


def edge_violation(rule: str, edge: ImportEdge) -> Violation:
    return Violation(
        rule=rule,
        importer=edge.importer,
        imported=edge.imported,
        detail=f"forbidden edge at line {edge.line}",
    )


def forbidden_edge_violations(edges: list[ImportEdge]) -> list[Violation]:
    from architecture_rules.video_action_set import stage6_forbidden_edge_violations
    from architecture_rules.video_action_set_modules import (
        stage7a_forbidden_edge_violations,
    )
    from architecture_rules.video_science_layers import (
        video_science_forbidden_edge_violations,
    )
    from architecture_rules.video_verification_layers import (
        stage7b_forbidden_edge_violations,
    )
    from architecture_rules.video_instrument_shell import (
        video_instrument_shell_forbidden_edge_violations,
    )

    violations: list[Violation] = []
    for edge in edges:
        violations.extend(video_science_forbidden_edge_violations(edge, edge_violation))
        violations.extend(stage6_forbidden_edge_violations(edge, edge_violation))
        violations.extend(stage7a_forbidden_edge_violations(edge, edge_violation))
        violations.extend(stage7b_forbidden_edge_violations(edge, edge_violation))
        violations.extend(
            video_instrument_shell_forbidden_edge_violations(edge, edge_violation)
        )
    return violations


def main() -> int:
    modules = discover_modules()
    if not modules:
        raise SystemExit(
            "Architecture check found zero production modules - "
            "package-boundaries.toml source roots are missing or misconfigured"
        )
    graph, edges = build_import_graph(modules)
    violations = [*cycle_violations(graph), *forbidden_edge_violations(edges)]
    if violations:
        print_violations(violations)
        return 1
    print(f"Architecture check: passed ({len(modules)} production modules inspected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
