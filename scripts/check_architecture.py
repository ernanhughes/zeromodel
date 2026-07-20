from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "zeromodel"
VIDEO_ACTION_SET_PREFIX = "zeromodel.video_action_set"
VIDEO_ACTION_SET_BENCHMARK_MODULE = "zeromodel.video_action_set_benchmark"
DOMAIN_VIDEO_ACTION_SET_PREFIX = "zeromodel.domains.video_action_set"
DB_ORM_PREFIX = "zeromodel.db.orm"
DB_STORES_PREFIX = "zeromodel.db.stores"
VIDEO_ACTION_SET_ORCHESTRATION_MODULES = {
    "zeromodel.domains.video_action_set.episode_plan_service",
    "zeromodel.domains.video_action_set.identity_service",
    "zeromodel.domains.video_action_set.engine",
    "zeromodel.domains.video_action_set.facade",
    "zeromodel.runtime",
}
VIDEO_ACTION_SET_PURE_DOMAIN_MODULES = {
    "zeromodel.domains.video_action_set.canonical_json",
    "zeromodel.domains.video_action_set.contracts",
    "zeromodel.domains.video_action_set.dto",
    "zeromodel.domains.video_action_set.episode_plan_service",
}
VIDEO_ACTION_SET_POLICY_MODULES = {
    f"{VIDEO_ACTION_SET_PREFIX}.contracts",
    f"{VIDEO_ACTION_SET_PREFIX}.mutation_audit",
    f"{VIDEO_ACTION_SET_PREFIX}.verification",
}


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


def module_name_for_path(path: Path) -> str:
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def package_name_for_module(module_name: str, path: Path) -> str:
    if path.name == "__init__.py":
        return module_name
    return module_name.rsplit(".", 1)[0]


def discover_modules() -> dict[str, Path]:
    modules = {
        module_name_for_path(path): path
        for path in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    }
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
        or base == "sqlalchemy"
        or base.startswith("sqlalchemy.")
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
                if (
                    imported is not None
                    or alias.name.startswith("tests")
                    or alias.name == "sqlalchemy"
                    or alias.name.startswith("sqlalchemy.")
                ):
                    edges.append(
                        ImportEdge(
                            importer=module_name,
                            imported=imported or alias.name,
                            line=node.lineno,
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            for candidate in import_from_candidates(node, module_name, path):
                imported = best_known_module(candidate, known_modules)
                if (
                    imported is not None
                    or candidate.startswith("tests")
                    or candidate == "sqlalchemy"
                    or candidate.startswith("sqlalchemy.")
                ):
                    edges.append(
                        ImportEdge(
                            importer=module_name,
                            imported=imported or candidate,
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


def is_video_action_set_runtime_module(module: str) -> bool:
    prefix = f"{VIDEO_ACTION_SET_PREFIX}."
    if not module.startswith(prefix):
        return False
    if module in VIDEO_ACTION_SET_POLICY_MODULES:
        return False
    suffix = module[len(prefix) :]
    return not (
        suffix.startswith("contracts.")
        or suffix.startswith("mutation_audit.")
        or suffix.startswith("verification.")
    )


def is_module_under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def is_sqlalchemy_import(module: str) -> bool:
    return module == "sqlalchemy" or module.startswith("sqlalchemy.")


def edge_violation(rule: str, edge: ImportEdge) -> Violation:
    return Violation(
        rule=rule,
        importer=edge.importer,
        imported=edge.imported,
        detail=f"forbidden edge at line {edge.line}",
    )


def legacy_forbidden_edge_violations(edge: ImportEdge) -> list[Violation]:
    violations: list[Violation] = []
    if edge.imported == "tests" or edge.imported.startswith("tests."):
        violations.append(edge_violation("zeromodel/ must not import tests.*", edge))
    if edge.importer == "zeromodel.artifact" and (
        edge.imported.startswith("zeromodel.video_")
        or edge.imported == VIDEO_ACTION_SET_PREFIX
        or edge.imported.startswith(f"{VIDEO_ACTION_SET_PREFIX}.")
    ):
        violations.append(
            edge_violation(
                "zeromodel.artifact must not import video research modules",
                edge,
            )
        )
    if edge.importer == f"{VIDEO_ACTION_SET_PREFIX}.contracts" and (
        edge.imported == VIDEO_ACTION_SET_PREFIX
        or edge.imported.startswith(f"{VIDEO_ACTION_SET_PREFIX}.")
    ):
        violations.append(
            edge_violation(
                "video_action_set/contracts.py must not import implementation modules",
                edge,
            )
        )
    if is_video_action_set_runtime_module(edge.importer) and (
        edge.imported == f"{VIDEO_ACTION_SET_PREFIX}.verification"
        or edge.imported.startswith(f"{VIDEO_ACTION_SET_PREFIX}.verification.")
        or edge.imported == f"{VIDEO_ACTION_SET_PREFIX}.mutation_audit"
        or edge.imported.startswith(f"{VIDEO_ACTION_SET_PREFIX}.mutation_audit.")
    ):
        violations.append(
            edge_violation(
                "runtime modules must not depend on verification or mutation_audit",
                edge,
            )
        )
    return violations


def rmdto_forbidden_edge_violations(edge: ImportEdge) -> list[Violation]:
    violations: list[Violation] = []
    if is_module_under(edge.importer, DOMAIN_VIDEO_ACTION_SET_PREFIX) and (
        is_module_under(edge.imported, DB_ORM_PREFIX)
        or is_module_under(edge.imported, DB_STORES_PREFIX)
        or is_module_under(edge.imported, "zeromodel.db.session")
        or is_sqlalchemy_import(edge.imported)
    ):
        violations.append(
            edge_violation(
                "video_action_set domain modules must not import persistence layers",
                edge,
            )
        )
    if is_module_under(edge.importer, DOMAIN_VIDEO_ACTION_SET_PREFIX) and (
        edge.imported == VIDEO_ACTION_SET_BENCHMARK_MODULE
    ):
        violations.append(
            edge_violation(
                "video_action_set domain modules must not import legacy benchmark",
                edge,
            )
        )
    if edge.importer in VIDEO_ACTION_SET_PURE_DOMAIN_MODULES and (
        is_module_under(edge.imported, "zeromodel.stores")
        or edge.imported == "zeromodel.runtime"
    ):
        violations.append(
            edge_violation(
                "video_action_set DTO, contracts, and Services must not import runtime layers",
                edge,
            )
        )
    if edge.importer == "zeromodel.runtime" and (
        is_module_under(edge.imported, "zeromodel.db")
        or is_sqlalchemy_import(edge.imported)
    ):
        violations.append(
            edge_violation(
                "zeromodel.runtime must not import database or SQLAlchemy modules",
                edge,
            )
        )
    if is_module_under(edge.importer, DB_ORM_PREFIX) and (
        edge.imported in VIDEO_ACTION_SET_ORCHESTRATION_MODULES
        or edge.imported == VIDEO_ACTION_SET_BENCHMARK_MODULE
    ):
        violations.append(
            edge_violation(
                "ORM modules must not import Services, Engines, Facades, Runtime, or legacy benchmark",
                edge,
            )
        )
    if is_module_under(edge.importer, DB_STORES_PREFIX) and (
        edge.imported in VIDEO_ACTION_SET_ORCHESTRATION_MODULES
        or edge.imported == VIDEO_ACTION_SET_BENCHMARK_MODULE
    ):
        violations.append(
            edge_violation(
                "database Store implementations must not import orchestration or legacy benchmark layers",
                edge,
            )
        )
    return violations


def forbidden_edge_violations(edges: list[ImportEdge]) -> list[Violation]:
    violations: list[Violation] = []
    for edge in edges:
        violations.extend(legacy_forbidden_edge_violations(edge))
        violations.extend(rmdto_forbidden_edge_violations(edge))
    return violations


def print_violations(violations: list[Violation]) -> None:
    for violation in sorted(
        violations, key=lambda item: (item.rule, item.importer, item.imported)
    ):
        print(f"Architecture violation: {violation.rule}", file=sys.stderr)
        print(f"  importer: {violation.importer}", file=sys.stderr)
        print(f"  imported: {violation.imported}", file=sys.stderr)
        print(f"  detail: {violation.detail}", file=sys.stderr)


def main() -> int:
    modules = discover_modules()
    graph, edges = build_import_graph(modules)
    violations = [*cycle_violations(graph), *forbidden_edge_violations(edges)]
    if violations:
        print_violations(violations)
        return 1
    print("Architecture check: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
