from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_BOUNDARIES_PATH = REPO_ROOT / "package-boundaries.toml"

# Generator version for this script's inventory schema/logic. Bump when the
# discovery, classification, or output shape changes in a way that would
# make an old inventory misleading if compared byte-for-byte against a new
# one.
GENERATOR_VERSION = "2.0.0"

# Non-production roots this script still inspects as repository tooling,
# not as production source. Production source is discovered exclusively
# from package-boundaries.toml's configured `source_root` entries (see
# discover_package_files). The historical monolithic root ("zeromodel/",
# package-boundaries.toml's forbidden_roots) is intentionally absent from
# this mapping: it is not scanned, so it can never be reported as current
# production implementation.
TOOLING_ROOTS: dict[str, str] = {
    "tests": "tooling",
    "examples": "examples",
    "scripts": "tooling",
    "research": "research",
    "integration_tests": "tooling",
}

CLASSIFICATIONS = {
    "core",
    "analysis",
    "observation",
    "vision",
    "video",
    "sqlalchemy",
    "artifacts",
    "trust",
    "navigation",
    "research",
    "examples",
    "tooling",
    "undecided",
}
STDLIB_HINTS = set(getattr(sys, "stdlib_module_names", ()))
TRACKED_EXTERNALS = {
    "numpy",
    "sqlalchemy",
    "torch",
    "torchvision",
    "transformers",
    "PIL",
    "pytest",
}


def load_package_boundaries() -> dict[str, dict[str, Any]]:
    data = tomllib.loads(PACKAGE_BOUNDARIES_PATH.read_text(encoding="utf-8"))
    return dict(data["packages"])


def load_release_version() -> str:
    data = tomllib.loads(PACKAGE_BOUNDARIES_PATH.read_text(encoding="utf-8"))
    return str(data["release_version"])


@dataclass(frozen=True)
class ModuleInfo:
    path: str
    module: str
    lines: int
    tree: ast.AST


@dataclass(frozen=True)
class ImportEdge:
    importer: str
    imported: str
    line: int
    kind: str
    resolved: bool


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def module_for_path(path: Path, base: Path) -> str:
    parts = list(path.relative_to(base).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def discover_tooling_files(
    boundaries: dict[str, dict[str, Any]] | None = None,
) -> list[Path]:
    """Discover repository-wide and package-local tooling modules.

    Package-local test roots are derived from each configured production
    ``source_root`` so the current nine-package inventory cannot silently omit
    the tests that provide most package evidence.
    """
    boundaries = boundaries if boundaries is not None else load_package_boundaries()
    paths: list[Path] = []
    for root in TOOLING_ROOTS:
        base = REPO_ROOT / root
        if base.exists():
            paths.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    for config in boundaries.values():
        package_tests = (REPO_ROOT / config["source_root"]).parent / "tests"
        if package_tests.exists():
            paths.extend(
                p for p in package_tests.rglob("*.py") if "__pycache__" not in p.parts
            )
    return sorted(set(paths), key=rel)


def is_test_path(path: str) -> bool:
    return (
        path.startswith(("tests/", "integration_tests/"))
        or path.startswith("packages/")
        and "/tests/" in path
    )


def discover_package_files(
    boundaries: dict[str, dict[str, Any]],
) -> dict[str, list[Path]]:
    """Production source, discovered exclusively from each configured
    package's `source_root` in package-boundaries.toml. Raises loudly if a
    configured package's source root is missing or contains zero Python
    modules - that is never a legitimate current-architecture state, it
    means either the checkout or package-boundaries.toml has drifted."""
    files_by_package: dict[str, list[Path]] = {}
    for key, config in boundaries.items():
        source_root = REPO_ROOT / config["source_root"]
        if not source_root.exists():
            raise SystemExit(
                f"analyze_package_inventory: configured source_root for "
                f"package '{key}' does not exist: {config['source_root']}"
            )
        found = sorted(
            (p for p in source_root.rglob("*.py") if "__pycache__" not in p.parts),
            key=rel,
        )
        if not found:
            raise SystemExit(
                f"analyze_package_inventory: package '{key}' source_root "
                f"{config['source_root']} contains zero Python modules"
            )
        files_by_package[key] = found
    return files_by_package


def load_modules(paths: Iterable[Path], base: Path) -> dict[str, ModuleInfo]:
    modules = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        module = module_for_path(path, base)
        modules[module] = ModuleInfo(
            rel(path),
            module,
            text.count("\n") + (0 if text.endswith("\n") else 1),
            ast.parse(text, filename=rel(path)),
        )
    return modules


def package_for(module: str, path: str) -> str:
    if path.endswith("__init__.py"):
        return module
    return module.rsplit(".", 1)[0] if "." in module else ""


def resolve_relative(module: str, path: str, level: int, imported: str | None) -> str:
    parts = package_for(module, path).split(".") if package_for(module, path) else []
    base = parts[: max(0, len(parts) - level + 1)]
    if imported:
        base.extend(imported.split("."))
    return ".".join(p for p in base if p)


def best_known(candidate: str, known: set[str]) -> str | None:
    parts = candidate.split(".")
    while parts:
        current = ".".join(parts)
        if current in known:
            return current
        parts.pop()
    return None


def edge_kind(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    kind = "runtime"
    cur: ast.AST | None = node
    while cur in parents:
        cur = parents[cur]
        if (
            isinstance(cur, ast.If)
            and isinstance(cur.test, ast.Name)
            and cur.test.id == "TYPE_CHECKING"
        ):
            return "type-checking"
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            kind = "deferred"
        if isinstance(cur, ast.Try):
            kind = "optional"
    return kind


def external_name(candidate: str, known: set[str]) -> str | None:
    top = candidate.split(".", 1)[0]
    if top in {
        "zeromodel",
        "tests",
        "examples",
        "scripts",
        "research",
        "integration_tests",
        "packages",
    }:
        return None
    if top in STDLIB_HINTS:
        return None
    if best_known(candidate, known):
        return None
    return top


def collect_imports(
    info: ModuleInfo, known: set[str]
) -> tuple[list[ImportEdge], set[str], list[dict[str, object]]]:
    parents = {
        child: parent
        for parent in ast.walk(info.tree)
        for child in ast.iter_child_nodes(parent)
    }
    edges: list[ImportEdge] = []
    externals: set[str] = set()
    dynamic: list[dict[str, object]] = []
    for node in ast.walk(info.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = best_known(alias.name, known)
                if resolved:
                    edges.append(
                        ImportEdge(
                            info.module,
                            resolved,
                            node.lineno,
                            edge_kind(node, parents),
                            True,
                        )
                    )
                elif ext := external_name(alias.name, known):
                    externals.add(ext)
        elif isinstance(node, ast.ImportFrom):
            base = (
                resolve_relative(info.module, info.path, node.level, node.module)
                if node.level
                else (node.module or "")
            )
            candidates = (
                [base]
                if any(a.name == "*" for a in node.names)
                else [f"{base}.{a.name}" for a in node.names]
            )
            for cand in candidates:
                resolved = best_known(cand, known)
                if resolved:
                    edges.append(
                        ImportEdge(
                            info.module,
                            resolved,
                            node.lineno,
                            edge_kind(node, parents),
                            True,
                        )
                    )
                elif ext := external_name(base, known):
                    externals.add(ext)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
                and node.func.attr == "import_module"
            ):
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    cand = node.args[0].value
                    resolved = best_known(cand, known)
                    if resolved:
                        edges.append(
                            ImportEdge(
                                info.module, resolved, node.lineno, "dynamic", True
                            )
                        )
                    else:
                        dynamic.append(
                            {
                                "importer": info.module,
                                "line": node.lineno,
                                "target": cand,
                            }
                        )
                else:
                    dynamic.append(
                        {
                            "importer": info.module,
                            "line": node.lineno,
                            "target": "<non-literal importlib.import_module>",
                        }
                    )
    return (
        sorted(set(edges), key=lambda e: (e.importer, e.imported, e.line, e.kind)),
        externals,
        dynamic,
    )


def public_symbols(info: ModuleInfo) -> list[str]:
    exports: list[str] = []
    all_value: list[str] | None = None
    for node in ast.iter_child_nodes(info.tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and not node.name.startswith("_"):
            exports.append(node.name)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "__all__" in names and isinstance(node.value, (ast.List, ast.Tuple)):
                all_value = [
                    e.value
                    for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
            exports.extend(n for n in names if n.isupper() or n.endswith("_VERSION"))
    return sorted(all_value if all_value is not None else set(exports))


def classify(
    path: str,
    module: str,
    package_key: str | None,
    boundaries: dict[str, dict[str, Any]],
) -> tuple[str, str, str, str, str, str, str]:
    """Classify a discovered module.

    Production modules are classified directly from the package whose
    source_root they were discovered under - no name-based heuristics are
    needed since discover_package_files already knows the owning package.
    Tooling-root modules (tests/examples/scripts/research/integration_tests)
    keep the previous path-prefix classification.
    """
    if is_test_path(path):
        return (
            "tooling",
            "",
            "",
            "test coverage and fixtures",
            "retain-tooling",
            "high",
            "Test module remains repository tooling for package boundary validation.",
        )
    if path.startswith("examples/"):
        return (
            "examples",
            "",
            "",
            "example or benchmark entry point",
            "retain-tooling",
            "high",
            "Example code demonstrates or benchmarks capabilities outside production package ownership.",
        )
    if path.startswith("scripts/"):
        return (
            "tooling",
            "",
            "",
            "repository tooling and release validation",
            "retain-tooling",
            "high",
            "Repository script is not a production import namespace.",
        )
    if path.startswith("integration_tests/"):
        return (
            "tooling",
            "",
            "",
            "cross-package integration test",
            "retain-tooling",
            "high",
            "Integration test exercises multiple production distributions together and is not itself a production namespace.",
        )
    if path.startswith("research/"):
        return (
            "research",
            "",
            "",
            "benchmark, evidence, or unpromoted experimental machinery",
            "retain-research",
            "high",
            "Research code stays outside the production package contract per architecture rules; not eligible for production promotion without separate review.",
        )
    if package_key is not None:
        config = boundaries[package_key]
        return (
            package_key,
            config["distribution"],
            module,
            "module implementation",
            "current",
            "high",
            f"Discovered under {config['source_root']} as configured in package-boundaries.toml; already at its current production location.",
        )
    return (
        "undecided",
        "",
        "",
        "unclassified path outside configured roots",
        "undecided",
        "low",
        "Path does not match any configured production source root or known tooling root.",
    )


def identity_schema(symbols: list[str], module: str) -> str:
    versions = [s for s in symbols if s.endswith("_VERSION") or s in {"__version__"}]
    if versions:
        return ";".join(versions)
    if any(
        t in module
        for t in [
            "artifact",
            "identity",
            "digest",
            "dto",
            "orm",
            "schema",
            "store",
            "matrix_blob",
        ]
    ):
        return "identity/schema behavior"
    return "none"


def strongly_connected(graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    idx: dict[str, int] = {}
    low: dict[str, int] = {}
    on: set[str] = set()
    out: list[list[str]] = []

    def visit(v: str) -> None:
        nonlocal index
        idx[v] = low[v] = index
        index += 1
        stack.append(v)
        on.add(v)
        for w in sorted(graph[v]):
            if w not in idx:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            comp = []
            while True:
                w = stack.pop()
                on.remove(w)
                comp.append(w)
                if w == v:
                    break
            out.append(sorted(comp))

    for v in sorted(graph):
        if v not in idx:
            visit(v)
    return [c for c in out if len(c) > 1 or any(v in graph[v] for v in c)]


def allowed_package_edges(boundaries: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    return {key: set(config["depends_on"]) for key, config in boundaries.items()}


def resolve_source_state() -> tuple[str, bool]:
    """Return the source baseline and whether the analyzed tree is dirty.

    Environment overrides support generation from an exported checkout that
    intentionally omits ``.git`` while keeping provenance explicit.
    """
    baseline_override = os.environ.get("ZEROMODEL_INVENTORY_BASELINE")
    dirty_override = os.environ.get("ZEROMODEL_INVENTORY_DIRTY")
    if baseline_override is not None:
        dirty = str(dirty_override or "false").lower() in {"1", "true", "yes"}
        return baseline_override, dirty
    try:
        baseline = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True
    return baseline, dirty


def make_inventory(
    generated_at: str | None = None,
    boundaries: dict[str, dict[str, Any]] | None = None,
) -> dict[str, object]:
    boundaries = boundaries if boundaries is not None else load_package_boundaries()
    baseline, source_tree_dirty = resolve_source_state()
    generated_at = generated_at or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")

    package_files = discover_package_files(boundaries)
    tooling_modules = load_modules(discover_tooling_files(boundaries), REPO_ROOT)
    modules: dict[str, ModuleInfo] = dict(tooling_modules)
    module_package_key: dict[str, str] = {}
    for key, paths in package_files.items():
        source_root = REPO_ROOT / boundaries[key]["source_root"]
        package_modules = load_modules(paths, source_root)
        for module_name in package_modules:
            module_package_key[module_name] = key
        modules.update(package_modules)
    modules = dict(sorted(modules.items()))

    known = set(modules)
    all_edges: list[ImportEdge] = []
    externals_by_module: dict[str, set[str]] = {}
    unresolved_dynamic: list[dict[str, object]] = []
    for info in modules.values():
        edges, externals, dynamic = collect_imports(info, known)
        all_edges.extend(edges)
        externals_by_module[info.module] = externals
        unresolved_dynamic.extend(dynamic)
    public_by_module = {m: public_symbols(i) for m, i in modules.items()}
    inbound = Counter(e.imported for e in all_edges if e.imported in known)
    outbound = Counter(e.importer for e in all_edges if e.imported in known)
    tests_by_mod: dict[str, set[str]] = defaultdict(set)
    examples_by_mod: dict[str, set[str]] = defaultdict(set)
    for e in all_edges:
        if e.imported in known and is_test_path(modules[e.importer].path):
            tests_by_mod[e.imported].add(modules[e.importer].path)
        if e.imported in known and e.importer.startswith("examples."):
            examples_by_mod[e.imported].add(modules[e.importer].path)
    rows = []
    class_by_module = {}
    for module, info in modules.items():
        package_key = module_package_key.get(module)
        cls, dist, ns, resp, action, conf, rationale = classify(
            info.path, module, package_key, boundaries
        )
        class_by_module[module] = cls
        block = (
            "Requires responsibility split table before extraction."
            if cls == "undecided"
            else ""
        )
        rows.append(
            {
                "path": info.path,
                "module": module,
                "lines": info.lines,
                "classification": cls,
                "target_distribution": dist,
                "target_namespace": ns,
                "responsibility": resp,
                "public_symbols": ";".join(public_by_module[module]),
                "inbound_internal_count": inbound[module],
                "outbound_internal_count": outbound[module],
                "external_dependencies": ";".join(sorted(externals_by_module[module])),
                "test_paths": ";".join(sorted(tests_by_mod[module])),
                "example_paths": ";".join(sorted(examples_by_mod[module])),
                "cli_entry_points": "console/module script"
                if info.path.startswith(("scripts/", "examples/"))
                or module.endswith("_cli")
                else "",
                "identity_or_schema_ownership": identity_schema(
                    public_by_module[module], module
                ),
                "move_action": action,
                "confidence": conf,
                "rationale": rationale,
                "blocking_questions": block,
            }
        )
    graph = {
        m: {e.imported for e in all_edges if e.importer == m and e.imported in known}
        for m in modules
    }
    edges = allowed_package_edges(boundaries)
    package_edges: dict[str, set[str]] = defaultdict(set)
    forbidden = []
    for e in all_edges:
        if e.imported in known:
            a = class_by_module[e.importer]
            b = class_by_module[e.imported]
            if a in boundaries and b in boundaries and a != b:
                package_edges[a].add(b)
                if b not in edges.get(a, set()):
                    forbidden.append(e)
    graph_json = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "inventory_kind": "current_architecture",
        "baseline_commit": baseline,
        "generated_at_utc": generated_at,
        "source_tree_dirty": source_tree_dirty,
        "modules": {
            m: {
                "path": i.path,
                "classification": class_by_module[m],
                "target_namespace": next(
                    r["target_namespace"] for r in rows if r["module"] == m
                ),
                "lines": i.lines,
            }
            for m, i in modules.items()
        },
        "edges": [
            e.__dict__
            for e in sorted(
                all_edges, key=lambda e: (e.importer, e.imported, e.line, e.kind)
            )
        ],
        "strongly_connected_components": [
            {"modules": c, "cycle_path": c + [c[0]]} for c in strongly_connected(graph)
        ],
        "unresolved_dynamic_imports": sorted(
            unresolved_dynamic, key=lambda d: (str(d["importer"]), int(d["line"]))
        ),
        "external_dependencies": {
            k: sorted(v) for k, v in sorted(externals_by_module.items()) if v
        },
    }
    return {
        "baseline": baseline,
        "generated_at": generated_at,
        "source_tree_dirty": source_tree_dirty,
        "rows": rows,
        "graph": graph_json,
        "class_counts": Counter(r["classification"] for r in rows),
        "package_edges": {k: sorted(v) for k, v in sorted(package_edges.items())},
        "forbidden": forbidden,
        "boundaries": boundaries,
    }


def write_outputs(
    data: dict[str, object], output_dir: Path | None = None
) -> None:
    arch = output_dir or REPO_ROOT / "docs" / "architecture"
    arch.mkdir(parents=True, exist_ok=True)
    csv_path = arch / "package-module-map-1.0.13.csv"
    json_path = arch / "package-import-graph-1.0.13.json"
    inv_path = arch / "package-inventory-1.0.13.md"
    findings_path = arch / "package-dependency-findings-1.0.13.md"
    rows = data["rows"]
    boundaries: dict[str, dict[str, Any]] = data["boundaries"]  # type: ignore[assignment]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(data["graph"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    counts = "\n".join(f"- {k}: {v}" for k, v in sorted(data["class_counts"].items()))
    forbidden = data["forbidden"]
    package_list = "\n".join(
        f"- `{config['distribution']}` (`{config['namespace']}`) - {config['source_root']}"
        for config in boundaries.values()
    )
    allowed_edges_text = "; ".join(
        f"{key}->{','.join(sorted(deps)) or '(none)'}"
        for key, deps in sorted(allowed_package_edges(boundaries).items())
    )
    inv_path.write_text(
        f"""# ZeroModel Current Architecture Package Inventory

**Status: current architecture inventory** (not a historical migration snapshot; the nine-package split described here is the present state of `main`, not a plan).

Generator version: `{GENERATOR_VERSION}`
Baseline commit: `{data["baseline"]}`
Generated (UTC): `{data["generated_at"]}`
Source tree dirty: `{str(data["source_tree_dirty"]).lower()}`

Generated artifacts:

- `docs/architecture/package-module-map-1.0.13.csv`
- `docs/architecture/package-import-graph-1.0.13.json`
- `docs/architecture/package-dependency-findings-1.0.13.md`

## Module Count By Classification

{counts}

## Production Packages

Production source is discovered exclusively from the `source_root` entries in `package-boundaries.toml` (the authoritative package configuration), one row per configured package:

{package_list}

The historical monolithic root (`package-boundaries.toml`'s `forbidden_roots = ["zeromodel"]`) is not scanned by this script and is never reported as current production implementation.

## Package Build And Data Inventory

Each production package under `packages/*/` ships its own `pyproject.toml`, distribution name, and version. `package-boundaries.toml` declares `release_version = "{load_release_version()}"` as the coordinated release-candidate version across all nine packages; see the individual package manifests under `packages/*/pyproject.toml` for exact per-package dependency declarations. The repository root `pyproject.toml` no longer declares a `[project]` section or builds a distribution of its own; it only holds shared tool configuration (pytest, ruff, mypy) that spans all nine packages via `pythonpath`/`mypy_path` entries under `packages/*/src`.

## Domain Boundary Inventory

The RMDTO target path is Runtime -> Facade -> Engine -> Service -> Store protocol -> Store implementation -> ORM. SQLAlchemy ownership is isolated under `packages/sqlalchemy/src/zeromodel/persistence/sqlalchemy`; video runtime and stores live under `packages/video/src/zeromodel/video` and are expected to stay SQLAlchemy-free at the domain-service layer. Suspicious and forbidden observed edges are ranked in the dependency findings document.

## Architecture Comparison

Allowed target graph (derived from `package-boundaries.toml` `depends_on`): {allowed_edges_text}.

Observed classification graph: `{json.dumps(data["package_edges"], sort_keys=True)}`.

Forbidden observed edge count: `{len(forbidden)}`.
""",
        encoding="utf-8",
    )
    finding_lines = [
        "# ZeroModel Current Architecture Package Dependency Findings",
        "",
        "**Status: current architecture findings** (not a historical migration snapshot).",
        "",
        f"Generator version: `{GENERATOR_VERSION}`",
        f"Baseline commit: `{data['baseline']}`",
        f"Generated (UTC): `{data['generated_at']}`",
        f"Source tree dirty: `{str(data['source_tree_dirty']).lower()}`",
        "",
    ]
    finding_lines += ["## Blocker", ""]
    if forbidden:
        for i, e in enumerate(forbidden[:20], 1):
            finding_lines += [
                f"### B{i}. Forbidden observed package edge `{e.importer}` -> `{e.imported}`",
                "",
                f"- Import edge: `{e.importer}` imports `{e.imported}` at line {e.line} ({e.kind}).",
                "- Conflict: observed classification graph contains an edge not permitted by package-boundaries.toml's `depends_on` graph.",
                "- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by an allowed dependency.",
                "",
            ]
    else:
        finding_lines += [
            "No blocker forbidden edges detected by the generated classifier.",
            "",
        ]
    finding_lines += [
        "## High",
        "",
        "- (none currently detected by the generated classifier; see Blocker section for any forbidden edges.)",
        "",
        "## Medium",
        "",
        "- (none currently detected by the generated classifier.)",
        "",
        "## Low",
        "",
        "- (none currently detected by the generated classifier.)",
        "",
    ]
    findings_path.write_text("\n".join(finding_lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--generated-at")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data = make_inventory(args.generated_at)
    if args.write:
        write_outputs(data, args.output_dir)
    else:
        print(
            json.dumps(
                {
                    "baseline": data["baseline"],
                    "module_count": len(data["rows"]),
                    "class_counts": data["class_counts"],
                },
                default=dict,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
