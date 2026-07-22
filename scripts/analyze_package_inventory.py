from __future__ import annotations

import argparse
import ast
import csv
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOTS = ("zeromodel", "scripts", "examples", "tests")
CLASSIFICATIONS = {
    "core",
    "analysis",
    "observation",
    "vision",
    "video",
    "sqlalchemy",
    "research",
    "examples",
    "tooling",
    "delete",
    "split",
    "undecided",
}
DISTRIBUTIONS = {
    "core": ("zeromodel", "zeromodel.core"),
    "analysis": ("zeromodel-analysis", "zeromodel.analysis"),
    "observation": ("zeromodel-observation", "zeromodel.observation"),
    "vision": ("zeromodel-vision", "zeromodel.vision"),
    "video": ("zeromodel-video", "zeromodel.video"),
    "sqlalchemy": ("zeromodel-sqlalchemy", "zeromodel.persistence.sqlalchemy"),
}
ALLOWED_PACKAGE_EDGES = {
    "analysis": {"core"},
    "observation": {"core"},
    "vision": {"core", "observation"},
    "video": {"core", "observation"},
    "sqlalchemy": {"core", "video"},
}
STDLIB_HINTS = set(getattr(__import__("sys"), "stdlib_module_names", ()))
TRACKED_EXTERNALS = {"numpy", "sqlalchemy", "torch", "torchvision", "transformers", "PIL", "pytest"}


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


def module_for_path(path: Path) -> str:
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def discover_python_files() -> list[Path]:
    paths: list[Path] = []
    for root in PY_ROOTS:
        base = REPO_ROOT / root
        if base.exists():
            paths.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(paths, key=rel)


def load_modules(paths: Iterable[Path]) -> dict[str, ModuleInfo]:
    modules = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        modules[module_for_path(path)] = ModuleInfo(rel(path), module_for_path(path), text.count("\n") + (0 if text.endswith("\n") else 1), ast.parse(text, filename=rel(path)))
    return dict(sorted(modules.items()))


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
        if isinstance(cur, ast.If) and isinstance(cur.test, ast.Name) and cur.test.id == "TYPE_CHECKING":
            return "type-checking"
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            kind = "deferred"
        if isinstance(cur, ast.Try):
            kind = "optional"
    return kind


def external_name(candidate: str, known: set[str]) -> str | None:
    top = candidate.split(".", 1)[0]
    if top in {"zeromodel", "tests", "examples", "scripts"}:
        return None
    if top in STDLIB_HINTS:
        return None
    if best_known(candidate, known):
        return None
    return top


def collect_imports(info: ModuleInfo, known: set[str]) -> tuple[list[ImportEdge], set[str], list[dict[str, object]]]:
    parents = {child: parent for parent in ast.walk(info.tree) for child in ast.iter_child_nodes(parent)}
    edges: list[ImportEdge] = []
    externals: set[str] = set()
    dynamic: list[dict[str, object]] = []
    for node in ast.walk(info.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = best_known(alias.name, known)
                if resolved:
                    edges.append(ImportEdge(info.module, resolved, node.lineno, edge_kind(node, parents), True))
                elif ext := external_name(alias.name, known):
                    externals.add(ext)
        elif isinstance(node, ast.ImportFrom):
            base = resolve_relative(info.module, info.path, node.level, node.module) if node.level else (node.module or "")
            candidates = [base] if any(a.name == "*" for a in node.names) else [f"{base}.{a.name}" for a in node.names]
            for cand in candidates:
                resolved = best_known(cand, known)
                if resolved:
                    edges.append(ImportEdge(info.module, resolved, node.lineno, edge_kind(node, parents), True))
                elif ext := external_name(base, known):
                    externals.add(ext)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "importlib" and node.func.attr == "import_module":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    cand = node.args[0].value
                    resolved = best_known(cand, known)
                    if resolved:
                        edges.append(ImportEdge(info.module, resolved, node.lineno, "dynamic", True))
                    else:
                        dynamic.append({"importer": info.module, "line": node.lineno, "target": cand})
                else:
                    dynamic.append({"importer": info.module, "line": node.lineno, "target": "<non-literal importlib.import_module>"})
    return sorted(set(edges), key=lambda e: (e.importer, e.imported, e.line, e.kind)), externals, dynamic


def public_symbols(info: ModuleInfo) -> list[str]:
    exports: list[str] = []
    all_value: list[str] | None = None
    for node in ast.iter_child_nodes(info.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            exports.append(node.name)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "__all__" in names and isinstance(node.value, (ast.List, ast.Tuple)):
                all_value = [e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            exports.extend(n for n in names if n.isupper() or n.endswith("_VERSION"))
    return sorted(all_value if all_value is not None else set(exports))


def classify(path: str, module: str, symbols: list[str], externals: set[str]) -> tuple[str, str, str, str, str, str, str]:
    responsibility = "module implementation"
    if path.startswith("tests/"):
        return "tooling", "", "", "test coverage and fixtures", "retain-tooling", "high", "Test module remains repository tooling for package extraction validation."
    if path.startswith("examples/"):
        return "examples", "", "", "example or benchmark entry point", "retain-tooling", "high", "Example code demonstrates or benchmarks capabilities outside production package ownership."
    if path.startswith("scripts/"):
        return "tooling", "", "", "repository tooling and release validation", "retain-tooling", "high", "Repository script is not a production import namespace."
    if module == "zeromodel":
        return "split", "", "", "root compatibility exports across all current capabilities", "split", "high", "Root initializer re-exports symbols from every capability and is explicitly removed by the architecture contract."
    if module.startswith("zeromodel.db"):
        cls = "sqlalchemy"
    elif any(token in module for token in ["benchmark", "evidence", "experiment", "calibration", "system_b", "local_correlation", "discriminative", "prospective", "complete_row", "equivalence"]):
        cls = "research"
    elif module.startswith("zeromodel.visual_address") or module in {"zeromodel.deployment_binding", "zeromodel.visual_policy"}:
        cls = "observation"
    elif module.startswith("zeromodel.visual") or module == "zeromodel.vision":
        cls = "vision"
    elif module.startswith("zeromodel.video") or module.startswith("zeromodel.domains.video_action_set") or module.startswith("zeromodel.stores") or module in {"zeromodel.runtime"}:
        cls = "video"
    elif module in {"zeromodel.artifact", "zeromodel.bundle", "zeromodel.content_identity", "zeromodel.matrix_blob", "zeromodel.metrics", "zeromodel.policy_lookup", "zeromodel.render", "zeromodel.views", "zeromodel.lua"}:
        cls = "core"
    elif module.startswith("zeromodel.adapters"):
        cls = "analysis"
    else:
        cls = "analysis"
    dist, ns = DISTRIBUTIONS.get(cls, ("", ""))
    if dist:
        ns = ns + "." + module.removeprefix("zeromodel.").replace(".", "_")
    if cls == "research":
        return cls, "", "", "benchmark, evidence, or unpromoted experimental machinery", "move-research", "medium", "Name/import context indicates benchmark or evidence machinery; production promotion needs architectural review."
    return cls, dist, ns, responsibility, "move", "medium", f"Classified by observed responsibility and package-system candidate ownership as {cls}."


def identity_schema(symbols: list[str], module: str) -> str:
    versions = [s for s in symbols if s.endswith("_VERSION") or s in {"__version__"}]
    if versions:
        return ";".join(versions)
    if any(t in module for t in ["artifact", "identity", "digest", "dto", "orm", "schema", "store", "matrix_blob"]):
        return "identity/schema behavior"
    return "none"


def strongly_connected(graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0; stack=[]; idx={}; low={}; on=set(); out=[]
    def visit(v: str) -> None:
        nonlocal index
        idx[v]=low[v]=index; index+=1; stack.append(v); on.add(v)
        for w in sorted(graph[v]):
            if w not in idx:
                visit(w); low[v]=min(low[v], low[w])
            elif w in on:
                low[v]=min(low[v], idx[w])
        if low[v] == idx[v]:
            comp=[]
            while True:
                w=stack.pop(); on.remove(w); comp.append(w)
                if w == v: break
            out.append(sorted(comp))
    for v in sorted(graph):
        if v not in idx:
            visit(v)
    return [c for c in out if len(c) > 1 or any(v in graph[v] for v in c)]


def make_inventory(generated_at: str | None = None) -> dict[str, object]:
    baseline = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    generated_at = generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    modules = load_modules(discover_python_files())
    known = set(modules)
    all_edges: list[ImportEdge] = []
    externals_by_module: dict[str, set[str]] = {}
    unresolved_dynamic: list[dict[str, object]] = []
    for info in modules.values():
        edges, externals, dynamic = collect_imports(info, known)
        all_edges.extend(edges); externals_by_module[info.module] = externals; unresolved_dynamic.extend(dynamic)
    public_by_module = {m: public_symbols(i) for m, i in modules.items()}
    inbound = Counter(e.imported for e in all_edges if e.imported in known)
    outbound = Counter(e.importer for e in all_edges if e.imported in known)
    tests_by_mod: dict[str, set[str]] = defaultdict(set); examples_by_mod: dict[str, set[str]] = defaultdict(set)
    for e in all_edges:
        if e.imported in known and e.importer.startswith("tests."): tests_by_mod[e.imported].add(modules[e.importer].path)
        if e.imported in known and e.importer.startswith("examples."): examples_by_mod[e.imported].add(modules[e.importer].path)
    rows=[]
    class_by_module={}
    for module, info in modules.items():
        cls, dist, ns, resp, action, conf, rationale = classify(info.path, module, public_by_module[module], externals_by_module[module])
        class_by_module[module]=cls
        block = "Requires responsibility split table before extraction." if cls == "split" else ""
        rows.append({
            "path": info.path, "module": module, "lines": info.lines, "classification": cls,
            "target_distribution": dist, "target_namespace": ns, "responsibility": resp,
            "public_symbols": ";".join(public_by_module[module]), "inbound_internal_count": inbound[module],
            "outbound_internal_count": outbound[module], "external_dependencies": ";".join(sorted(externals_by_module[module])),
            "test_paths": ";".join(sorted(tests_by_mod[module])), "example_paths": ";".join(sorted(examples_by_mod[module])),
            "cli_entry_points": "console/module script" if info.path.startswith(("scripts/", "examples/")) or module.endswith("_cli") else "",
            "identity_or_schema_ownership": identity_schema(public_by_module[module], module), "move_action": action,
            "confidence": conf, "rationale": rationale, "blocking_questions": block,
        })
    graph = {m: {e.imported for e in all_edges if e.importer == m and e.imported in known} for m in modules}
    package_edges=defaultdict(set)
    forbidden=[]
    for e in all_edges:
        if e.imported in known:
            a=class_by_module[e.importer]; b=class_by_module[e.imported]
            if a in DISTRIBUTIONS and b in DISTRIBUTIONS and a != b:
                package_edges[a].add(b)
                if b not in ALLOWED_PACKAGE_EDGES.get(a, set()):
                    forbidden.append(e)
    graph_json = {
        "schema_version": 1, "baseline_commit": baseline, "generated_at_utc": generated_at,
        "modules": {m: {"path": i.path, "classification": class_by_module[m], "target_namespace": next(r["target_namespace"] for r in rows if r["module"] == m), "lines": i.lines} for m, i in modules.items()},
        "edges": [e.__dict__ for e in sorted(all_edges, key=lambda e: (e.importer, e.imported, e.line, e.kind))],
        "strongly_connected_components": [{"modules": c, "cycle_path": c + [c[0]]} for c in strongly_connected(graph)],
        "unresolved_dynamic_imports": sorted(unresolved_dynamic, key=lambda d: (str(d["importer"]), int(d["line"]))),
        "external_dependencies": {k: sorted(v) for k, v in sorted(externals_by_module.items()) if v},
    }
    return {"baseline": baseline, "rows": rows, "graph": graph_json, "class_counts": Counter(r["classification"] for r in rows), "package_edges": {k: sorted(v) for k, v in sorted(package_edges.items())}, "forbidden": forbidden}


def write_outputs(data: dict[str, object]) -> None:
    arch = REPO_ROOT / "docs" / "architecture"
    csv_path = arch / "package-module-map-1.0.13.csv"
    json_path = arch / "package-import-graph-1.0.13.json"
    inv_path = arch / "package-inventory-1.0.13.md"
    findings_path = arch / "package-dependency-findings-1.0.13.md"
    rows = data["rows"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    json_path.write_text(json.dumps(data["graph"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = "\n".join(f"- {k}: {v}" for k, v in sorted(data["class_counts"].items()))
    forbidden = data["forbidden"]
    inv_path.write_text(f"""# ZeroModel 1.0.13 Package Inventory

Baseline commit: `{data['baseline']}`

Generated artifacts:

- `docs/architecture/package-module-map-1.0.13.csv`
- `docs/architecture/package-import-graph-1.0.13.json`
- `docs/architecture/package-dependency-findings-1.0.13.md`

## Module Count By Classification

{counts}

## Public Root API

`zeromodel/__init__.py` currently re-exports symbols from core, analysis, observation, vision, video, and research/evidence modules. The approved package architecture removes this compatibility surface instead of preserving aliases. See the CSV `public_symbols` and inbound test/example columns for defining-module and consumer evidence.

## Package Build And Data Inventory

Current `pyproject.toml` discovers `zeromodel*`, ships the monolithic `zeromodel` distribution at version `1.0.12`, declares NumPy as the only base runtime dependency, and puts SQLAlchemy, Torch, TorchVision, Transformers, and Pillow behind optional extras. `tool.pytest.ini_options.pythonpath = [\".\"]` means tests can rely on repository-root imports that future wheels must not assume.

## Domain Boundary Inventory

The RMDTO target path is Runtime -> Facade -> Engine -> Service -> Store protocol -> Store implementation -> ORM. Current SQLAlchemy ownership is isolated under `zeromodel/db`; `zeromodel/runtime.py` and `zeromodel/stores` are classified as video and should remain SQLAlchemy-free. Suspicious and forbidden proposed edges are ranked in the dependency findings document.

## Split Analysis

| current module | responsibility fragment | target module | target package | symbols to move | inbound callers | identity/schema risk | recommended split order |
|---|---|---|---|---|---|---|---|
| zeromodel | root compatibility re-exports | package-local `__init__.py` files | core/analysis/observation/vision/video/sqlalchemy | all current `__all__` entries | tests and examples using `from zeromodel import ...` | high: root API removal changes import identity | remove root re-exports after package-local public APIs are declared |

## Architecture Comparison

Allowed target graph: analysis->core; observation->core; vision->observation/core; video->observation/core; sqlalchemy->video/core; research->any production package.

Observed proposed classification graph: `{json.dumps(data['package_edges'], sort_keys=True)}`.

Forbidden proposed edge count: `{len(forbidden)}`.
""", encoding="utf-8")
    finding_lines = ["# ZeroModel 1.0.13 Package Dependency Findings", "", f"Baseline commit: `{data['baseline']}`", ""]
    finding_lines += ["## Blocker", ""]
    if forbidden:
        for i, e in enumerate(forbidden[:20], 1):
            finding_lines += [f"### B{i}. Forbidden proposed package edge `{e.importer}` -> `{e.imported}`", "", f"- Paths/import edge: `{e.importer}` imports `{e.imported}` at line {e.line} ({e.kind}).", "- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.", "- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.", "- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.", "- Blocks Stage 1.0.13A: yes.", ""]
    else:
        finding_lines += ["No blocker forbidden proposed edges detected by the generated classifier.", ""]
    finding_lines += ["## High", "", "- Root `zeromodel/__init__.py` imports heavyweight and research-facing modules at import time, contradicting the lightweight core import requirement. Remedy: remove root compatibility exports during Stage 1.0.13A after package-local APIs are declared.", "", "## Medium", "", "- Optional dependencies are declared globally in one distribution, while modules requiring them are intermingled in the `zeromodel` namespace. Remedy: move dependency-owning implementations to vision, research, or sqlalchemy packages before publishing wheels.", "", "## Low", "", "- CI and release scripts assume one distribution and one `dist/` directory. Remedy: replace with workspace-aware build matrix in Stage 1.0.13H.", ""]
    findings_path.write_text("\n".join(finding_lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--generated-at")
    args = parser.parse_args()
    data = make_inventory(args.generated_at)
    if args.write:
        write_outputs(data)
    else:
        print(json.dumps({"baseline": data["baseline"], "module_count": len(data["rows"]), "class_counts": data["class_counts"]}, default=dict, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
