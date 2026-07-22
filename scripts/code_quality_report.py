from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "quality-baseline.toml"
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pypackages__",
    "__pycache__",
    "build",
    "dist",
    "env",
    "ENV",
    "venv",
}
EXCLUDED_PREFIXES = ("docs/results/",)
NESTING_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.ExceptHandler,
    ast.Match,
)


class ConfigError(ValueError):
    """Raised when the quality baseline cannot be trusted."""


@dataclass(frozen=True)
class LegacyException:
    path: str
    reason: str
    maximum_lines: int
    owner: str

    def as_dict(self) -> dict[str, object]:
        return {
            "maximum_lines": self.maximum_lines,
            "owner": self.owner,
            "path": self.path,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class QualityConfig:
    new_module_max_lines: int
    module_warning_lines: int
    new_function_max_lines: int
    function_warning_lines: int
    new_class_max_lines: int
    maximum_function_parameters: int
    maximum_ast_nesting: int
    legacy_exceptions: dict[str, LegacyException]

    def quality_thresholds(self) -> dict[str, int]:
        return {
            "function_warning_lines": self.function_warning_lines,
            "maximum_ast_nesting": self.maximum_ast_nesting,
            "maximum_function_parameters": self.maximum_function_parameters,
            "module_warning_lines": self.module_warning_lines,
            "new_class_max_lines": self.new_class_max_lines,
            "new_function_max_lines": self.new_function_max_lines,
            "new_module_max_lines": self.new_module_max_lines,
        }


@dataclass(frozen=True)
class FunctionMetric:
    name: str
    qualname: str
    line: int
    line_count: int
    positional_parameters: int
    keyword_only_parameters: int
    total_parameters: int

    def as_dict(self) -> dict[str, object]:
        return {
            "keyword_only_parameters": self.keyword_only_parameters,
            "line": self.line,
            "line_count": self.line_count,
            "name": self.name,
            "positional_parameters": self.positional_parameters,
            "qualname": self.qualname,
            "total_parameters": self.total_parameters,
        }


@dataclass(frozen=True)
class ClassMetric:
    name: str
    qualname: str
    line: int
    line_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "line": self.line,
            "line_count": self.line_count,
            "name": self.name,
            "qualname": self.qualname,
        }


@dataclass(frozen=True)
class FileMetric:
    path: str
    physical_line_count: int
    nonblank_line_count: int
    top_level_function_count: int
    top_level_class_count: int
    functions: list[FunctionMetric]
    classes: list[ClassMetric]
    maximum_ast_nesting: int
    local_imports: list[str]


def node_line_count(node: ast.AST) -> int:
    end_lineno = getattr(node, "end_lineno", None)
    lineno = getattr(node, "lineno", None)
    if not isinstance(end_lineno, int) or not isinstance(lineno, int):
        return 0
    return end_lineno - lineno + 1


def strip_inline_comment(line: str) -> str:
    in_string = False
    escaped = False
    for index, char in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "#":
            return line[:index]
    return line


def parse_toml_string(raw: str, line_number: int) -> str:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ConfigError(f"line {line_number}: invalid string value {raw!r}") from exc
    if not isinstance(value, str):
        raise ConfigError(f"line {line_number}: expected a string value")
    return value


def parse_toml_scalar(raw: str, line_number: int) -> int | str:
    value = raw.strip()
    if value.startswith('"') and value.endswith('"'):
        return parse_toml_string(value, line_number)
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    raise ConfigError(f"line {line_number}: unsupported TOML scalar {value!r}")


def parse_section(raw: str, line_number: int) -> tuple[str, str | None]:
    if not raw.startswith("[") or not raw.endswith("]"):
        raise ConfigError(f"line {line_number}: invalid TOML section {raw!r}")
    section = raw[1:-1].strip()
    if section == "quality":
        return ("quality", None)
    legacy_prefix = "legacy_exceptions."
    if section.startswith(legacy_prefix):
        path_raw = section[len(legacy_prefix) :].strip()
        path = parse_toml_string(path_raw, line_number)
        return ("legacy_exceptions", normalize_path(path))
    raise ConfigError(f"line {line_number}: unknown section [{section}]")


def parse_baseline_file(path: Path) -> QualityConfig:
    if not path.exists():
        raise ConfigError(f"quality baseline is missing: {path.relative_to(REPO_ROOT)}")

    current_section: tuple[str, str | None] | None = None
    quality: dict[str, int | str] = {}
    legacy_sections: dict[str, dict[str, int | str]] = {}

    for line_number, original_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = strip_inline_comment(original_line).strip()
        if not line:
            continue
        if line.startswith("["):
            current_section = parse_section(line, line_number)
            if (
                current_section[0] == "legacy_exceptions"
                and current_section[1] is not None
            ):
                legacy_sections.setdefault(current_section[1], {})
            continue
        if current_section is None:
            raise ConfigError(f"line {line_number}: key outside a section")
        if "=" not in line:
            raise ConfigError(f"line {line_number}: expected key = value")
        key, raw_value = [part.strip() for part in line.split("=", 1)]
        value = parse_toml_scalar(raw_value, line_number)
        if current_section[0] == "quality":
            quality[key] = value
        elif current_section[1] is not None:
            legacy_sections[current_section[1]][key] = value

    return build_quality_config(quality, legacy_sections)


def build_quality_config(
    quality: dict[str, int | str],
    legacy_sections: dict[str, dict[str, int | str]],
) -> QualityConfig:
    required_quality_keys = {
        "function_warning_lines",
        "maximum_ast_nesting",
        "maximum_function_parameters",
        "module_warning_lines",
        "new_class_max_lines",
        "new_function_max_lines",
        "new_module_max_lines",
    }
    extra_quality_keys = set(quality) - required_quality_keys
    missing_quality_keys = required_quality_keys - set(quality)
    if extra_quality_keys:
        raise ConfigError(f"unknown quality keys: {sorted(extra_quality_keys)}")
    if missing_quality_keys:
        raise ConfigError(f"missing quality keys: {sorted(missing_quality_keys)}")

    quality_values: dict[str, int] = {}
    for key in sorted(required_quality_keys):
        value = quality[key]
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(f"quality.{key} must be a positive integer")
        quality_values[key] = value

    legacy_exceptions: dict[str, LegacyException] = {}
    for exception_path, values in sorted(legacy_sections.items()):
        reason = values.get("reason")
        maximum_lines = values.get("maximum_lines")
        owner = values.get("owner")
        if not isinstance(reason, str) or not reason.strip():
            raise ConfigError(f"{exception_path}: legacy exception requires a reason")
        if not isinstance(maximum_lines, int) or maximum_lines <= 0:
            raise ConfigError(
                f"{exception_path}: legacy exception requires a positive maximum_lines"
            )
        if not isinstance(owner, str) or not owner.strip():
            raise ConfigError(f"{exception_path}: legacy exception requires an owner")
        legacy_exceptions[exception_path] = LegacyException(
            path=exception_path,
            reason=reason,
            maximum_lines=maximum_lines,
            owner=owner,
        )

    return QualityConfig(legacy_exceptions=legacy_exceptions, **quality_values)


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def relative_path(path: Path) -> str:
    return normalize_path(path.relative_to(REPO_ROOT).as_posix())


def is_excluded(path: Path) -> bool:
    rel = relative_path(path)
    if any(part in EXCLUDED_PARTS for part in path.relative_to(REPO_ROOT).parts):
        return True
    return rel.startswith(EXCLUDED_PREFIXES)


def discover_python_files(roots: Sequence[Path] | None = None) -> list[Path]:
    search_roots = roots or [REPO_ROOT]
    files: list[Path] = []
    for root in search_roots:
        resolved = root if root.is_absolute() else REPO_ROOT / root
        if resolved.is_file() and resolved.suffix == ".py":
            if not is_excluded(resolved):
                files.append(resolved)
        elif resolved.is_dir():
            files.extend(
                path for path in resolved.rglob("*.py") if not is_excluded(path)
            )
    return sorted(files, key=relative_path)


def module_name_for_path(path: str) -> str:
    parts = Path(path).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def package_name_for_module(module_name: str, path: str) -> str:
    if path.endswith("/__init__.py"):
        return module_name
    if "." not in module_name:
        return ""
    return module_name.rsplit(".", 1)[0]


def resolve_relative_import(
    module_name: str, path: str, level: int, imported: str | None
) -> str:
    package = package_name_for_module(module_name, path)
    package_parts = package.split(".") if package else []
    if level > len(package_parts):
        return imported or ""
    base_parts = package_parts[: len(package_parts) - level + 1]
    if imported:
        base_parts.extend(imported.split("."))
    return ".".join(part for part in base_parts if part)


def collect_local_imports(tree: ast.AST, path: str) -> list[str]:
    module_name = module_name_for_path(path)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "zeromodel" or alias.name.startswith("zeromodel."):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                module = resolve_relative_import(
                    module_name, path, node.level, node.module
                )
            else:
                module = node.module or ""
            if module == "zeromodel":
                for alias in node.names:
                    if alias.name == "*":
                        imports.add(module)
                    else:
                        imports.add(f"{module}.{alias.name}")
            elif module.startswith("zeromodel."):
                imports.add(module)
    return sorted(imports)


def statement_child_bodies(node: ast.stmt) -> list[list[ast.stmt]]:
    bodies: list[list[ast.stmt]] = []
    for _field_name, value in ast.iter_fields(node):
        if isinstance(value, list) and all(
            isinstance(item, ast.stmt) for item in value
        ):
            bodies.append(value)
        elif isinstance(value, ast.stmt):
            bodies.append([value])
    return bodies


def function_metric(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: tuple[str, ...],
) -> FunctionMetric:
    positional_parameters = len(node.args.posonlyargs) + len(node.args.args)
    keyword_only_parameters = len(node.args.kwonlyargs)
    total_parameters = positional_parameters + keyword_only_parameters
    if node.args.vararg is not None:
        total_parameters += 1
    if node.args.kwarg is not None:
        total_parameters += 1
    return FunctionMetric(
        name=node.name,
        qualname=".".join([*parents, node.name]),
        line=node.lineno,
        line_count=node_line_count(node),
        positional_parameters=positional_parameters,
        keyword_only_parameters=keyword_only_parameters,
        total_parameters=total_parameters,
    )


def class_metric(node: ast.ClassDef, parents: tuple[str, ...]) -> ClassMetric:
    return ClassMetric(
        name=node.name,
        qualname=".".join([*parents, node.name]),
        line=node.lineno,
        line_count=node_line_count(node),
    )


def collect_definitions(
    tree: ast.Module,
) -> tuple[list[FunctionMetric], list[ClassMetric]]:
    functions: list[FunctionMetric] = []
    classes: list[ClassMetric] = []
    pending: list[tuple[ast.stmt, tuple[str, ...]]] = [
        (node, ()) for node in reversed(tree.body)
    ]

    while pending:
        node, parents = pending.pop()
        if isinstance(node, ast.ClassDef):
            classes.append(class_metric(node, parents))
            child_parents = (*parents, node.name)
            pending.extend((child, child_parents) for child in reversed(node.body))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(function_metric(node, parents))
            child_parents = (*parents, node.name)
            pending.extend((child, child_parents) for child in reversed(node.body))
        else:
            for body in reversed(statement_child_bodies(node)):
                pending.extend((child, parents) for child in reversed(body))

    return functions, classes


def max_ast_nesting(node: ast.AST, depth: int = 0) -> int:
    maximum = depth
    pending = [(node, depth)]
    while pending:
        current, current_depth = pending.pop()
        next_depth = (
            current_depth + 1 if isinstance(current, NESTING_NODES) else current_depth
        )
        maximum = max(maximum, next_depth)
        pending.extend((child, next_depth) for child in ast.iter_child_nodes(current))
    return maximum


def inspect_file(path: Path) -> FileMetric:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text, filename=relative_path(path))
    functions, classes = collect_definitions(tree)

    top_level_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    top_level_classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    return FileMetric(
        path=relative_path(path),
        physical_line_count=len(lines),
        nonblank_line_count=sum(1 for line in lines if line.strip()),
        top_level_function_count=len(top_level_functions),
        top_level_class_count=len(top_level_classes),
        functions=sorted(functions, key=lambda metric: (metric.line, metric.qualname)),
        classes=sorted(classes, key=lambda metric: (metric.line, metric.qualname)),
        maximum_ast_nesting=max_ast_nesting(tree),
        local_imports=collect_local_imports(tree, relative_path(path)),
    )


def limit_findings(
    metric: FileMetric, config: QualityConfig
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    hard_limits: list[str] = []

    if metric.physical_line_count > config.module_warning_lines:
        warnings.append(
            f"module has {metric.physical_line_count} lines "
            f"> warning threshold {config.module_warning_lines}"
        )
    if metric.physical_line_count > config.new_module_max_lines:
        hard_limits.append(
            f"module has {metric.physical_line_count} lines "
            f"> hard limit {config.new_module_max_lines}"
        )

    for function in metric.functions:
        if function.line_count > config.function_warning_lines:
            warnings.append(
                f"function {function.qualname} has {function.line_count} lines "
                f"> warning threshold {config.function_warning_lines}"
            )
        if function.line_count > config.new_function_max_lines:
            hard_limits.append(
                f"function {function.qualname} has {function.line_count} lines "
                f"> hard limit {config.new_function_max_lines}"
            )
        if function.total_parameters > config.maximum_function_parameters:
            hard_limits.append(
                f"function {function.qualname} has {function.total_parameters} parameters "
                f"> hard limit {config.maximum_function_parameters}"
            )

    for class_metric in metric.classes:
        if class_metric.line_count > config.new_class_max_lines:
            hard_limits.append(
                f"class {class_metric.qualname} has {class_metric.line_count} lines "
                f"> hard limit {config.new_class_max_lines}"
            )

    if metric.maximum_ast_nesting > config.maximum_ast_nesting:
        hard_limits.append(
            f"maximum AST nesting is {metric.maximum_ast_nesting} "
            f"> hard limit {config.maximum_ast_nesting}"
        )

    return warnings, hard_limits


def metric_as_dict(
    metric: FileMetric,
    warnings: list[str],
    hard_limits: list[str],
    enforcement_errors: list[str],
    legacy_exception: LegacyException | None,
) -> dict[str, object]:
    return {
        "classes": [class_metric.as_dict() for class_metric in metric.classes],
        "enforcement_errors": enforcement_errors,
        "functions": [function.as_dict() for function in metric.functions],
        "hard_limit_excesses": hard_limits,
        "legacy_exception": legacy_exception.as_dict() if legacy_exception else None,
        "local_zeromodel_imports": metric.local_imports,
        "maximum_ast_nesting": metric.maximum_ast_nesting,
        "nonblank_line_count": metric.nonblank_line_count,
        "path": metric.path,
        "physical_line_count": metric.physical_line_count,
        "top_level_class_count": metric.top_level_class_count,
        "top_level_function_count": metric.top_level_function_count,
        "warnings": warnings,
    }


def build_report(
    config: QualityConfig, roots: Sequence[Path] | None = None
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    errors: list[str] = []
    seen_paths: set[str] = set()
    warning_count = 0
    hard_limit_file_count = 0
    ceiling_violation_count = 0

    for path in discover_python_files(roots):
        rel = relative_path(path)
        seen_paths.add(rel)
        try:
            metric = inspect_file(path)
        except SyntaxError as exc:
            line = exc.lineno or 0
            errors.append(f"{rel}:{line}: Python source cannot be parsed: {exc.msg}")
            continue
        warnings, hard_limits = limit_findings(metric, config)
        warning_count += len(warnings)
        if hard_limits:
            hard_limit_file_count += 1

        legacy_exception = config.legacy_exceptions.get(rel)
        enforcement_errors: list[str] = []
        if legacy_exception is not None:
            if metric.physical_line_count > legacy_exception.maximum_lines:
                ceiling_violation_count += 1
                enforcement_errors.append(
                    f"legacy ceiling exceeded: {metric.physical_line_count} lines "
                    f"> maximum_lines {legacy_exception.maximum_lines}"
                )
        else:
            enforcement_errors.extend(hard_limits)
        errors.extend(f"{rel}: {message}" for message in enforcement_errors)

        files.append(
            metric_as_dict(
                metric=metric,
                warnings=warnings,
                hard_limits=hard_limits,
                enforcement_errors=enforcement_errors,
                legacy_exception=legacy_exception,
            )
        )

    if roots is None:
        missing_exceptions = sorted(set(config.legacy_exceptions) - seen_paths)
        for exception_path in missing_exceptions:
            errors.append(
                f"{exception_path}: legacy exception references a missing Python source file"
            )

    return {
        "config": config.quality_thresholds(),
        "errors": errors,
        "files": files,
        "legacy_exceptions": [
            exception.as_dict()
            for exception in sorted_exceptions(config.legacy_exceptions)
        ],
        "summary": {
            "enforcement_error_count": len(errors),
            "file_count": len(files),
            "files_exceeding_hard_limits": hard_limit_file_count,
            "legacy_ceiling_violation_count": ceiling_violation_count,
            "legacy_exception_count": len(config.legacy_exceptions),
            "warning_count": warning_count,
        },
    }


def sorted_exceptions(exceptions: dict[str, LegacyException]) -> list[LegacyException]:
    return [exceptions[path] for path in sorted(exceptions)]


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def markdown_report(report: dict[str, object]) -> str:
    summary = require_mapping(report["summary"], "summary")
    files = require_list(report["files"], "files")
    errors = require_list(report["errors"], "errors")
    legacy_exceptions = require_list(report["legacy_exceptions"], "legacy_exceptions")

    lines = [
        "# ZeroModel Code Quality Report",
        "",
        "## Summary",
        f"- Python files: {summary['file_count']}",
        f"- Warning findings: {summary['warning_count']}",
        f"- Files exceeding hard thresholds: {summary['files_exceeding_hard_limits']}",
        f"- Legacy exceptions: {summary['legacy_exception_count']}",
        f"- Legacy ceiling violations: {summary['legacy_ceiling_violation_count']}",
        f"- Enforcement errors: {summary['enforcement_error_count']}",
        "",
        "## Legacy Exceptions",
    ]

    if legacy_exceptions:
        for exception in legacy_exceptions:
            item = require_mapping(exception, "legacy exception")
            lines.append(
                f"- `{item['path']}`: ceiling {item['maximum_lines']} lines; "
                f"owner `{item['owner']}`; {item['reason']}"
            )
    else:
        lines.append("- None")

    warning_files = files_with_entries(files, "warnings")
    hard_limit_files = files_with_entries(files, "hard_limit_excesses")
    lines.extend(["", "## Warning Thresholds"])
    lines.extend(format_file_findings(warning_files, "warnings"))
    lines.extend(["", "## Hard Thresholds"])
    lines.extend(format_file_findings(hard_limit_files, "hard_limit_excesses"))
    lines.extend(["", "## Enforcement"])
    if errors:
        for error in errors:
            lines.append(f"- {error}")
    else:
        lines.append("- No enforcement errors")

    return "\n".join(lines) + "\n"


def require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} is not a mapping")
    return value


def require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} is not a list")
    return value


def files_with_entries(files: list[object], key: str) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for file_record in files:
        record = require_mapping(file_record, "file record")
        entries = require_list(record[key], key)
        if entries:
            selected.append(record)
    return selected


def format_file_findings(file_records: list[dict[str, object]], key: str) -> list[str]:
    if not file_records:
        return ["- None"]
    lines: list[str] = []
    for record in file_records:
        lines.append(f"- `{record['path']}`")
        entries = require_list(record[key], key)
        for entry in entries:
            lines.append(f"  - {entry}")
    return lines


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the ZeroModel code quality report."
    )
    parser.add_argument(
        "--json", type=Path, help="Write deterministic JSON report to this path."
    )
    parser.add_argument(
        "--markdown", type=Path, help="Write Markdown report to this path."
    )
    parser.add_argument(
        "--path",
        action="append",
        type=Path,
        dest="paths",
        help="Limit the report to a file or directory. May be passed more than once.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        config = parse_baseline_file(CONFIG_PATH)
        report = build_report(config, args.paths)
    except ConfigError as exc:
        print(f"Quality configuration error: {exc}", file=sys.stderr)
        return 2

    markdown = markdown_report(report)
    if args.json is not None:
        write_report(args.json, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown is not None:
        write_report(args.markdown, markdown)
    if args.json is None and args.markdown is None:
        print(markdown, end="")
    else:
        summary = require_mapping(report["summary"], "summary")
        print(
            "Quality report: "
            f"{summary['file_count']} files, "
            f"{summary['warning_count']} warnings, "
            f"{summary['files_exceeding_hard_limits']} files over hard thresholds, "
            f"{summary['legacy_exception_count']} legacy exceptions"
        )

    errors = require_list(report["errors"], "errors")
    if errors:
        print("Quality report: failed", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
