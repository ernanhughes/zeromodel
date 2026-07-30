from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "demos" / "catalog.json"
RESULTS = ROOT / "docs" / "results" / "demos"
SITE_BUILD = ROOT / "build" / "site"
VALID_STATES = {"defined", "measured", "hypothesis"}
VALID_PROFILES = {"fast", "extended", "external", "research"}
REQUIRED_HEADINGS = (
    "## What this demonstrates",
    "## Why it matters",
    "## Source and package mapping",
    "## Application",
    "## Boundaries and limitations",
    "## Reproduction record",
)


@dataclass(frozen=True)
class Demo:
    id: str
    title: str
    summary: str
    notebook: str
    source_examples: tuple[str, ...]
    packages: tuple[str, ...]
    evidence_state: str
    execution_profile: str
    network: bool
    timeout_seconds: int
    website_order: int
    applications: tuple[str, ...]


def load_catalog() -> tuple[Demo, ...]:
    document = json.loads(CATALOG.read_text(encoding="utf-8"))
    demos = [
        Demo(
            id=item["id"],
            title=item["title"],
            summary=item["summary"],
            notebook=item["notebook"],
            source_examples=tuple(item["source_examples"]),
            packages=tuple(item["packages"]),
            evidence_state=item["evidence_state"],
            execution_profile=item["execution_profile"],
            network=bool(item["network"]),
            timeout_seconds=int(item["timeout_seconds"]),
            website_order=int(item["website_order"]),
            applications=tuple(item["applications"]),
        )
        for item in document["demos"]
    ]
    return tuple(sorted(demos, key=lambda demo: (demo.website_order, demo.id)))


def _markdown(notebook: dict[str, Any]) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    )


def validate_demo(demo: Demo) -> list[str]:
    errors: list[str] = []
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-"
    if not demo.id or any(character not in allowed for character in demo.id):
        errors.append(f"{demo.id!r}: invalid demo id")
    if demo.evidence_state not in VALID_STATES:
        errors.append(f"{demo.id}: invalid evidence state")
    if demo.execution_profile not in VALID_PROFILES:
        errors.append(f"{demo.id}: invalid execution profile")
    if demo.timeout_seconds <= 0:
        errors.append(f"{demo.id}: timeout must be positive")
    for source in demo.source_examples:
        if not (ROOT / source).is_file():
            errors.append(f"{demo.id}: missing source {source}")
    path = ROOT / demo.notebook
    if not path.is_file():
        errors.append(f"{demo.id}: missing notebook {demo.notebook}")
        return errors
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{demo.id}: invalid notebook JSON: {exc}"]
    metadata = notebook.get("metadata", {}).get("zeromodel_demo", {})
    expected = {
        "id": demo.id,
        "evidence_state": demo.evidence_state,
        "execution_profile": demo.execution_profile,
    }
    if metadata != expected:
        errors.append(f"{demo.id}: notebook metadata does not match catalogue")
    markdown = _markdown(notebook)
    for heading in REQUIRED_HEADINGS:
        if heading not in markdown:
            errors.append(f"{demo.id}: missing heading {heading}")
    return errors


def validate(demos: Iterable[Demo]) -> None:
    demos = tuple(demos)
    ids = [demo.id for demo in demos]
    errors = [
        f"duplicate demo id: {demo_id}"
        for demo_id in sorted(set(ids))
        if ids.count(demo_id) > 1
    ]
    for demo in demos:
        errors.extend(validate_demo(demo))
    if errors:
        raise SystemExit("Demo validation failed:\n- " + "\n- ".join(errors))
    print(f"Validated {len(demos)} executable demos")


def _run(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def _revision() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def execute(demo: Demo) -> None:
    if demo.network:
        raise SystemExit(f"{demo.id}: networked demos are not executed automatically")
    output = RESULTS / demo.id
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={demo.timeout_seconds}",
        "--ExecutePreprocessor.kernel_name=python3",
        "--output=executed.ipynb",
        f"--output-dir={output}",
        str(ROOT / demo.notebook),
    ]
    _run(command)
    record = {
        "demo_id": demo.id,
        "source_notebook": demo.notebook,
        "zeromodel_version": (ROOT / "VERSION").read_text().strip(),
        "git_revision": _revision(),
        "command": command,
    }
    (output / "execution.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render(demo: Demo) -> None:
    source = RESULTS / demo.id / "executed.ipynb"
    if not source.is_file():
        raise SystemExit(f"{demo.id}: execute before rendering")
    _run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--HTMLExporter.exclude_input_prompt=True",
            "--HTMLExporter.exclude_output_prompt=True",
            "--output=index.html",
            f"--output-dir={source.parent}",
            str(source),
        ]
    )


def _catalog_html(demos: Iterable[Demo]) -> str:
    cards = []
    for demo in demos:
        cards.append(
            "<article><p class='badge'>"
            + html.escape(demo.evidence_state)
            + " · "
            + html.escape(demo.execution_profile)
            + "</p><h2><a href='"
            + html.escape(demo.id)
            + "/'>"
            + html.escape(demo.title)
            + "</a></h2><p>"
            + html.escape(demo.summary)
            + "</p><p><strong>Packages:</strong> "
            + " · ".join(map(html.escape, demo.packages))
            + "</p><a href='"
            + html.escape(demo.id)
            + "/'>Open demonstration →</a></article>"
        )
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZeroModel demonstrations</title><style>
body{margin:0;background:#0c1015;color:#eef3f8;font:16px system-ui}
main{max-width:1100px;margin:auto;padding:64px 24px}a{color:inherit}
h1{font-size:clamp(2.6rem,7vw,5.4rem);line-height:.96;max-width:850px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}
article{background:#121922;border:1px solid #2a3541;border-radius:18px;padding:24px}
article p{color:#b9c4cf}.badge{text-transform:uppercase;letter-spacing:.08em}
</style></head><body><main><a href="../">← ZeroModel</a>
<p>EXECUTABLE EVIDENCE CATALOGUE</p>
<h1>See what ZeroModel does, then inspect how it did it.</h1>
<p>Every page is generated from an executed notebook linked to production code.</p>
<section class="grid">""" + "".join(cards) + "</section></main></body></html>"


def build_site(demos: Iterable[Demo]) -> None:
    demos = tuple(demos)
    if SITE_BUILD.exists():
        shutil.rmtree(SITE_BUILD)
    shutil.copytree(ROOT / "site", SITE_BUILD)
    target = SITE_BUILD / "demos"
    target.mkdir(parents=True)
    payload = [asdict(demo) | {"url": f"{demo.id}/"} for demo in demos]
    (target / "catalog.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target / "index.html").write_text(_catalog_html(demos), encoding="utf-8")
    for demo in demos:
        source = RESULTS / demo.id / "index.html"
        if not source.is_file():
            raise SystemExit(f"{demo.id}: rendered HTML is missing")
        destination = target / demo.id
        destination.mkdir()
        shutil.copy2(source, destination / "index.html")
        shutil.copy2(RESULTS / demo.id / "execution.json", destination)


def selected(demos: Iterable[Demo], profiles: list[str] | None) -> tuple[Demo, ...]:
    if not profiles:
        return tuple(demos)
    return tuple(demo for demo in demos if demo.execution_profile in profiles)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("validate", "execute", "render", "site", "all")
    )
    parser.add_argument("--profile", action="append", choices=sorted(VALID_PROFILES))
    args = parser.parse_args(argv)
    demos = load_catalog()
    validate(demos)
    chosen = selected(demos, args.profile)
    if not chosen:
        raise SystemExit("No demos matched the selected profile")
    if args.command in {"execute", "all"}:
        for demo in chosen:
            execute(demo)
    if args.command in {"render", "all"}:
        for demo in chosen:
            render(demo)
    if args.command in {"site", "all"}:
        build_site(chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
