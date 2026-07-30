from __future__ import annotations

import argparse
import html
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "demos" / "catalog.json"
INVENTORY = ROOT / "demos" / "example-inventory.json"
RESULTS = ROOT / "docs" / "results" / "demos"
SITE_BUILD = ROOT / "build" / "site"

VALID_STATES = {"defined", "measured", "hypothesis"}
VALID_PROFILES = {"fast", "extended", "external", "research"}
VALID_ROLES = {
    "benchmark",
    "documentation",
    "external",
    "integration",
    "public_demo",
    "research",
    "runtime_helper",
    "supporting_runner",
}
VALID_STATUSES = {"planned", "published", "supporting"}
REQUIRED_HEADINGS = (
    "## What this demonstrates",
    "## Why it matters",
    "## Source and package mapping",
    "## Application",
    "## Boundaries and limitations",
    "## Reproduction record",
)
TRACKED_DISTRIBUTIONS = (
    "zeromodel",
    "zeromodel-analysis",
    "zeromodel-observation",
    "zeromodel-vision",
    "zeromodel-video",
    "numpy",
    "jupyter",
    "nbconvert",
    "matplotlib",
    "pillow",
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


@dataclass(frozen=True)
class ExampleEntry:
    path: str
    role: str
    execution_profile: str
    status: str
    demo_ids: tuple[str, ...]
    notes: str


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


def load_inventory() -> tuple[ExampleEntry, ...]:
    document = json.loads(INVENTORY.read_text(encoding="utf-8"))
    return tuple(
        ExampleEntry(
            path=item["path"],
            role=item["role"],
            execution_profile=item["execution_profile"],
            status=item["status"],
            demo_ids=tuple(item.get("demo_ids", [])),
            notes=item["notes"],
        )
        for item in document["entries"]
    )


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


def validate_inventory(
    demos: Iterable[Demo], entries: Iterable[ExampleEntry]
) -> list[str]:
    demos = tuple(demos)
    entries = tuple(entries)
    errors: list[str] = []
    demo_ids = {demo.id for demo in demos}
    by_path = {entry.path: entry for entry in entries}

    if len(by_path) != len(entries):
        errors.append("example inventory contains duplicate paths")

    for entry in entries:
        if not (ROOT / entry.path).is_file():
            errors.append(f"inventory: missing path {entry.path}")
        if entry.role not in VALID_ROLES:
            errors.append(f"inventory: invalid role for {entry.path}")
        if entry.execution_profile not in VALID_PROFILES:
            errors.append(f"inventory: invalid profile for {entry.path}")
        if entry.status not in VALID_STATUSES:
            errors.append(f"inventory: invalid status for {entry.path}")
        if entry.status == "published" and not entry.demo_ids:
            errors.append(f"inventory: published path lacks a demo id: {entry.path}")
        unknown = sorted(set(entry.demo_ids) - demo_ids)
        if unknown:
            errors.append(
                f"inventory: unknown demo ids for {entry.path}: {unknown}"
            )

    for demo in demos:
        for source in demo.source_examples:
            entry = by_path.get(source)
            if entry is None:
                errors.append(f"{demo.id}: source is absent from example inventory: {source}")
            elif demo.id not in entry.demo_ids:
                errors.append(
                    f"{demo.id}: inventory source is not linked to demo: {source}"
                )
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
    entries = load_inventory()
    errors.extend(validate_inventory(demos, entries))
    if errors:
        raise SystemExit("Demo validation failed:\n- " + "\n- ".join(errors))
    print(
        f"Validated {len(demos)} executable demos and "
        f"{len(entries)} classified example entrypoints"
    )


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


def _environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in TRACKED_DISTRIBUTIONS:
        try:
            packages[name] = distribution_version(name)
        except PackageNotFoundError:
            continue
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "packages": packages,
    }


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
        "environment": _environment(),
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
    demos = tuple(demos)
    cards: list[str] = []
    for demo in demos:
        applications = " · ".join(map(html.escape, demo.applications))
        cards.append(
            "<article data-state='"
            + html.escape(demo.evidence_state)
            + "' data-profile='"
            + html.escape(demo.execution_profile)
            + "' data-packages='"
            + html.escape(" ".join(demo.packages))
            + "'><p class='badge'>"
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
            + "</p><p><strong>Applications:</strong> "
            + applications
            + "</p><a href='"
            + html.escape(demo.id)
            + "/'>Open demonstration →</a></article>"
        )
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZeroModel demonstrations</title><style>
body{margin:0;background:#0c1015;color:#eef3f8;font:16px system-ui}
main{max-width:1180px;margin:auto;padding:64px 24px}a{color:inherit}
h1{font-size:clamp(2.6rem,7vw,5.4rem);line-height:.96;max-width:900px}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin:32px 0}
button{background:#121922;color:#eef3f8;border:1px solid #2a3541;border-radius:999px;padding:10px 14px}
button[aria-pressed="true"]{background:#eef3f8;color:#0c1015}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}
article{background:#121922;border:1px solid #2a3541;border-radius:18px;padding:24px}
article p{color:#b9c4cf}.badge{text-transform:uppercase;letter-spacing:.08em}
.meta{display:flex;gap:18px;flex-wrap:wrap;color:#b9c4cf}
</style></head><body><main><a href="../">← ZeroModel</a>
<p>EXECUTABLE EVIDENCE CATALOGUE</p>
<h1>See what ZeroModel does, then inspect how it did it.</h1>
<p>Every page is generated from an executed notebook linked to production code.</p>
<p class="meta"><span>""" + str(len(demos)) + """ published demos</span>
<a href="inventory/">Browse the example inventory →</a></p>
<div class="toolbar" aria-label="Filter demonstrations">
<button type="button" data-filter="all" aria-pressed="true">All</button>
<button type="button" data-filter="defined" aria-pressed="false">Defined</button>
<button type="button" data-filter="measured" aria-pressed="false">Measured</button>
<button type="button" data-filter="core" aria-pressed="false">Core</button>
<button type="button" data-filter="analysis" aria-pressed="false">Analysis</button>
<button type="button" data-filter="vision" aria-pressed="false">Vision</button>
<button type="button" data-filter="video" aria-pressed="false">Video</button>
</div><section class="grid">""" + "".join(cards) + """</section></main>
<script>
const buttons=[...document.querySelectorAll("[data-filter]")];
const cards=[...document.querySelectorAll("article")];
for(const button of buttons){button.addEventListener("click",()=>{
 const filter=button.dataset.filter;
 for(const item of buttons)item.setAttribute("aria-pressed",String(item===button));
 for(const card of cards){
  const show=filter==="all"||card.dataset.state===filter||
   card.dataset.profile===filter||card.dataset.packages.split(" ").includes(filter);
  card.hidden=!show;
 }
});}
</script></body></html>"""


def _inventory_html(entries: Iterable[ExampleEntry]) -> str:
    entries = tuple(sorted(entries, key=lambda entry: (entry.status, entry.role, entry.path)))
    rows = "".join(
        "<tr><td><code>"
        + html.escape(entry.path)
        + "</code></td><td>"
        + html.escape(entry.role)
        + "</td><td>"
        + html.escape(entry.execution_profile)
        + "</td><td>"
        + html.escape(entry.status)
        + "</td><td>"
        + html.escape(" · ".join(entry.demo_ids) or "—")
        + "</td><td>"
        + html.escape(entry.notes)
        + "</td></tr>"
        for entry in entries
    )
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZeroModel example inventory</title><style>
body{margin:0;background:#0c1015;color:#eef3f8;font:15px system-ui}
main{max-width:1400px;margin:auto;padding:48px 20px}a{color:inherit}
table{width:100%;border-collapse:collapse;background:#121922}
th,td{padding:12px;border:1px solid #2a3541;text-align:left;vertical-align:top}
th{background:#17212c}code{font-size:.88em}p{color:#b9c4cf}
</style></head><body><main><a href="../">← Demonstrations</a>
<h1>Example inventory</h1>
<p>Each entrypoint is classified before it becomes a public notebook. Supporting
runners remain linked to their parent demonstration rather than becoming duplicate pages.</p>
<table><thead><tr><th>Path</th><th>Role</th><th>Profile</th><th>Status</th>
<th>Demo</th><th>Notes</th></tr></thead><tbody>""" + rows + """</tbody></table>
</main></body></html>"""


def _inject_demo_links() -> None:
    index = SITE_BUILD / "index.html"
    source = index.read_text(encoding="utf-8")
    replacements = {
        '      <a href="#evidence">Evidence</a>':
            '      <a href="#evidence">Evidence</a>\n'
            '      <a href="demos/">Demonstrations</a>',
        '<a class="button secondary" href="https://github.com/ernanhughes/zeromodel/blob/main/docs/spec/vpm-artifact-v0.md">Read the draft spec</a>':
            '<a class="button secondary" href="demos/">Browse executed demos</a>',
        '<span class="pending">Benchmarks being rebuilt</span>':
            '<a href="demos/">Open executable evidence</a>',
        '<a class="button primary" href="https://github.com/ernanhughes/zeromodel">View the rebuild on GitHub</a>':
            '<a class="button primary" href="demos/">Browse demonstrations</a>',
    }
    for old, new in replacements.items():
        if old not in source:
            raise SystemExit(f"site/index.html is missing expected demo-link hook: {old}")
        source = source.replace(old, new, 1)
    index.write_text(source, encoding="utf-8")


def build_site(demos: Iterable[Demo]) -> None:
    demos = tuple(demos)
    entries = load_inventory()
    if SITE_BUILD.exists():
        shutil.rmtree(SITE_BUILD)
    shutil.copytree(ROOT / "site", SITE_BUILD)
    _inject_demo_links()
    target = SITE_BUILD / "demos"
    target.mkdir(parents=True)
    payload = [asdict(demo) | {"url": f"{demo.id}/"} for demo in demos]
    (target / "catalog.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target / "inventory.json").write_text(
        json.dumps([asdict(entry) for entry in entries], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (target / "index.html").write_text(_catalog_html(demos), encoding="utf-8")
    inventory_target = target / "inventory"
    inventory_target.mkdir()
    (inventory_target / "index.html").write_text(
        _inventory_html(entries), encoding="utf-8"
    )

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
