"""Video action-set mutation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Sequence,
)
from research.video_action_set.mutation_audit import (
    build_mutation_audit_payload as _build_mutation_audit_payload,
    build_repeated_mutation_audit_payload as _build_repeated_mutation_audit_payload,
    _changed_snapshot_files,
    evaluate_mutation_case as _evaluate_mutation_case,
    _mutation_isolation_report,
)
from research.video_action_set.mutation_matrix import (
    MUTATION_MATRIX_VERSION,
    mutation_catalogue,
    validate_mutation_catalogue,
)
from research.video_action_set.reference_verification import _finding
from research.video_action_set.verification import (
    build_unavailable_repeated_mutation_audit as _build_unavailable_repeated_mutation_audit,
    build_verification_closure as _build_verification_closure,
    verification_summary as _verification_summary,
)
from zeromodel.video.domains.video_action_set.artifact_io import (
    _read_json,
    _sha256,
)
from research.video_action_set.mutation_filesystem import (
    _apply_reference_mutation,
    _directory_snapshot,
    _mutation_structural_snapshot,
)
from research.video_action_set.verification_orchestration import (
    verify_reference_instrument,
    verify_reference_read_only,
)


def run_reference_mutation_audit(
    output_dir: Path, repo_root: Path, *, mutation_names: Sequence[str] | None = None
) -> dict[str, Any]:
    import shutil
    import tempfile

    requested = (
        None if mutation_names is None else {str(name) for name in mutation_names}
    )
    catalogue = mutation_catalogue()
    selected_cases = tuple(
        case
        for case in catalogue
        if requested is None or case["mutation_id"] in requested
    )
    catalogue_findings = validate_mutation_catalogue()
    if requested is not None:
        selected_names = {str(case["mutation_id"]) for case in selected_cases}
        catalogue_findings.extend(
            _finding(
                "mutation_not_declared",
                "requested mutation is not declared",
                mutation=name,
            )
            for name in sorted(requested - selected_names)
        )
    base = verify_reference_instrument(output_dir, repo_root)
    if not base["verified"] or catalogue_findings:
        return _build_mutation_audit_payload(
            matrix_version=MUTATION_MATRIX_VERSION,
            catalogue=catalogue,
            selected_cases=selected_cases,
            catalogue_findings=catalogue_findings,
            results=(),
            base_verified=False,
            base_primary_failure_code=base.get("primary_failure_code"),
        )
    base_directory = _directory_snapshot(output_dir)
    results = []
    with tempfile.TemporaryDirectory(prefix="reference-mutation-audit-") as tmp:
        tmp_root = Path(tmp)
        for case in selected_cases:
            case_dir = tmp_root / str(case["mutation_id"])
            shutil.copytree(output_dir, case_dir)
            application_error = None
            try:
                _apply_reference_mutation(case_dir, str(case["mutation_id"]))
                changed_files = _changed_snapshot_files(
                    base_directory, _directory_snapshot(case_dir)
                )
                before = _mutation_structural_snapshot(
                    output_dir, only_files=changed_files
                )
                after = _mutation_structural_snapshot(
                    case_dir, only_files=changed_files
                )
                isolation = _mutation_isolation_report(before, after, case)
                report = verify_reference_instrument(
                    case_dir,
                    repo_root,
                    enabled_gates=case["validation_metadata"]["gate_scope"],
                    stop_after_first_failure=True,
                )
            except Exception as exc:  # pragma: no cover - historical audit boundary.
                application_error = type(exc).__name__
                isolation = {
                    "changed_fields": [],
                    "expected_changed_files": list(
                        case.get("expected_changed_files", [])
                    ),
                    "unexpected_changed_fields": [],
                    "changed_field_count": 0,
                    "isolation_passed": False,
                    "mutation_effect_digest": "sha256:application-error",
                }
                report = {"gates": []}
            results.append(
                _evaluate_mutation_case(
                    case=case,
                    report=report,
                    isolation=isolation,
                    application_error=application_error,
                )
            )
    return _build_mutation_audit_payload(
        matrix_version=MUTATION_MATRIX_VERSION,
        catalogue=catalogue,
        selected_cases=selected_cases,
        catalogue_findings=catalogue_findings,
        results=results,
    )


def run_repeated_reference_mutation_audit(
    output_dir: Path, repo_root: Path, *, mutation_names: Sequence[str] | None = None
) -> dict[str, Any]:
    first = run_reference_mutation_audit(
        output_dir, repo_root, mutation_names=mutation_names
    )
    second = run_reference_mutation_audit(
        output_dir, repo_root, mutation_names=mutation_names
    )
    return _build_repeated_mutation_audit_payload(
        matrix_version=MUTATION_MATRIX_VERSION, first=first, second=second
    )


def build_reference_closure_report(
    output_dir: Path, repo_root: Path, *, include_mutation_audit: bool = True
) -> dict[str, Any]:
    verification = verify_reference_instrument(output_dir, repo_root)
    repeated = (
        run_repeated_reference_mutation_audit(output_dir, repo_root)
        if include_mutation_audit
        else _build_unavailable_repeated_mutation_audit()
    )
    read_only = verify_reference_read_only(output_dir, repo_root)
    episode_plan_path = output_dir / "episode-plan.json"
    plans = (
        _read_json(episode_plan_path).get("splits", {})
        if episode_plan_path.exists()
        else {}
    )
    return _build_verification_closure(
        verification=verification,
        repeated_mutation_audit=repeated,
        read_only=read_only,
        split_plan_identities={
            split: _sha256(context) for split, context in plans.items()
        },
    )


def verify_instrument(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    closure = build_reference_closure_report(
        output_dir, repo_root, include_mutation_audit=False
    )
    return _verification_summary(closure)


def _run_adversarial_mutation_checks(output_dir: Path) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    audit = run_reference_mutation_audit(output_dir, repo_root)
    return [
        row["mutation"]
        for row in audit.get("mutations", [])
        if not row.get("expected_code_matched", False)
    ]
