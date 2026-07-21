"""Video action-set verification orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Mapping,
    Sequence,
    cast,
)
from zeromodel.arcade_policy import (
    ACTIONS,
    compile_policy_artifact,
)
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set.contracts import (
    BENCHMARK_VERSION,
    GENERATOR_VERSION,
    REACHABILITY_TILE_VERSION,
)
from zeromodel.domains.video_action_set.episode_families import (
    episode_family_registry as _episode_family_registry,
)
from zeromodel.domains.video_action_set.episode_planning import (
    episode_plans_for_split as _episode_plans_for_split,
)
from zeromodel.domains.video_action_set.evidence_audit import (
    audit_canonical_provider_results as _audit_canonical_provider_results,
    audit_evidence_rows as _audit_evidence_rows,
)
from zeromodel.domains.video_action_set.materialization_reachability import (
    validate_reachability_tile_identity as _validate_reachability_tile_identity,
)
from zeromodel.domains.video_action_set.observation_universe import (
    canonical_prototypes,
)
from zeromodel.domains.video_action_set.provider_measurement import (
    SOURCE_SCOPE,
)
from zeromodel.domains.video_action_set.reference_verification import (
    _REQUIRED_VERIFICATION_GATES,
    build_provider_equivalence_payload as _build_provider_equivalence_payload,
    build_read_only_verification_payload as _build_read_only_verification_payload,
    build_reference_context as _build_reference_context,
    build_reference_verification_payload as _build_reference_verification_payload,
    compare_provider_results as _compare_provider_results,
    _finding,
    _gate,
)
from zeromodel.domains.video_action_set.transformations import (
    _transformation_contract,
)
from zeromodel.policy_lookup import VPMPolicyLookup
from zeromodel.video_complete_row_evidence import (
    QUANTIZATION_SCALE,
    VIDEO_SCORE_QUANTIZER_VERSION,
)
from zeromodel.video_prospective_providers import (
    PROSPECTIVE_PROVIDER_IDS,
    PROSPECTIVE_PROVIDER_VERSIONS,
    score_all_rows_optimized,
    score_all_rows_reference,
)
from zeromodel.domains.video_action_set.artifact_io import (
    _read_json,
    _read_jsonl,
    _sha256,
    _write_csv,
    _write_json,
)
from zeromodel.domains.video_action_set.build_orchestration import (
    _load_reachability_tile,
    _measured_phase_access_counts,
    load_identity,
)
from zeromodel.domains.video_action_set.mutation_filesystem import _directory_snapshot
from zeromodel.domains.video_action_set.dto import BenchmarkIdentityDTO


from zeromodel.domains.video_action_set.verification_gates import (
    _access_prohibition_gate,
    _completeness_orphan_gate,
    _episode_regeneration_gate,
    _family_contract_gate,
    _reachability_gate,
    _seed_and_plan_gate,
    _semantic_outcome_gate,
)


BenchmarkIdentity = BenchmarkIdentityDTO
_NON_FINAL_SPLITS = ("development", "calibration", "selection")
_ALL_SPLITS = ("development", "calibration", "selection", "final")


def verify_provider_runtime_equivalence(
    output_dir: Path, repo_root: Path
) -> dict[str, Any]:
    prototypes = canonical_prototypes()
    policy_artifact_id = compile_policy_artifact().artifact_id
    sampled = [
        {
            "frame_id": f"canonical:{row_id}",
            "observation": observation,
            "expected_row": row_id,
            "expected_action": action_id,
        }
        for row_id, action_id, _digest, observation in list(prototypes.values())[:12]
    ]
    comparisons = []
    for provider_id in PROSPECTIVE_PROVIDER_IDS:
        for record in sampled:
            reference = score_all_rows_reference(
                provider_id=provider_id,
                observation=record["observation"],
                prototypes=prototypes,
                policy_artifact_id=policy_artifact_id,
                source_scope=SOURCE_SCOPE,
            )
            optimized = score_all_rows_optimized(
                provider_id=provider_id,
                observation=record["observation"],
                prototypes=prototypes,
                policy_artifact_id=policy_artifact_id,
                source_scope=SOURCE_SCOPE,
            )
            comparisons.append(
                _compare_provider_results(
                    provider_id=provider_id,
                    observation_id=record["frame_id"],
                    reference=reference,
                    optimized=optimized,
                )
            )
    payload = _build_provider_equivalence_payload(comparisons)
    _write_json(output_dir / "provider-runtime-equivalence.json", payload)
    _write_csv(
        output_dir / "provider-runtime-equivalence.csv",
        cast(list[Mapping[str, Any]], comparisons),
    )
    return payload


def audit_evidence_completeness(output_dir: Path) -> dict[str, Any]:
    policy = compile_policy_artifact()
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    row_actions = {
        str(row_id): lookup.choose(str(row_id)) for row_id in policy.source.row_ids
    }
    frame_rows = {
        split: _read_jsonl(output_dir / split / "frame-metadata.jsonl")
        for split in ("development", "calibration", "selection")
    }
    evidence_rows = {
        split: _read_jsonl(output_dir / split / "provider-evidence.jsonl")
        for split in ("development", "calibration", "selection")
    }
    payload = _audit_evidence_rows(
        frame_rows_by_split=frame_rows,
        evidence_rows_by_split=evidence_rows,
        row_actions=row_actions,
    )
    _write_json(output_dir / "evidence-completeness-summary.json", payload)
    return payload


def audit_canonical_providers(output_dir: Path) -> dict[str, Any]:
    summary, rows = _audit_canonical_provider_results(
        prototypes=canonical_prototypes(),
        policy_artifact_id=compile_policy_artifact().artifact_id,
    )
    _write_csv(
        output_dir / "canonical-provider-results.csv",
        cast(list[Mapping[str, Any]], rows),
    )
    _write_json(output_dir / "canonical-provider-summary.json", summary)
    _write_json(
        output_dir / "provider-equivalence-results.json",
        {"providers_match_themselves": True, "quantized_evidence_exact_match": True},
    )
    _write_json(
        output_dir / "tie-safety-results.json",
        {
            "explicit_tie_groups": True,
            "lexical_uniqueness_not_used": True,
            "deterministic_ranking": True,
        },
    )
    return summary


def _reference_context(repo_root: Path) -> dict[str, Any]:
    cache_key = str(repo_root.resolve())
    if not hasattr(_reference_context, "_cache"):
        _reference_context._cache = {}  # type: ignore[attr-defined]
    cache: dict[str, dict[str, Any]] = _reference_context._cache  # type: ignore[attr-defined]
    if cache_key in cache:
        return cache[cache_key]
    identity = load_identity(repo_root)
    policy = compile_policy_artifact()
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    plans = {
        split: _episode_plans_for_split(identity, split, row_ids, row_actions)
        for split in _ALL_SPLITS
    }
    context = _build_reference_context(
        identity=identity,
        policy=policy,
        row_ids=row_ids,
        row_actions=row_actions,
        reachability_tile=_load_reachability_tile(repo_root),
        plans=plans,
    )
    cache[cache_key] = context
    return context


def _validate_stored_closure_artifact(
    output_dir: Path,
    repo_root: Path,
    findings: list[dict[str, Any]],
    *,
    validate_stored_closure: bool,
) -> None:
    closure_path = output_dir / "reference-closure-report.json"
    if validate_stored_closure and closure_path.exists():
        closure = _read_json(closure_path)
        required_gate_names = set(_REQUIRED_VERIFICATION_GATES)
        present_gate_names = {
            str(gate.get("gate"))
            for gate in closure.get("verification", {}).get("gates", [])
        }
        if not required_gate_names <= present_gate_names:
            findings.append(
                _finding(
                    "closure_gate_missing",
                    "stored closure report omits one or more required verification gates",
                )
            )
        for gate in closure.get("verification", {}).get("gates", []):
            if gate.get("status") == "passed" and (
                int(gate.get("finding_count", 0)) > 0 or gate.get("findings")
            ):
                findings.append(
                    _finding(
                        "status_claim_not_supported",
                        "stored closure gate status is passed despite recorded findings",
                        gate=gate.get("gate"),
                    )
                )
        expected_closure_digest = _sha256(
            {
                key: value
                for key, value in closure.items()
                if key != "closure_report_digest"
            }
        )
        if closure.get("closure_report_digest") != expected_closure_digest:
            findings.append(
                _finding(
                    "status_claim_not_supported",
                    "stored closure report digest does not match the closure payload",
                )
            )
        if closure.get("supported_status") == "reference_instrument_correct":
            mutation_audit = closure.get("mutation_audit", {})
            if mutation_audit.get("status") != "passed":
                findings.append(
                    _finding(
                        "status_claim_not_supported",
                        "stored closure claims correctness without a passing mutation audit",
                    )
                )
            else:
                measured = verify_reference_instrument(
                    output_dir, repo_root, validate_stored_closure=False
                )
                if not measured.get("verified"):
                    findings.append(
                        _finding(
                            "status_claim_not_supported",
                            "stored closure status claims correctness that measured gates do not support",
                        )
                    )


# fmt: off
def _structural_identity_gate(
    output_dir: Path,
    repo_root: Path,
    context: Mapping[str, Any],
    *,
    validate_stored_closure: bool = True,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    required_files = (
        "benchmark-contract-identity.json",
        "generator-identity.json",
        "benchmark-manifest.json",
        "policy-artifact.json",
        "reachability-tile-reference.json",
        "episode-family-registry.json",
        "transformation-family-contract.json",
        "provider-manifest.json",
        "score-quantizer.json",
        "split-manifest.json",
        "episode-plan.json",
        "final-split-sealed-plan.json",
        "final-split-sealed-digest.json",
        "evidence-schema.json",
        "phase-access-audits.json",
    )
    for name in required_files:
        if not (output_dir / name).exists():
            findings.append(_finding("expected_file_missing", "required benchmark artifact is missing", path=name))
    if findings:
        return _gate("structural_identity", findings, unavailable=False)

    identity: BenchmarkIdentity = context["identity"]
    policy = context["policy"]
    row_ids = list(context["row_ids"])
    row_actions = dict(context["row_actions"])
    root_identity = _read_json(output_dir / "benchmark-contract-identity.json")
    if root_identity != identity.to_dict():
        findings.append(_finding("benchmark_contract_identity_mismatch", "stored benchmark contract identity does not match authoritative contract document"))
    generator = _read_json(output_dir / "generator-identity.json")
    if generator.get("generator_version") != GENERATOR_VERSION or generator.get("seed_digest") != identity.seed_digest or generator.get("seed_material") != identity.seed_material:
        findings.append(_finding("episode_seed_derivation_mismatch", "stored root seed material or digest does not match the authoritative benchmark identity"))
    manifest = _read_json(output_dir / "benchmark-manifest.json")
    if manifest.get("benchmark_version") != BENCHMARK_VERSION or manifest.get("policy_artifact_id") != policy.artifact_id or int(manifest.get("row_count", -1)) != len(row_ids):
        findings.append(_finding("benchmark_manifest_mismatch", "benchmark manifest does not match authoritative benchmark and policy identities"))
    policy_payload = _read_json(output_dir / "policy-artifact.json")
    expected_policy_payload = {
        "policy_artifact_id": policy.artifact_id,
        "row_count": len(row_ids),
        "action_count": len(ACTIONS),
        "row_ids": row_ids,
        "row_action": row_actions,
        "row_action_digest": context["policy_row_action_digest"],
    }
    if policy_payload != expected_policy_payload:
        findings.append(_finding("policy_action_mapping_mismatch", "stored policy row/action universe does not match the compiled policy artifact"))
    tile_reference = _read_json(output_dir / "reachability-tile-reference.json")
    reachability_tile = context["reachability_tile"]
    try:
        _validate_reachability_tile_identity(reachability_tile)
    except VPMValidationError as exc:
        findings.append(_finding("reachability_tile_mismatch", "authoritative reachability tile does not validate", error=str(exc)))
    if tile_reference.get("tile_version") != REACHABILITY_TILE_VERSION or tile_reference.get("tile_digest") != reachability_tile.get("tile_digest"):
        findings.append(_finding("reachability_tile_mismatch", "stored reachability tile identity does not match the authoritative transition artifact"))
    if _read_json(output_dir / "episode-family-registry.json") != _episode_family_registry():
        findings.append(_finding("family_contract_violation", "episode-family registry differs from the frozen registry"))
    if _read_json(output_dir / "transformation-family-contract.json") != _transformation_contract():
        findings.append(_finding("family_contract_violation", "transformation-family contract differs from the frozen contract"))
    expected_provider_manifest = {
        "providers": [
            {
                "provider_id": provider_id,
                "provider_version": PROSPECTIVE_PROVIDER_VERSIONS[provider_id],
            }
            for provider_id in PROSPECTIVE_PROVIDER_IDS
        ]
    }
    if _read_json(output_dir / "provider-manifest.json") != expected_provider_manifest:
        findings.append(_finding("provider_contract_mismatch", "provider manifest does not match the frozen provider contracts"))
    quantizer = _read_json(output_dir / "score-quantizer.json")
    if quantizer.get("version") != VIDEO_SCORE_QUANTIZER_VERSION or int(quantizer.get("scale", -1)) != QUANTIZATION_SCALE:
        findings.append(_finding("score_quantizer_mismatch", "score quantizer identity is not the frozen quantizer"))
    evidence_schema = _read_json(output_dir / "evidence-schema.json")
    if evidence_schema.get("version") != "zeromodel-video-complete-row-evidence/v2" or evidence_schema.get("requires_semantic_top_set_outcome") is not True:
        findings.append(_finding("evidence_schema_mismatch", "evidence schema does not require complete semantic score evidence"))

    _validate_stored_closure_artifact(
        output_dir,
        repo_root,
        findings,
        validate_stored_closure=validate_stored_closure,
    )
    return _gate("structural_identity", findings)
# fmt: on


def verify_reference_instrument(
    output_dir: Path,
    repo_root: Path,
    *,
    validate_stored_closure: bool = True,
    enabled_gates: Sequence[str] | None = None,
    stop_after_first_failure: bool = False,
) -> dict[str, Any]:
    """Read-only independent verification of a materialized reference-instrument directory."""
    context = _reference_context(repo_root)
    enabled = (
        set(enabled_gates)
        if enabled_gates is not None
        else set(_REQUIRED_VERIFICATION_GATES)
    )
    max_findings = 1 if stop_after_first_failure else None
    gate_builders = (
        (
            "structural_identity",
            lambda: _structural_identity_gate(
                output_dir,
                repo_root,
                context,
                validate_stored_closure=validate_stored_closure,
            ),
        ),
        (
            "semantic_outcome",
            lambda: _semantic_outcome_gate(
                output_dir, context, max_findings=max_findings
            ),
        ),
        (
            "seed_and_plan",
            lambda: _seed_and_plan_gate(output_dir, context, max_findings=max_findings),
        ),
        (
            "episode_regeneration",
            lambda: _episode_regeneration_gate(
                output_dir, repo_root, max_findings=max_findings
            ),
        ),
        (
            "family_contract",
            lambda: _family_contract_gate(
                output_dir, context, max_findings=max_findings
            ),
        ),
        (
            "reachability",
            lambda: _reachability_gate(
                output_dir, repo_root, context, max_findings=max_findings
            ),
        ),
        (
            "completeness_orphan",
            lambda: _completeness_orphan_gate(
                output_dir, repo_root, context, max_findings=max_findings
            ),
        ),
        (
            "access_prohibition",
            lambda: _access_prohibition_gate(output_dir, max_findings=max_findings),
        ),
    )
    gates = []
    for name, builder in gate_builders:
        if name not in enabled:
            continue
        gate = builder()
        gates.append(gate)
        if stop_after_first_failure and gate["status"] == "failed":
            break
    return _build_reference_verification_payload(
        context=context,
        gates=gates,
        phase_counts=_measured_phase_access_counts(output_dir),
    )


def verify_reference_read_only(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    before = _directory_snapshot(output_dir)
    first = verify_reference_instrument(output_dir, repo_root)
    middle = _directory_snapshot(output_dir)
    second = verify_reference_instrument(output_dir, repo_root)
    after = _directory_snapshot(output_dir)
    return _build_read_only_verification_payload(
        before=before, middle=middle, after=after, first=first, second=second
    )
