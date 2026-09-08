"""Bounded Return experiment: recovery after memory loses authority.

Follows the frozen spec ``docs/research/return-bounded-recovery.md`` —
no implementation beyond it. Reuses the future-memory harness
(generation, perturbations, native stacks) and extends it with:

- a coarse observation profile (half-pooled L) with its own field
  schema, P3 memory, empirical transition memory, declarations, and
  validity state, all train-side fitted and frozen;
- a compiled 16-row Return VPM policy consulted through a bounded
  driver (at most one REOBSERVE and one FALLBACK; budget enforced
  independently of table scores);
- arms A (gate-only control), B (reobserve), C (fallback),
  D (combined), and E (fallback-directly ablation);
- Return metrics (invocation, operations, support-restored, fallback
  acceptance/correctness, trajectory decomposition, success by
  trigger), per-episode halves, and train-frozen regional mismatch
  telemetry (reported only, never gated).

Run (from ``examples/``)::

    PYTHONPATH=../packages/core/src ../venv/Scripts/python -m \\
        visual_transition_benchmark.return_benchmark --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _part in ("packages/core/src", "examples"):
    _path = str(_REPO_ROOT / _part)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from visual_transition_benchmark import future_memory_benchmark as fmb  # noqa: E402
from visual_transition_benchmark.domains.warehouse import model as wh_model  # noqa: E402
from zeromodel.perception import (  # noqa: E402
    CoupledLoopInputsDTO,
    DeclarationScopeDTO,
    DiscreteActionSchemaDTO,
    MemoryAuthorityContextDTO,
    TransitionActionDeclarationDTO,
    TransitionExpectationSetDTO,
    VisualTransitionAnalysisDTO,
    WorldActionPolicyDTO,
    build_dataset_manifest,
    build_grid_field_schema,
    build_transition_evidence_vpm,
    compile_return_policy,
    decide_with_return,
    encode_discrete_action,
    encode_source_array,
    evaluate_transition_conformance,
    fit_action_conditioned_transition_model,
    fit_baseline_nearest_neighbor,
    predict_action_with_future_memory,
    predict_baseline_action,
    project_expected_transition,
    record_verification_event,
    verify_expected_transition,
)
from zeromodel.perception.dataset import RecordedInteractionDTO  # noqa: E402
from zeromodel.perception.expectations import (  # noqa: E402
    PerceptionRegionAnnotationDTO,
)
from zeromodel.perception.fields import VPMFieldSchemaDTO  # noqa: E402
from zeromodel.perception.representation import SourceVPMDTO  # noqa: E402
from zeromodel.perception.transition_conformance import (  # noqa: E402
    TransitionExpectationDTO,
)
from zeromodel.video.arcade_policy.model import (  # noqa: E402
    ShooterConfig,
)

# Gate policy for THIS experiment: insufficient support abstains, so the
# INSUFFICIENT trigger is reachable. Coupling is bounded to the rank-0
# candidate so every veto is decidable (a vetoed rank-0 always abstains
# instead of silently reranking); System A is re-baselined under the
# identical policy (frozen spec §6).
#
# Claim boundary: with the native candidate_count=3, vetoed candidates
# are routed around internally and Return rarely fires — Return is not
# load-bearing in the normal multi-candidate path. candidate_count=1 is
# an experimental terminalization that isolates Return; it must not
# quietly become the default production configuration.
STRICT_POLICY = WorldActionPolicyDTO(
    candidate_count=1,
    min_support=10,
    low_confidence_threshold=0.35,
    reject_on_contradiction=True,
    reject_on_ood=True,
    reject_on_insufficient=True,
)


@dataclass(frozen=True)
class ReturnArm:
    name: str
    use_return: bool
    allow_reobserve: bool
    allow_fallback: bool


ARMS = (
    ReturnArm("A gate-only", False, False, False),
    ReturnArm("B reobserve", True, True, False),
    ReturnArm("C fallback", True, False, True),
    ReturnArm("D combined", True, True, True),
    ReturnArm("E fallback-direct", False, False, False),
)


@dataclass(frozen=True)
class CoarseStack:
    profile_id: str
    field_schema: VPMFieldSchemaDTO
    predictor: object
    empirical: object
    declarations: DeclarationScopeDTO
    loop: CoupledLoopInputsDTO


def downsample_frame(frame: np.ndarray) -> np.ndarray:
    """Half-pool a captured grayscale frame (privilege-free transform)."""
    array = np.asarray(frame, dtype=np.uint8)
    assert array.ndim == 2
    height, width = array.shape
    trimmed = array[: height - height % 2, : width - width % 2]
    pooled = trimmed.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))
    return np.round(pooled).astype(np.uint8)


def _encode_coarse(frame: np.ndarray) -> SourceVPMDTO:
    return encode_source_array(downsample_frame(frame), fmb.SPEC)


def _coarse_band_for_field(y0: int, x0: int, width_px: int) -> str:
    # Native bands remapped to coarse coordinates (each coarse row covers
    # two native rows; each coarse column covers two native columns).
    if 2 * y0 + 1 >= 11:
        return "tank"
    if 2 * y0 + 1 >= 2 and 2 * y0 <= 4:
        return "alien"
    if 2 * y0 + 1 >= 7 and 2 * y0 <= 8 and 2 * x0 >= width_px - 4:
        return "cooldown"
    return "background"


def build_coarse_arcade_setup(
    native: fmb.DomainSetup, probe_frame: np.ndarray
) -> fmb.DomainSetup:
    schema = build_grid_field_schema(
        _encode_coarse(probe_frame), tile_width=2, tile_height=1, channel_mode="joint"
    )
    height_px = probe_frame.shape[1]
    by_band: dict[str, list[str]] = {
        name: [] for name in ("tank", "alien", "cooldown", "background")
    }
    for field in schema.fields:
        by_band[_coarse_band_for_field(field.y0, field.x0, height_px)].append(
            field.field_id
        )
    total = sum(len(ids) for ids in by_band.values())
    assert total == len(schema.fields), "coarse bands must partition fields"
    by_name = {
        name: PerceptionRegionAnnotationDTO.create(
            schema, tuple(sorted(ids)), label=name
        )
        for name, ids in sorted(by_band.items())
        if ids
    }
    annotations = tuple(by_name[name] for name in sorted(by_name))
    native_by_id = {}
    for annotation in native.annotations:
        native_by_id[annotation.annotation_id] = annotation.label
    expectations: dict[str, tuple[TransitionExpectationDTO, ...]] = {}
    for action, native_expectations in native.expectations_by_action.items():
        remade = []
        for native_expectation in native_expectations:
            names = tuple(
                sorted(
                    native_by_id[annotation_id]
                    for annotation_id in native_expectation.annotation_ids
                )
            )
            remade.append(
                TransitionExpectationDTO.create(
                    field_schema_id=schema.field_schema_id,
                    annotation_ids=tuple(
                        sorted(by_name[name].annotation_id for name in names)
                    ),
                    expected_change=native_expectation.expected_change,
                    minimum_mean_absolute_change=(
                        native_expectation.minimum_mean_absolute_change
                    ),
                    maximum_mean_absolute_change=(
                        native_expectation.maximum_mean_absolute_change
                    ),
                    minimum_changed_fraction=(
                        native_expectation.minimum_changed_fraction
                    ),
                    maximum_changed_fraction=(
                        native_expectation.maximum_changed_fraction
                    ),
                    minimum_signed_change_magnitude=(
                        native_expectation.minimum_signed_change_magnitude
                    ),
                )
            )
        expectations[action] = tuple(remade)
    return fmb.DomainSetup(
        name=native.name + "-coarse",
        action_labels=native.action_labels,
        field_schema=schema,
        annotations=annotations,
        expectations_by_action=expectations,
    )


def build_coarse_warehouse_setup(
    native: fmb.DomainSetup, probe_frame: np.ndarray
) -> fmb.DomainSetup:
    schema = build_grid_field_schema(
        _encode_coarse(probe_frame), tile_width=5, tile_height=1, channel_mode="joint"
    )
    canvas = PerceptionRegionAnnotationDTO.create(
        schema,
        tuple(field.field_id for field in schema.fields),
        label="canvas",
    )
    change = TransitionExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        annotation_ids=(canvas.annotation_id,),
        expected_change="change",
        minimum_mean_absolute_change=0.008,
        minimum_changed_fraction=0.005,
    )
    stable = TransitionExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        annotation_ids=(canvas.annotation_id,),
        expected_change="stable",
        maximum_mean_absolute_change=0.0,
        maximum_changed_fraction=0.0,
    )
    expectations = {
        action: (change,) for action in wh_model.ACTIONS if action != "WAIT"
    }
    expectations["WAIT"] = (stable,)
    return fmb.DomainSetup(
        name=native.name + "-coarse",
        action_labels=native.action_labels,
        field_schema=schema,
        annotations=(canvas,),
        expectations_by_action=expectations,
    )


def build_coarse_manifest(
    transitions: list,
    action_schema: DiscreteActionSchemaDTO,
    *,
    split_seed: str,
) -> tuple[object, dict[str, SourceVPMDTO]]:
    interactions: list[RecordedInteractionDTO] = []
    sources: dict[str, SourceVPMDTO] = {}
    for record in transitions:
        before = _encode_coarse(record.frame_before)
        after = _encode_coarse(record.frame_after)
        sources[before.source_vpm_id] = before
        sources[after.source_vpm_id] = after
        interactions.append(
            RecordedInteractionDTO.from_vpms(
                sequence_id=record.episode_id,
                step_index=record.step_number,
                source=before,
                target=encode_discrete_action(record.action, action_schema),
                next_source=after,
            )
        )
    manifest = build_dataset_manifest(
        interactions,
        source_encoder_spec_ids=[fmb.SPEC.encoder_spec_id],
        split_seed=split_seed,
        reject_errors=False,
    )
    allowed = {"identical_source_across_splits"}
    unexpected = {
        finding.code for finding in manifest.findings if finding.severity == "error"
    } - allowed
    if unexpected:
        raise ValueError(f"unexpected dataset findings: {sorted(unexpected)}")
    return manifest, sources


def fit_coarse_stack(
    transitions: list,
    action_schema: DiscreteActionSchemaDTO,
    setup: fmb.DomainSetup,
    *,
    split_seed: str,
) -> CoarseStack:
    manifest, sources = build_coarse_manifest(
        transitions, action_schema, split_seed=split_seed
    )
    predictor = fit_baseline_nearest_neighbor(manifest, sources, training_split="all")
    empirical = fit_action_conditioned_transition_model(
        manifest,
        sources,
        setup.field_schema,
        config=fmb.MODEL_CONFIG,
        training_split="all",
        change_threshold=fmb.CHANGE_THRESHOLD,
        annotations=tuple(setup.annotations),
    )
    declarations = DeclarationScopeDTO.create(
        dict(setup.expectations_by_action), tuple(setup.annotations)
    )
    loop = CoupledLoopInputsDTO(
        profile_id="coarse-L",
        predictor_model=predictor,
        transition_model=empirical,
        field_schema=setup.field_schema,
        declarations=declarations,
        policy=STRICT_POLICY,
    )
    return CoarseStack(
        profile_id="coarse-L",
        field_schema=setup.field_schema,
        predictor=predictor,
        empirical=empirical,
        declarations=declarations,
        loop=loop,
    )


def _reobserve_fn(source: SourceVPMDTO) -> SourceVPMDTO:
    return _encode_coarse(np.asarray(source.to_array(), dtype=np.uint8))


def _mass_quadrant(frame: np.ndarray) -> str:
    """Coarse observation cluster: quadrant holding the most nonzero mass.

    Predeclared 2x2 geometry evaluated on the before-frame only;
    deterministic, no fitting, frozen before evaluation.
    """
    array = np.asarray(frame)
    height, width = array.shape[0], array.shape[1]
    mid_row, mid_col = height // 2, width // 2
    masses = {
        "top-left": array[:mid_row, :mid_col],
        "top-right": array[:mid_row, mid_col:],
        "bottom-left": array[mid_row:, :mid_col],
        "bottom-right": array[mid_row:, mid_col:],
    }
    counts = {name: int(np.count_nonzero(region)) for name, region in masses.items()}
    return max(sorted(counts), key=lambda name: counts[name])


def _verify_enacted(
    transition_model,
    before: SourceVPMDTO,
    after: SourceVPMDTO,
    record,
    setup,
    *,
    min_support: int,
) -> tuple:
    projected = project_expected_transition(
        transition_model,
        before,
        record.action,
        setup.field_schema,
        min_support=min_support,
    )
    observed = build_transition_evidence_vpm(
        before,
        after,
        setup.field_schema,
        annotations=tuple(setup.annotations),
        change_threshold=fmb.CHANGE_THRESHOLD,
    )
    action_declaration = TransitionActionDeclarationDTO.create(
        action_type="discrete_action",
        payload={"action_label": record.action},
    )
    enacted_expectations = setup.expectations_by_action.get(record.action, ())
    if enacted_expectations:
        report = evaluate_transition_conformance(
            observed, enacted_expectations, tuple(setup.annotations)
        )
        expectation_set = TransitionExpectationSetDTO.create(enacted_expectations)
    else:
        fallback_expectations = next(iter(setup.expectations_by_action.values()))
        report = evaluate_transition_conformance(
            observed, fallback_expectations, tuple(setup.annotations)
        )
        expectation_set = TransitionExpectationSetDTO.create(fallback_expectations)
    analysis = VisualTransitionAnalysisDTO.create(
        transition=observed,
        action=action_declaration,
        expectation_set=expectation_set,
        conformance_report=report,
    )
    verdict = verify_expected_transition(
        projected,
        analysis,
        tolerance=fmb.VERIFICATION_TOLERANCE,
        change_epsilon=fmb.VERIFICATION_EPSILON,
    )
    matched = fmb._projection_matches(
        projected,
        observed,
        tolerance=fmb.VERIFICATION_TOLERANCE,
        change_epsilon=fmb.VERIFICATION_EPSILON,
    )
    mae = None
    if projected.status == "supported":
        observed_by_id = {item.field_id: item for item in observed.fields}
        mae = float(
            np.mean(
                [
                    abs(
                        field.expected_after_mean
                        - observed_by_id[field.field_id].after_mean
                    )
                    for field in projected.fields
                ]
            )
        )
    return verdict, matched, mae


def _score_return_query(
    arm: ReturnArm,
    record,
    setup: fmb.DomainSetup,
    native_loop,
    coarse: Optional[tuple],
    validity: dict,
    coarse_validity: dict,
    return_lookup,
    return_artifact_id: str,
) -> dict:
    before = fmb._encode_frame(record.frame_before)
    after = fmb._encode_frame(record.frame_after)
    baseline = predict_baseline_action(native_loop.predictor_model, before)
    base_ok = baseline.candidates[0].action_label == record.label
    row: dict = {
        "episode_id": record.episode_id,
        "step_number": record.step_number,
        "quadrant": None,
        "base_ok": base_ok,
        "initial_ok": None,
        "initial_accepted": None,
        "final_ok": None,
        "accepted": None,
        "decided_by": None,
        "disposition": None,
        "trigger": None,
        "invoked": False,
        "reobserved": False,
        "fallback_used": False,
        "support_restored": None,
        "fallback_accepted": None,
        "fallback_correct": None,
        "verdict": None,
        "matched": None,
        "mae": None,
    }
    if arm.name == "E fallback-direct":
        final_ok = base_ok
        row.update(
            final_ok=final_ok,
            initial_ok=base_ok,
            accepted=True,
            decided_by="BASELINE",
            disposition=None,
        )
        return row
    out = predict_action_with_future_memory(
        native_loop.predictor_model,
        native_loop.transition_model,
        before,
        setup.field_schema,
        declarations=native_loop.declarations,
        policy=STRICT_POLICY,
        authority=MemoryAuthorityContextDTO.create(validity),
    )
    initial_ok = (out.selected_action == record.label) if out.accepted else False
    row["initial_ok"] = initial_ok
    row["initial_accepted"] = out.accepted
    if not arm.use_return:
        row.update(
            final_ok=initial_ok,
            accepted=out.accepted,
            decided_by="PRIMARY",
            disposition=None,
        )
        return row
    row["invoked"] = not out.accepted
    coarse_stack, coarse_setup = coarse if coarse is not None else (None, None)
    trajectory = decide_with_return(
        native_loop,
        before,
        authority=MemoryAuthorityContextDTO.create(validity),
        return_lookup=return_lookup,
        return_artifact_id=return_artifact_id,
        fallback_predictor=native_loop.predictor_model if arm.allow_fallback else None,
        coarse=coarse_stack.loop
        if (arm.allow_reobserve and coarse_stack is not None)
        else None,
        coarse_authority=MemoryAuthorityContextDTO.create(coarse_validity)
        if (arm.allow_reobserve and coarse_stack is not None)
        else None,
        reobserve_source=_reobserve_fn if arm.allow_reobserve else None,
    )
    final_ok = (
        (trajectory.final_action == record.label) if trajectory.accepted else False
    )
    row.update(
        final_ok=final_ok,
        accepted=trajectory.accepted,
        decided_by=trajectory.decided_by,
        disposition=trajectory.disposition,
    )
    if trajectory.decisions:
        first = trajectory.decisions[0]
        row["trigger"] = first.trigger
        row["reobserved"] = any(
            item.operation == "REOBSERVE" for item in trajectory.decisions
        )
        row["fallback_used"] = any(
            item.operation == "FALLBACK" for item in trajectory.decisions
        )
        if row["reobserved"] and coarse_stack is not None:
            coarse_source = _reobserve_fn(before)
            coarse_projected = project_expected_transition(
                coarse_stack.empirical,
                coarse_source,
                record.action,
                coarse_stack.field_schema,
                min_support=STRICT_POLICY.min_support,
            )
            row["support_restored"] = coarse_projected.status == "supported"
        if row["fallback_used"]:
            fallback_prediction = predict_baseline_action(
                native_loop.predictor_model, before
            )
            row["fallback_accepted"] = fallback_prediction.status == "accepted"
            row["fallback_correct"] = (
                fallback_prediction.selected_action == record.label
            )
    verdict, matched, mae = _verify_enacted(
        native_loop.transition_model,
        before,
        after,
        record,
        setup,
        min_support=STRICT_POLICY.min_support,
    )
    row.update(verdict=verdict.status, matched=matched, mae=mae)
    validity[record.action] = record_verification_event(
        validity.get(record.action),
        verdict,
        transition_model_id=native_loop.transition_model.model_id,
        action_label=record.action,
        field_schema_id=setup.field_schema.field_schema_id,
        training_dataset_id=native_loop.transition_model.dataset_id,
    )
    if coarse is not None and arm.allow_reobserve:
        coarse_stack, coarse_setup = coarse
        coarse_source = _reobserve_fn(before)
        coarse_after = _encode_coarse(record.frame_after)
        coarse_verdict, _, _ = _verify_enacted(
            coarse_stack.empirical,
            coarse_source,
            coarse_after,
            record,
            coarse_setup,
            min_support=STRICT_POLICY.min_support,
        )
        coarse_validity[record.action] = record_verification_event(
            coarse_validity.get(record.action),
            coarse_verdict,
            transition_model_id=coarse_stack.empirical.model_id,
            action_label=record.action,
            field_schema_id=coarse_stack.field_schema.field_schema_id,
            training_dataset_id=coarse_stack.empirical.dataset_id,
        )
    height, width = np.asarray(record.frame_before).shape[:2]
    row["quadrant"] = f"{_mass_quadrant(record.frame_before)}|{record.action}"
    return row


def _return_harmed(row: dict) -> bool:
    # Harm is scoped to recovery commits: the primary/baseline paths can
    # never accrue Return harm. A recovery commit is harm when it overturns
    # a correct initial OR converts a safe abstention into a wrong action
    # (the frozen definition's principal harm mode).
    if row["decided_by"] not in ("REOBSERVE", "FALLBACK"):
        return False
    if not row["accepted"] or row["final_ok"]:
        return False
    return bool(row["initial_ok"] or not row.get("initial_accepted"))


def _aggregate_return_rows(rows: list[dict]) -> dict:
    total = len(rows)
    base_correct = sum(1 for row in rows if row["base_ok"])
    initial_correct = sum(1 for row in rows if row["initial_ok"])
    final_correct = sum(1 for row in rows if row["final_ok"])
    answered = sum(1 for row in rows if row["accepted"])
    initial_answered = sum(1 for row in rows if row.get("initial_accepted"))
    gain = sum(
        1 for row in rows if row["initial_ok"] is False and row["final_ok"] is True
    )
    # Frozen definition: harm counts correct→wrong AND abstain→wrong-commit,
    # the latter being Return's principal harm mode (safe abstention converted
    # into an incorrect commitment). Scoped to recovery commits so the
    # primary/baseline paths can never accrue Return harm.
    harm = sum(1 for row in rows if _return_harmed(row))
    invoked = sum(1 for row in rows if row["invoked"])
    reobserved = sum(1 for row in rows if row["reobserved"])
    fallback_used = sum(1 for row in rows if row["fallback_used"])
    restored = [row["support_restored"] for row in rows if row["support_restored"]]
    fallback_accepted = [
        row["fallback_accepted"] for row in rows if row["fallback_accepted"] is not None
    ]
    fallback_correct = [
        row["fallback_correct"] for row in rows if row["fallback_correct"] is not None
    ]
    abstained = sum(1 for row in rows if row["disposition"] == "ABSTAIN")
    escalated = sum(1 for row in rows if row["disposition"] == "ESCALATE")
    decided_by: dict[str, int] = {}
    triggers: dict[str, int] = {}
    for row in rows:
        if row["decided_by"] is not None:
            decided_by[row["decided_by"]] = decided_by.get(row["decided_by"], 0) + 1
        if row["trigger"] is not None:
            triggers[row["trigger"]] = triggers.get(row["trigger"], 0) + 1
    verification: dict[str, int] = {}
    matched = matched_total = 0
    maes = []
    for row in rows:
        if row["verdict"] is not None:
            verification[row["verdict"]] = verification.get(row["verdict"], 0) + 1
            matched_total += 1
            matched += 1 if row["matched"] else 0
        if row["mae"] is not None:
            maes.append(row["mae"])
    denominator = gain + harm
    return {
        "n": total,
        "base_acc": base_correct / total if total else float("nan"),
        "initial_acc": initial_correct / total if total else float("nan"),
        "final_acc": final_correct / total if total else float("nan"),
        "initial_coverage": initial_answered / total if total else float("nan"),
        "coverage": answered / total if total else float("nan"),
        "return_gain": gain / total if total else float("nan"),
        "return_harm": harm / total if total else float("nan"),
        "return_net": (gain - harm) / total if total else float("nan"),
        "invocation_rate": invoked / total if total else float("nan"),
        "reobserve_rate": reobserved / total if total else float("nan"),
        "fallback_rate": fallback_used / total if total else float("nan"),
        "support_restored_rate": sum(restored) / len(restored)
        if restored
        else float("nan"),
        "fallback_acceptance_rate": sum(fallback_accepted) / len(fallback_accepted)
        if fallback_accepted
        else float("nan"),
        "fallback_correct_rate": sum(fallback_correct) / len(fallback_correct)
        if fallback_correct
        else float("nan"),
        "abstention_rate": abstained / total if total else float("nan"),
        "escalation_rate": escalated / total if total else float("nan"),
        "decided_by": decided_by,
        "triggers": triggers,
        "verification": verification,
        "projection_match_rate": matched / matched_total
        if matched_total
        else float("nan"),
        "future_mae": float(np.mean(maes)) if maes else float("nan"),
        "gain_harm_ratio": gain / denominator if denominator else float("nan"),
    }


def _regional_telemetry(rows: list[dict]) -> dict:
    """Mismatch rate by predeclared coarse cluster (telemetry only)."""
    by_cluster: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["verdict"] is None or row["quadrant"] is None:
            continue
        cell = by_cluster.setdefault(row["quadrant"], {"n": 0, "bad": 0})
        cell["n"] += 1
        if row["verdict"] in {
            "future_projection_mismatch",
            "declared_expectation_violation",
        }:
            cell["bad"] += 1
    return {
        key: round(value["bad"] / value["n"], 3)
        for key, value in sorted(by_cluster.items())
    }


def evaluate_return_split(
    arm: ReturnArm,
    transitions: list,
    setup: fmb.DomainSetup,
    native_loop,
    coarse: Optional[tuple],
    return_lookup,
    return_artifact_id: str,
) -> tuple[dict, dict]:
    rows: list[dict] = []
    validity: dict = {}
    coarse_validity: dict = {}
    for record in transitions:
        rows.append(
            _score_return_query(
                arm,
                record,
                setup,
                native_loop,
                coarse,
                validity,
                coarse_validity,
                return_lookup,
                return_artifact_id,
            )
        )
    metrics = _aggregate_return_rows(rows)
    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(row["episode_id"], []).append(row)
    first_rows: list[dict] = []
    second_rows: list[dict] = []
    for episode_rows in by_episode.values():
        ordered = sorted(episode_rows, key=lambda row: row["step_number"])
        cut = len(ordered) // 2
        first_rows.extend(ordered[:cut])
        second_rows.extend(ordered[cut:])
    metrics["halves"] = {
        "first": _aggregate_return_rows(first_rows),
        "second": _aggregate_return_rows(second_rows),
    }
    return metrics, metrics["halves"], rows


def run_return_domain(
    setup: fmb.DomainSetup,
    coarse_setup: fmb.DomainSetup,
    train: list,
    splits: dict[str, list],
    *,
    split_seed: str,
    return_lookup,
    return_artifact_id: str,
) -> dict:
    action_schema = DiscreteActionSchemaDTO.from_labels(
        sorted({record.action for record in train})
    )
    manifest, sources = fmb.build_manifest(train, action_schema, split_seed=split_seed)
    memory = fmb.fit_all(manifest, sources, setup)
    declarations = DeclarationScopeDTO.create(
        dict(setup.expectations_by_action), tuple(setup.annotations)
    )
    native_loop = CoupledLoopInputsDTO(
        profile_id="native-L",
        predictor_model=memory.predictor,
        transition_model=memory.empirical,
        field_schema=setup.field_schema,
        declarations=declarations,
        policy=STRICT_POLICY,
    )
    coarse = fit_coarse_stack(
        train, action_schema, coarse_setup, split_seed=split_seed + "/coarse"
    )
    coarse_pair = (coarse, coarse_setup)
    result: dict = {"splits": {}}
    for split_name, transitions in splits.items():
        result["splits"][split_name] = {}
        regional_rows: list[dict] = []
        for arm in ARMS:
            metrics, _, rows = evaluate_return_split(
                arm,
                transitions,
                setup,
                native_loop,
                coarse_pair,
                return_lookup,
                return_artifact_id,
            )
            result["splits"][split_name][arm.name] = metrics
            if arm.name == "D combined":
                regional_rows = rows
        result["splits"][split_name]["regional"] = _regional_telemetry(regional_rows)
    return result


_RETURN_METRIC_KEYS = (
    "base_acc",
    "initial_acc",
    "final_acc",
    "initial_coverage",
    "coverage",
    "return_gain",
    "return_harm",
    "return_net",
    "invocation_rate",
    "reobserve_rate",
    "fallback_rate",
    "support_restored_rate",
    "fallback_acceptance_rate",
    "fallback_correct_rate",
    "abstention_rate",
    "escalation_rate",
    "projection_match_rate",
    "future_mae",
    "gain_harm_ratio",
)


def _average_return_cells(cells: list[dict]) -> dict:
    merged: dict = {"n": sum(cell["n"] for cell in cells)}
    for key in _RETURN_METRIC_KEYS:
        values = [
            cell[key]
            for cell in cells
            if not (isinstance(cell[key], float) and np.isnan(cell[key]))
        ]
        merged[key] = float(np.mean(values)) if values else float("nan")
    for key in ("verification", "decided_by", "triggers"):
        combined: dict = {}
        for cell in cells:
            for status, count in cell[key].items():
                combined[status] = combined.get(status, 0) + count
        merged[key] = combined
    return merged


def print_return_table(results: dict) -> None:
    for domain_name, domain in results["domains"].items():
        print(f"\n### {domain_name} (seeds {results['seeds']})")
        print(
            "| split | arm | base->initial->final | coverage | r_gain | "
            "r_harm | r_net | invoked | abstain | escalate |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        for split_name, arms in domain["splits"].items():
            for arm in ARMS:
                metrics = arms[arm.name]
                print(
                    f"| {split_name} | {arm.name} | "
                    f"{metrics['base_acc']:.3f}->{metrics['initial_acc']:.3f}"
                    f"->{metrics['final_acc']:.3f} | "
                    f"{metrics['coverage']:.3f} | {metrics['return_gain']:.3f} | "
                    f"{metrics['return_harm']:.3f} | {metrics['return_net']:+.3f} | "
                    f"{metrics['invocation_rate']:.3f} | "
                    f"{metrics['abstention_rate']:.3f} | "
                    f"{metrics['escalation_rate']:.3f} |"
                )
        print("  mechanism: support-restored / fallback-accept / fallback-correct")
        for split_name, arms in domain["splits"].items():
            for arm in ARMS:
                if not arm.use_return:
                    continue
                metrics = arms[arm.name]
                print(
                    f"  [{split_name}] {arm.name}: "
                    f"{metrics['support_restored_rate']:.3f} / "
                    f"{metrics['fallback_acceptance_rate']:.3f} / "
                    f"{metrics['fallback_correct_rate']:.3f} "
                    f"decided_by={metrics['decided_by']} "
                    f"triggers={metrics['triggers']}"
                )
        print("  halves (first -> second): net / harm / coverage")
        for split_name, arms in domain["splits"].items():
            for arm in ARMS:
                if not arm.use_return:
                    continue
                halves = arms[arm.name]["halves"]
                first, second = halves["first"], halves["second"]
                print(
                    f"  [{split_name}] {arm.name}: "
                    f"{first['return_net']:+.3f}->{second['return_net']:+.3f} / "
                    f"{first['return_harm']:.3f}->{second['return_harm']:.3f} / "
                    f"{first['coverage']:.3f}->{second['coverage']:.3f}"
                )
        print("  regional mismatch telemetry (cluster: rate):")
        for split_name, arms in domain["splits"].items():
            print(f"  [{split_name}]:", arms["regional"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--train-episodes", type=int, default=24)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "return_benchmark",
    )
    args = parser.parse_args(argv)
    seeds = [int(part) for part in args.seeds.split(",")]
    return_artifact, return_lookup = compile_return_policy()
    return_artifact_id: str = return_artifact.artifact_id
    results: dict = {
        "seeds": seeds,
        "return_policy_id": return_artifact_id,
        "domains": {},
    }
    per_seed: list[dict] = []
    for seed in seeds:
        arcade_setup = fmb._arcade_setup()
        arcade_train = fmb.generate_arcade_episodes(
            prefix="train",
            episode_count=args.train_episodes,
            seed_offset=seed * 10_000,
            config=ShooterConfig(),
        )
        arcade_clean = fmb.generate_arcade_episodes(
            prefix="clean",
            episode_count=args.eval_episodes,
            seed_offset=seed * 10_000 + 1_000_000,
            config=ShooterConfig(),
        )
        arcade_changed = fmb.generate_arcade_episodes(
            prefix="changed",
            episode_count=args.eval_episodes,
            seed_offset=seed * 10_000 + 2_000_000,
            config=ShooterConfig(wave=fmb.ARCADE_CHANGED_WAVE),
        )
        arcade_splits = {
            "clean": arcade_clean,
            "nuisance-background": fmb.perturb_background_shift(arcade_clean),
            "nuisance-noise": fmb.perturb_pixel_noise(arcade_clean, seed + 77),
            "transition-change": arcade_changed,
        }
        arcade_coarse = build_coarse_arcade_setup(
            arcade_setup, arcade_clean[0].frame_before
        )
        per_seed.append(
            (
                "arcade",
                run_return_domain(
                    arcade_setup,
                    arcade_coarse,
                    arcade_train,
                    arcade_splits,
                    split_seed=f"return-arcade/{seed}",
                    return_lookup=return_lookup,
                    return_artifact_id=return_artifact_id,
                ),
            )
        )
        warehouse_setup = fmb._warehouse_setup()
        warehouse_train = fmb.generate_warehouse_episodes(
            prefix="train",
            episode_count=args.train_episodes,
            seed_offset=seed * 10_000 + 5_000,
            goal=(3, 3),
        )
        warehouse_clean = fmb.generate_warehouse_episodes(
            prefix="clean",
            episode_count=args.eval_episodes,
            seed_offset=seed * 10_000 + 1_005_000,
            goal=(3, 3),
        )
        warehouse_changed = fmb.generate_warehouse_episodes(
            prefix="changed",
            episode_count=args.eval_episodes,
            seed_offset=seed * 10_000 + 2_005_000,
            goal=fmb.WAREHOUSE_CHANGED_GOAL,
        )
        warehouse_splits = {
            "clean": warehouse_clean,
            "nuisance-background": fmb.perturb_background_shift(warehouse_clean),
            "nuisance-noise": fmb.perturb_pixel_noise(warehouse_clean, seed + 913),
            "transition-change": warehouse_changed,
        }
        warehouse_coarse = build_coarse_warehouse_setup(
            warehouse_setup, warehouse_clean[0].frame_before
        )
        per_seed.append(
            (
                "warehouse",
                run_return_domain(
                    warehouse_setup,
                    warehouse_coarse,
                    warehouse_train,
                    warehouse_splits,
                    split_seed=f"return-warehouse/{seed}",
                    return_lookup=return_lookup,
                    return_artifact_id=return_artifact_id,
                ),
            )
        )
    results["domains"] = _average_return_seeds(per_seed)
    print_return_table(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "return-results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"\nwrote {args.output_dir / 'return-results.json'}")
    return 0


def _average_return_seeds(per_seed: list[tuple[str, dict]]) -> dict:
    domains: dict = {}
    for domain_name, _ in per_seed:
        runs = [run for name, run in per_seed if name == domain_name]
        averaged: dict = {"splits": {}}
        for split_name in runs[0]["splits"]:
            averaged["splits"][split_name] = {}
            for arm_name, cell in runs[0]["splits"][split_name].items():
                if arm_name == "regional":
                    combined: dict = {}
                    for run in runs:
                        for key, value in run["splits"][split_name]["regional"].items():
                            combined[key] = combined.get(key, 0.0) + value
                    count = len(runs)
                    averaged["splits"][split_name]["regional"] = {
                        key: round(value / count, 3) for key, value in combined.items()
                    }
                    continue
                cells = [run["splits"][split_name][arm_name] for run in runs]
                merged = _average_return_cells(cells)
                merged["halves"] = {
                    half: _average_return_cells(
                        [
                            run["splits"][split_name][arm_name]["halves"][half]
                            for run in runs
                        ]
                    )
                    for half in ("first", "second")
                }
                averaged["splits"][split_name][arm_name] = merged
        domains[domain_name] = averaged
    return domains


if __name__ == "__main__":
    raise SystemExit(main())
