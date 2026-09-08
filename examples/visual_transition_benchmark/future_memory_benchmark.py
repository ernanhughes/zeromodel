"""Structured-future-memory experiment on the visual transition benchmark.

Research question: does adding structured action-conditioned future memory
improve ZeroModel action selection, especially under visual perturbation,
without causing excessive correction harm?

Design (extends ``examples/visual_transition_benchmark`` rather than building
an unrelated harness):

- Drives the REAL environments (``TinyArcadeShooter.step`` /
  ``warehouse.model.step``) and the REAL renderers for every frame.
- Reuses, unmodified: P1 source/action encoding, P2 interaction ledger +
  manifests, P3 baseline predictor, P4A field schemas, P6 annotations,
  P18A transition evidence, P18B conformance, P18 transition analysis, and
  the arcade adapter's field schema / annotations / per-action expectations.
- Trains memories ONLY on ordinary on-policy transitions from disjoint
  train episodes. Dev / clean / nuisance / transition-change episodes are
  never fitted. Fault-category transitions are used ONLY for the
  expected-vs-observed verification stress, never for training or selection.

Systems: A baseline; B baseline + empirical gate; C baseline + compiled
ridge gate; D shared-relevance predictor + empirical gate.

Run (from ``examples/``)::

    PYTHONPATH=../packages/core/src ../venv/Scripts/python -m \\
        visual_transition_benchmark.future_memory_benchmark --seeds 0,1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _part in ("packages/core/src", "examples"):
    _path = str(_REPO_ROOT / _part)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from zeromodel.perception import (  # noqa: E402
    DeclarationScopeDTO,
    DiscreteActionSchemaDTO,
    FutureMemoryValidityDTO,
    MemoryAuthorityContextDTO,
    SourceImageEncoderSpecDTO,
    TransitionActionDeclarationDTO,
    TransitionExpectationSetDTO,
    TransitionModelConfigDTO,
    VisualTransitionAnalysisDTO,
    WorldActionPolicyDTO,
    build_dataset_manifest,
    build_grid_field_schema,
    build_transition_evidence_vpm,
    encode_discrete_action,
    encode_source_array,
    evaluate_transition_conformance,
    fit_action_conditioned_transition_model,
    fit_baseline_nearest_neighbor,
    fit_compiled_transition_model,
    fit_shared_field_relevance,
    predict_action_with_future_memory,
    predict_baseline_action,
    predict_relevance_weighted_action,
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
    ACTIONS as ARCADE_ACTIONS,
)
from zeromodel.video.arcade_policy.model import ShooterConfig  # noqa: E402
from visual_transition_benchmark import dataset as arcade_ds  # noqa: E402
from visual_transition_benchmark import zeromodel_adapter as arcade_zm  # noqa: E402
from visual_transition_benchmark.domains.warehouse import dataset as wh_ds  # noqa: E402
from visual_transition_benchmark.domains.warehouse import model as wh_model  # noqa: E402

# ---------------------------------------------------------------------------
# Declared, frozen experiment parameters (fixed a priori; nothing is tuned on
# dev/eval — dev is reported as a replication).
# ---------------------------------------------------------------------------

SPEC = SourceImageEncoderSpecDTO(color_space="L")
CHANGE_THRESHOLD = 8
MODEL_CONFIG = TransitionModelConfigDTO(
    neighbor_count=16,
    min_support=10,
    ood_spread_factor=3.0,
    ambiguity_threshold=0.2,
    change_epsilon=1e-3,
)
POLICY = WorldActionPolicyDTO(
    candidate_count=3,
    min_support=10,
    low_confidence_threshold=0.35,
    reject_on_contradiction=True,
    reject_on_ood=True,
    reject_on_insufficient=False,
)
RIDGE_ALPHA = 1.0
VERIFICATION_TOLERANCE = 0.02
VERIFICATION_EPSILON = 0.01
NUISANCE_BACKGROUND_SHIFT = 30
NUISANCE_NOISE_AMPLITUDE = 3
TRAIN_EPISODES = 24
EVAL_EPISODES = 10
STEPS_PER_EPISODE = 12

ARCADE_CHANGED_WAVE = (6, 0, 5, 1)
WAREHOUSE_CHANGED_GOAL = (1, 1)


# ---------------------------------------------------------------------------
# Correct-action labels (harness task definitions, not production claims).
# Arcade mirrors the repo's own optimal policy table
# (video.arcade_policy.model._action_values). Warehouse is greedy navigation
# to the goal cell among passable moves; pushes/door are distractors.
# ---------------------------------------------------------------------------


def arcade_label(tank_x: int, target_x: Optional[int], cooldown: int) -> str:
    if target_x is None:
        return "STAY"
    if cooldown == 0 and tank_x == target_x:
        return "FIRE"
    if tank_x > target_x:
        return "LEFT"
    if tank_x < target_x:
        return "RIGHT"
    return "STAY"


_WAREHOUSE_MOVES = (
    ("UP", (-1, 0)),
    ("DOWN", (1, 0)),
    ("LEFT", (0, -1)),
    ("RIGHT", (0, 1)),
)


def _warehouse_blocked(state: wh_model.WarehouseState, position) -> bool:
    if wh_model.is_wall(position):
        return True
    if position == wh_model.DOOR_POSITION:
        return not state.door_open
    if state.crate_at(position) is not None:
        return True
    return False


def warehouse_label(
    state: wh_model.WarehouseState, goal: tuple[int, int]
) -> str:
    def _manhattan(position) -> int:
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

    if state.robot == goal:
        return "WAIT"
    best: Optional[str] = None
    best_distance = _manhattan(state.robot)
    for name, (delta_row, delta_col) in _WAREHOUSE_MOVES:
        candidate = (state.robot[0] + delta_row, state.robot[1] + delta_col)
        if _warehouse_blocked(state, candidate):
            continue
        distance = _manhattan(candidate)
        if distance < best_distance:
            best, best_distance = f"MOVE_{name}", distance
    return best or "WAIT"


# ---------------------------------------------------------------------------
# Episode generation (real environments, real renderers).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedTransition:
    episode_id: str
    step_number: int
    frame_before: np.ndarray
    frame_after: np.ndarray
    action: str
    label: str
    state_before: dict
    is_faulty: bool = False


def _arcade_start(rng: random.Random, config: ShooterConfig) -> arcade_ds.ArcadeState:
    aliens = tuple(
        column for column in config.wave if rng.random() < 0.6
    ) or (config.wave[0],)
    return arcade_ds.ArcadeState(
        tank_x=rng.randint(0, config.width - 1),
        aliens=aliens,
        cooldown=rng.choice((0, 0, 0, 1)),
    )


def generate_arcade_episodes(
    *,
    prefix: str,
    episode_count: int,
    seed_offset: int,
    config: ShooterConfig,
) -> list[GeneratedTransition]:
    transitions: list[GeneratedTransition] = []
    for index in range(episode_count):
        rng = random.Random(seed_offset + index)
        episode_id = f"{prefix}-{index:04d}"
        state = _arcade_start(rng, config)
        for step_number in range(STEPS_PER_EPISODE):
            if state.target_x is None:
                break
            action = arcade_label(state.tank_x, state.target_x, state.cooldown)
            after = arcade_ds.true_next_state(state, action)
            transitions.append(
                GeneratedTransition(
                    episode_id=episode_id,
                    step_number=step_number,
                    frame_before=arcade_ds.render(state),
                    frame_after=arcade_ds.render(after),
                    action=action,
                    label=action,  # on-policy greedy: enacted == optimal
                    state_before=state.as_dict(),
                )
            )
            state = after
    return transitions


def _warehouse_start(rng: random.Random) -> wh_model.WarehouseState:
    cells = [
        (row, col)
        for row in wh_model.INTERIOR
        for col in wh_model.INTERIOR
        if (row, col) != wh_model.DOOR_POSITION
    ]
    rng.shuffle(cells)
    robot = cells.pop()
    crate_count = rng.choice((0, 1, 1, 2))
    crates = tuple(sorted(cells[:crate_count]))
    return wh_model.WarehouseState(
        robot=robot,
        crates=crates,
        door_open=rng.random() < 0.7,
        battery=rng.randint(1, wh_model.MAX_BATTERY),
    )


def generate_warehouse_episodes(
    *,
    prefix: str,
    episode_count: int,
    seed_offset: int,
    goal: tuple[int, int],
) -> list[GeneratedTransition]:
    """Rollouts with hash-gated exploration.

    The behavior is a deterministic function of state (greedy navigation,
    with every 5th state-hash exploring a hash-chosen action), so identical
    sources never map to conflicting actions while off-policy consequences
    (pushes, door, blocked moves) still enter memory.
    """
    transitions: list[GeneratedTransition] = []
    for index in range(episode_count):
        rng = random.Random(seed_offset + index)
        episode_id = f"{prefix}-{index:04d}"
        state = _warehouse_start(rng)
        waits = 0
        for step_number in range(STEPS_PER_EPISODE):
            label = warehouse_label(state, goal)
            digest = int(
                hashlib.sha256(
                    json.dumps(state.as_dict(), sort_keys=True).encode("utf-8")
                ).hexdigest(),
                16,
            )
            if digest % 5 == 0:
                action = wh_model.ACTIONS[digest % len(wh_model.ACTIONS)]
            else:
                action = label
            after = wh_model.step(state, action)
            transitions.append(
                GeneratedTransition(
                    episode_id=episode_id,
                    step_number=step_number,
                    frame_before=wh_ds.render(state),
                    frame_after=wh_ds.render(after),
                    action=action,
                    label=label,
                    state_before=state.as_dict(),
                )
            )
            state = after
            if action == "WAIT" and label == "WAIT":
                waits += 1
                if waits >= 2:
                    break
            else:
                waits = 0
    return transitions


# ---------------------------------------------------------------------------
# Nuisance perturbations (renderer-safe appearance shifts).
# ---------------------------------------------------------------------------


def perturb_background_shift(
    transitions: list[GeneratedTransition], shift: int = NUISANCE_BACKGROUND_SHIFT
) -> list[GeneratedTransition]:
    """Shift background pixels identically in both frames.

    Only pixels that are background in *both* frames move, so P18A deltas
    are preserved exactly while whole-image predictor distances shift: a
    DreamWAM-style nuisance shift.
    """
    perturbed: list[GeneratedTransition] = []
    for record in transitions:
        before = record.frame_before.copy()
        after = record.frame_after.copy()
        background = (record.frame_before == 0) & (record.frame_after == 0)
        before[background] = shift
        after[background] = shift
        perturbed.append(
            GeneratedTransition(
                episode_id=record.episode_id,
                step_number=record.step_number,
                frame_before=before,
                frame_after=after,
                action=record.action,
                label=record.label,
                state_before=record.state_before,
            )
        )
    return perturbed


def perturb_pixel_noise(
    transitions: list[GeneratedTransition],
    seed: int,
    amplitude: int = NUISANCE_NOISE_AMPLITUDE,
) -> list[GeneratedTransition]:
    """Sub-threshold pixel noise (below P18A change_threshold)."""
    rng = np.random.default_rng(seed)
    perturbed: list[GeneratedTransition] = []
    for record in transitions:
        before = np.clip(
            record.frame_before.astype(np.int16)
            + rng.integers(-amplitude, amplitude + 1, size=record.frame_before.shape),
            0,
            255,
        ).astype(np.uint8)
        after = np.clip(
            record.frame_after.astype(np.int16)
            + rng.integers(-amplitude, amplitude + 1, size=record.frame_after.shape),
            0,
            255,
        ).astype(np.uint8)
        perturbed.append(
            GeneratedTransition(
                episode_id=record.episode_id,
                step_number=record.step_number,
                frame_before=before,
                frame_after=after,
                action=record.action,
                label=record.label,
                state_before=record.state_before,
            )
        )
    return perturbed


# ---------------------------------------------------------------------------
# Perception wiring: sources, manifests, models.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainSetup:
    name: str
    action_labels: tuple[str, ...]
    field_schema: VPMFieldSchemaDTO
    annotations: tuple
    expectations_by_action: Mapping[str, tuple[TransitionExpectationDTO, ...]]


def _warehouse_setup() -> DomainSetup:
    probe = encode_source_array(
        np.zeros((34, 30), dtype=np.uint8), SPEC
    )
    schema = build_grid_field_schema(
        probe, tile_width=5, tile_height=2, channel_mode="joint"
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
        action: (change,)
        for action in wh_model.ACTIONS
        if action != "WAIT"
    }
    expectations["WAIT"] = (stable,)
    return DomainSetup(
        name="warehouse",
        action_labels=tuple(sorted(wh_model.ACTIONS)),
        field_schema=schema,
        annotations=(canvas,),
        expectations_by_action=expectations,
    )


def _arcade_setup() -> DomainSetup:
    return DomainSetup(
        name="arcade",
        action_labels=tuple(sorted(ARCADE_ACTIONS)),
        field_schema=arcade_zm.FIELD_SCHEMA,
        annotations=arcade_zm.ANNOTATIONS_TUPLE,
        expectations_by_action=dict(arcade_zm.EXPECTATIONS_BY_ACTION),
    )


def _encode_frame(frame: np.ndarray) -> SourceVPMDTO:
    array = np.asarray(frame, dtype=np.uint8)
    assert array.ndim == 2, f"expected grayscale frame, got shape {array.shape}"
    return encode_source_array(array, SPEC)


def build_manifest(
    transitions: list[GeneratedTransition],
    action_schema: DiscreteActionSchemaDTO,
    *,
    split_seed: str,
) -> tuple[object, dict[str, SourceVPMDTO]]:
    interactions: list[RecordedInteractionDTO] = []
    sources: dict[str, SourceVPMDTO] = {}
    for record in transitions:
        before = _encode_frame(record.frame_before)
        after = _encode_frame(record.frame_after)
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
        source_encoder_spec_ids=[SPEC.encoder_spec_id],
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


@dataclass(frozen=True)
class FittedMemory:
    predictor: object
    empirical: object
    compiled: object
    relevance: object
    shared_predictor_weights: Optional[dict]


def fit_all(
    manifest,
    sources: dict[str, SourceVPMDTO],
    setup: DomainSetup,
) -> FittedMemory:
    predictor = fit_baseline_nearest_neighbor(
        manifest, sources, training_split="all"
    )
    empirical = fit_action_conditioned_transition_model(
        manifest,
        sources,
        setup.field_schema,
        config=MODEL_CONFIG,
        training_split="all",
        change_threshold=CHANGE_THRESHOLD,
        annotations=tuple(setup.annotations),
    )
    compiled = fit_compiled_transition_model(
        manifest,
        sources,
        setup.field_schema,
        config=MODEL_CONFIG,
        training_split="all",
        change_threshold=CHANGE_THRESHOLD,
        ridge_alpha=RIDGE_ALPHA,
        annotations=tuple(setup.annotations),
    )
    relevance = fit_shared_field_relevance(
        manifest, sources, setup.field_schema, training_split="all"
    )
    return FittedMemory(
        predictor=predictor,
        empirical=empirical,
        compiled=compiled,
        relevance=relevance,
        shared_predictor_weights=dict(relevance.weights),
    )


# ---------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemSpec:
    name: str
    predictor_kind: str  # "uniform" | "shared"
    future_kind: Optional[str]  # None | "empirical" | "compiled"


SYSTEMS = (
    SystemSpec("A baseline", "uniform", None),
    SystemSpec("B +empirical gate", "uniform", "empirical"),
    SystemSpec("C +compiled ridge", "uniform", "compiled"),
    SystemSpec("D +shared repr", "shared", "empirical"),
)


def _projection_matches(
    projected,
    observed,
    *,
    tolerance: float,
    change_epsilon: float,
) -> bool:
    """Independent per-field match check (ignores declared expectations)."""
    del change_epsilon
    observed_by_id = {item.field_id: item for item in observed.fields}
    for field in projected.fields:
        observed_field = observed_by_id[field.field_id]
        allowed = tolerance + 2.0 * field.signed_change_dispersion
        if (
            abs(observed_field.after_mean - field.expected_after_mean)
            > allowed
        ):
            return False
    return True


def evaluate_split(
    system: SystemSpec,
    transitions: list[GeneratedTransition],
    setup: DomainSetup,
    memory: FittedMemory,
    declarations: DeclarationScopeDTO,
) -> tuple[dict, dict]:
    """Score one split and return (metrics, halves).

    Memory validity accumulates online within the split: each query consults
    the validity learned from earlier queries' verification events, so the
    second half measures the informed agent loop against the cold first half.
    """
    rows: list[dict] = []
    validity: dict[str, FutureMemoryValidityDTO] = {}
    for index, record in enumerate(transitions):
        before = _encode_frame(record.frame_before)
        after = _encode_frame(record.frame_after)
        weights = (
            memory.shared_predictor_weights
            if system.predictor_kind == "shared"
            else None
        )
        uniform_baseline = predict_baseline_action(memory.predictor, before)
        if system.predictor_kind == "shared":
            baseline = predict_relevance_weighted_action(
                memory.predictor, before, setup.field_schema, memory.relevance
            )
        else:
            baseline = uniform_baseline
        base_top = baseline.candidates[0].action_label
        # System D reports the relevance-weighted ranking as its baseline;
        # the gate below always consumes the verbatim P3 ranking for audit
        # comparability, with the shared representation shaping projection.
        base_ok = base_top == record.label
        row = {
            "episode_id": record.episode_id,
            "step_number": record.step_number,
            "vetoes": 0,
            "stale_flags": 0,
            "may_veto_flags": 0,
            "base_ok": base_ok,
            "coupled_ok": False,
            "accepted": True,
            "reranked": False,
            "rejected": False,
            "verdict": None,
            "matched": None,
            "matched_conformant": None,
            "mae": None,
            "supported": False,
            "dir_hits": 0,
            "dir_total": 0,
            "prec_hits": 0,
            "prec_total": 0,
            "rec_hits": 0,
            "rec_total": 0,
            "disp_confirmed": None,
            "disp_mismatch": None,
        }
        if system.future_kind is None:
            selected, accepted = base_top, True
        else:
            transition_model = (
                memory.empirical
                if system.future_kind == "empirical"
                else memory.compiled
            )
            out = predict_action_with_future_memory(
                memory.predictor,
                transition_model,
                before,
                setup.field_schema,
                declarations=declarations,
                policy=POLICY,
                field_weights=weights,
                baseline_override=baseline
                if system.predictor_kind == "shared"
                else None,
                authority=MemoryAuthorityContextDTO.create(validity)
                if system.future_kind is not None
                else None,
            )
            accepted = out.accepted
            selected = out.selected_action
            gate_top = out.baseline.candidates[0].action_label
            row["reranked"] = bool(accepted and selected != gate_top)
            row["rejected"] = not accepted
            row["vetoes"] = sum(
                1
                for item in out.candidates
                if item.status == "contradicted_by_transition_expectation"
            )
            row["stale_flags"] = sum(
                1 for item in out.candidates if item.memory_authority == "STALE"
            )
            row["may_veto_flags"] = sum(
                1 for item in out.candidates if item.memory_authority == "MAY_VETO"
            )
        coupled_ok = (selected == record.label) if accepted else False
        row["coupled_ok"] = coupled_ok
        row["accepted"] = accepted
        if system.future_kind is not None:
            transition_model = (
                memory.empirical
                if system.future_kind == "empirical"
                else memory.compiled
            )
            projected = project_expected_transition(
                transition_model,
                before,
                record.action,
                setup.field_schema,
                field_weights=weights,
                min_support=POLICY.min_support,
            )
            if projected.status == "supported":
                row["supported"] = True
                observed = build_transition_evidence_vpm(
                    before,
                    after,
                    setup.field_schema,
                    annotations=tuple(setup.annotations),
                    change_threshold=CHANGE_THRESHOLD,
                )
                observed_by_id = {
                    item.field_id: item for item in observed.fields
                }
                errors = [
                    abs(
                        field.expected_after_mean
                        - observed_by_id[field.field_id].after_mean
                    )
                    for field in projected.fields
                ]
                row["mae"] = float(np.mean(errors))
                for field in projected.fields:
                    observed_field = observed_by_id[field.field_id]
                    expected_direction = (
                        1
                        if field.expected_mean_signed_change
                        > MODEL_CONFIG.change_epsilon
                        else (
                            -1
                            if field.expected_mean_signed_change
                            < -MODEL_CONFIG.change_epsilon
                            else 0
                        )
                    )
                    observed_direction = (
                        1
                        if observed_field.mean_signed_change
                        > MODEL_CONFIG.change_epsilon
                        else (
                            -1
                            if observed_field.mean_signed_change
                            < -MODEL_CONFIG.change_epsilon
                            else 0
                        )
                    )
                    if expected_direction != 0:
                        row["dir_total"] += 1
                        row["dir_hits"] += (
                            1
                            if expected_direction == observed_direction
                            else 0
                        )
                    predicted_changed = (
                        abs(field.expected_mean_signed_change)
                        > MODEL_CONFIG.change_epsilon
                    )
                    actually_changed = observed_field.changed_value_count > 0
                    row["prec_total"] += 1 if predicted_changed else 0
                    row["prec_hits"] += (
                        1 if predicted_changed and actually_changed else 0
                    )
                    row["rec_total"] += 1 if actually_changed else 0
                    row["rec_hits"] += (
                        1 if predicted_changed and actually_changed else 0
                    )
            # Expected-vs-observed verification loop for the enacted action.
            observed = build_transition_evidence_vpm(
                before,
                after,
                setup.field_schema,
                annotations=tuple(setup.annotations),
                change_threshold=CHANGE_THRESHOLD,
            )
            action_declaration = TransitionActionDeclarationDTO.create(
                action_type="discrete_action",
                payload={"action_label": record.action},
            )
            enacted_expectations = setup.expectations_by_action.get(
                record.action, ()
            )
            if enacted_expectations:
                report = evaluate_transition_conformance(
                    observed,
                    enacted_expectations,
                    tuple(setup.annotations),
                )
                expectation_set = TransitionExpectationSetDTO.create(
                    enacted_expectations
                )
            else:
                report = evaluate_transition_conformance(
                    observed,
                    (next(iter(setup.expectations_by_action.values()))),
                    tuple(setup.annotations),
                )
                expectation_set = TransitionExpectationSetDTO.create(
                    (next(iter(setup.expectations_by_action.values()))),
                )
            analysis = VisualTransitionAnalysisDTO.create(
                transition=observed,
                action=action_declaration,
                expectation_set=expectation_set,
                conformance_report=report,
            )
            projected_enacted = project_expected_transition(
                transition_model,
                before,
                record.action,
                setup.field_schema,
                field_weights=weights,
                min_support=POLICY.min_support,
            )
            verdict = verify_expected_transition(
                projected_enacted,
                analysis,
                tolerance=VERIFICATION_TOLERANCE,
                change_epsilon=VERIFICATION_EPSILON,
            )
            row["verdict"] = verdict.status
            matched = _projection_matches(
                projected_enacted,
                observed,
                tolerance=VERIFICATION_TOLERANCE,
                change_epsilon=VERIFICATION_EPSILON,
            )
            row["matched"] = matched
            if analysis.status != "nonconformant":
                row["matched_conformant"] = matched
            mean_dispersion = float(
                np.mean(
                    [
                        field.signed_change_dispersion
                        for field in projected_enacted.fields
                    ]
                )
            )
            if verdict.status == "confirmed":
                row["disp_confirmed"] = mean_dispersion
            elif verdict.status == "future_projection_mismatch":
                row["disp_mismatch"] = mean_dispersion
            # Close the agent loop: this verification event becomes validity
            # evidence regulating the NEXT decision for this action.
            validity[record.action] = record_verification_event(
                validity.get(record.action),
                verdict,
                transition_model_id=transition_model.model_id,
                action_label=record.action,
                field_schema_id=setup.field_schema.field_schema_id,
                training_dataset_id=transition_model.dataset_id,
            )
        rows.append(row)
    metrics = _aggregate_rows(rows, system.future_kind is not None)
    # Halves are per-episode (first vs second half of each episode's steps,
    # pooled): with validity accumulating online, the second half measures
    # the informed loop against the cold first half on the same episodes.
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
        "first": _aggregate_rows(first_rows, system.future_kind is not None),
        "second": _aggregate_rows(second_rows, system.future_kind is not None),
    }
    return metrics, metrics["halves"]


def _aggregate_rows(rows: list[dict], has_future: bool) -> dict:
    total = len(rows)
    base_correct = sum(1 for row in rows if row["base_ok"])
    coupled_correct = sum(1 for row in rows if row["coupled_ok"])
    answered = sum(1 for row in rows if row["accepted"])
    gain = sum(1 for row in rows if not row["base_ok"] and row["coupled_ok"])
    harm = sum(1 for row in rows if row["base_ok"] and not row["coupled_ok"])
    rerank = sum(1 for row in rows if row["reranked"])
    rejects = sum(1 for row in rows if row["rejected"])
    maes = [row["mae"] for row in rows if row["mae"] is not None]
    dir_hits = sum(row["dir_hits"] for row in rows)
    dir_total = sum(row["dir_total"] for row in rows)
    prec_hits = sum(row["prec_hits"] for row in rows)
    prec_total = sum(row["prec_total"] for row in rows)
    rec_hits = sum(row["rec_hits"] for row in rows)
    rec_total = sum(row["rec_total"] for row in rows)
    verification: dict[str, int] = {}
    matched = matched_total = matched_conformant = matched_conformant_total = 0
    disp_confirmed: list[float] = []
    disp_mismatch: list[float] = []
    for row in rows:
        if row["verdict"] is not None:
            verification[row["verdict"]] = verification.get(row["verdict"], 0) + 1
            matched_total += 1
            matched += 1 if row["matched"] else 0
            if row["matched_conformant"] is not None:
                matched_conformant_total += 1
                matched_conformant += 1 if row["matched_conformant"] else 0
        if row["disp_confirmed"] is not None:
            disp_confirmed.append(row["disp_confirmed"])
        if row["disp_mismatch"] is not None:
            disp_mismatch.append(row["disp_mismatch"])
    precision = prec_hits / prec_total if prec_total else float("nan")
    recall = rec_hits / rec_total if rec_total else float("nan")
    denominator = precision + recall
    f1 = (
        2 * precision * recall / denominator
        if prec_total and rec_total and denominator
        else float("nan")
    )
    vetoes = [row["vetoes"] for row in rows]
    stale = [row["stale_flags"] for row in rows]
    may_veto = [row["may_veto_flags"] for row in rows]
    return {
        "n": total,
        "base_acc": base_correct / total if total else float("nan"),
        "coupled_acc": coupled_correct / total if total else float("nan"),
        "base_coverage": 1.0,
        "coverage": answered / total if total else float("nan"),
        "gain": gain / total if total else float("nan"),
        "harm": harm / total if total else float("nan"),
        "net": (gain - harm) / total if total else float("nan"),
        "rerank_rate": rerank / total if total else float("nan"),
        "reject_rate": rejects / total if total else float("nan"),
        "future_mae": float(np.mean(maes)) if maes else float("nan"),
        "future_supported_frac": len(maes) / total if has_future and total else float("nan"),
        "signed_direction_acc": dir_hits / dir_total if dir_total else float("nan"),
        "changed_precision": precision,
        "changed_recall": recall,
        "changed_f1": f1,
        "verification": verification,
        "projection_match_rate": matched / matched_total
        if has_future and matched_total
        else float("nan"),
        "projection_match_rate_conformant": (
            matched_conformant / matched_conformant_total
            if has_future and matched_conformant_total
            else float("nan")
        ),
        "mean_dispersion_confirmed": float(np.mean(disp_confirmed))
        if disp_confirmed
        else float("nan"),
        "mean_dispersion_mismatch": float(np.mean(disp_mismatch))
        if disp_mismatch
        else float("nan"),
        "vetoes_per_query": float(np.mean(vetoes)) if vetoes else float("nan"),
        "stale_flags_per_query": float(np.mean(stale)) if stale else float("nan"),
        "may_veto_flags_per_query": float(np.mean(may_veto))
        if may_veto
        else float("nan"),
    }


def run_domain(
    setup: DomainSetup,
    train: list[GeneratedTransition],
    splits: dict[str, list[GeneratedTransition]],
    *,
    split_seed: str,
) -> dict:
    action_schema = DiscreteActionSchemaDTO.from_labels(
        sorted({record.action for record in train})
    )
    manifest, sources = build_manifest(train, action_schema, split_seed=split_seed)
    memory = fit_all(manifest, sources, setup)
    declarations = DeclarationScopeDTO.create(
        dict(setup.expectations_by_action), tuple(setup.annotations)
    )
    result: dict = {
        "field_weights_shared": list(memory.relevance.weights),
        "splits": {},
    }
    for split_name, transitions in splits.items():
        result["splits"][split_name] = {}
        for system in SYSTEMS:
            metrics, halves = evaluate_split(
                system, transitions, setup, memory, declarations
            )
            metrics["halves"] = halves
            result["splits"][split_name][system.name] = metrics
    return result


def print_table(results: dict) -> None:
    for domain_name, domain in results["domains"].items():
        print(f"\n### {domain_name} (seeds {results['seeds']})")
        print(
            "| split | system | acc(base->coupled) | coverage | gain | harm | "
            "net | rerank | future MAE | changed F1 |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        for split_name, systems in domain["splits"].items():
            for system in SYSTEMS:
                metrics = systems[system.name]
                print(
                    f"| {split_name} | {system.name} | "
                    f"{metrics['base_acc']:.3f}->{metrics['coupled_acc']:.3f} | "
                    f"{metrics['coverage']:.3f} | {metrics['gain']:.3f} | "
                    f"{metrics['harm']:.3f} | {metrics['net']:+.3f} | "
                    f"{metrics['rerank_rate']:.3f} | "
                    f"{metrics['future_mae']:.4f} | {metrics['changed_f1']:.3f} |"
                )
        print(
            "shared field weights (min/mean/max):",
            (
                round(
                    min(
                        weight for _, weight in domain["field_weights_shared"]
                    ),
                    3,
                ),
                round(
                    sum(weight for _, weight in domain["field_weights_shared"])
                    / len(domain["field_weights_shared"]),
                    3,
                ),
                round(
                    max(
                        weight for _, weight in domain["field_weights_shared"]
                    ),
                    3,
                ),
            ),
        )
        for split_name, systems in domain["splits"].items():
            for system in SYSTEMS:
                if system.future_kind:
                    print(
                        f"  [{split_name}] {system.name} verification:",
                        systems[system.name]["verification"],
                    )
        print("  authority halves (first -> second): harm / coverage / net")
        for split_name, systems in domain["splits"].items():
            for system in SYSTEMS:
                if system.future_kind:
                    halves = systems[system.name]["halves"]
                    first, second = halves["first"], halves["second"]
                    print(
                        f"  [{split_name}] {system.name}: "
                        f"{first['harm']:.3f}->{second['harm']:.3f} / "
                        f"{first['coverage']:.3f}->{second['coverage']:.3f} / "
                        f"{first['net']:+.3f}->{second['net']:+.3f}"
                    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--train-episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--steps", type=int, default=STEPS_PER_EPISODE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "future_memory_benchmark",
    )
    args = parser.parse_args(argv)
    seeds = [int(part) for part in args.seeds.split(",")]
    results: dict = {
        "seeds": seeds,
        "policy": POLICY.policy_id,
        "domains": {},
    }
    per_seed: list[dict] = []
    for seed in seeds:
        # --- arcade ---
        arcade_setup = _arcade_setup()
        arcade_train = generate_arcade_episodes(
            prefix="train",
            episode_count=args.train_episodes,
            seed_offset=seed * 10_000,
            config=ShooterConfig(),
        )
        arcade_clean = generate_arcade_episodes(
            prefix="clean",
            episode_count=args.eval_episodes,
            seed_offset=seed * 10_000 + 1_000_000,
            config=ShooterConfig(),
        )
        arcade_changed = generate_arcade_episodes(
            prefix="changed",
            episode_count=args.eval_episodes,
            seed_offset=seed * 10_000 + 2_000_000,
            config=ShooterConfig(wave=ARCADE_CHANGED_WAVE),
        )
        arcade_splits = {
            "clean": arcade_clean,
            "nuisance-background": perturb_background_shift(arcade_clean),
            "nuisance-noise": perturb_pixel_noise(arcade_clean, seed + 77),
            "transition-change": arcade_changed,
        }
        per_seed.append(
            ("arcade", run_domain(arcade_setup, arcade_train, arcade_splits, split_seed=f"future-memory-arcade/{seed}"))
        )
        # --- warehouse ---
        warehouse_setup = _warehouse_setup()
        warehouse_train = generate_warehouse_episodes(
            prefix="train",
            episode_count=args.train_episodes,
            seed_offset=seed * 10_000 + 5_000,
            goal=(3, 3),
        )
        warehouse_clean = generate_warehouse_episodes(
            prefix="clean",
            episode_count=args.eval_episodes,
            seed_offset=seed * 10_000 + 1_005_000,
            goal=(3, 3),
        )
        warehouse_changed = generate_warehouse_episodes(
            prefix="changed",
            episode_count=args.eval_episodes,
            seed_offset=seed * 10_000 + 2_005_000,
            goal=WAREHOUSE_CHANGED_GOAL,
        )
        warehouse_splits = {
            "clean": warehouse_clean,
            "nuisance-background": perturb_background_shift(warehouse_clean),
            "nuisance-noise": perturb_pixel_noise(warehouse_clean, seed + 913),
            "transition-change": warehouse_changed,
        }
        per_seed.append(
            (
                "warehouse",
                run_domain(
                    warehouse_setup,
                    warehouse_train,
                    warehouse_splits,
                    split_seed=f"future-memory-warehouse/{seed}",
                ),
            )
        )
    results["domains"] = _average_seeds(per_seed)
    print_table(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "future-memory-results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"\nwrote {args.output_dir / 'future-memory-results.json'}")
    return 0


_METRIC_KEYS = (
    "base_acc",
    "coupled_acc",
    "base_coverage",
    "coverage",
    "gain",
    "harm",
    "net",
    "rerank_rate",
    "reject_rate",
    "future_mae",
    "future_supported_frac",
    "signed_direction_acc",
    "changed_precision",
    "changed_recall",
    "changed_f1",
    "mean_dispersion_confirmed",
    "mean_dispersion_mismatch",
    "projection_match_rate",
    "projection_match_rate_conformant",
    "vetoes_per_query",
    "stale_flags_per_query",
    "may_veto_flags_per_query",
)


def _average_cells(cells: list[dict]) -> dict:
    merged: dict = {"n": sum(cell["n"] for cell in cells)}
    for key in _METRIC_KEYS:
        values = [
            cell[key]
            for cell in cells
            if not (isinstance(cell[key], float) and np.isnan(cell[key]))
        ]
        merged[key] = float(np.mean(values)) if values else float("nan")
    verification: dict[str, int] = {}
    for cell in cells:
        for status, count in cell["verification"].items():
            verification[status] = verification.get(status, 0) + count
    merged["verification"] = verification
    return merged


def _average_seeds(per_seed: list[tuple[str, dict]]) -> dict:
    domains: dict = {}
    for domain_name, _ in per_seed:
        runs = [run for name, run in per_seed if name == domain_name]
        averaged: dict = {
            "field_weights_shared": runs[0]["field_weights_shared"],
            "splits": {},
        }
        for split_name in runs[0]["splits"]:
            averaged["splits"][split_name] = {}
            for system_name in runs[0]["splits"][split_name]:
                cells = [run["splits"][split_name][system_name] for run in runs]
                merged = _average_cells(cells)
                merged["halves"] = {
                    half: _average_cells(
                        [run["splits"][split_name][system_name]["halves"][half] for run in runs]
                    )
                    for half in ("first", "second")
                }
                averaged["splits"][split_name][system_name] = merged
        domains[domain_name] = averaged
    return domains


if __name__ == "__main__":
    raise SystemExit(main())
