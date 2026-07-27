"""Arcade domain adapter: declares candidate regions and evidence requirements
for the six required arcade benchmark cases, and builds development/evaluation
samples from the existing ``dataset.py`` transition generator.

Ground-truth discipline: ``dataset.TransitionRecord`` does not expose what was
*actually rendered* for a faulty transition (only the logically-correct
``state_after`` and the observed-from-pixels ``observed_changed_components``).
For "did this band change" (presence), the observed-from-pixels signal is a
perfectly good, already-computed ground truth and works across every
category, faulty or not. For numeric/position/identity properties -- where
"the correct value" and "what a fault deliberately rendered instead" are
different things the decoder cannot be blamed for conflating -- these cases
draw development/evaluation samples from ``ORDINARY_CATEGORIES`` only, so
"decoding accuracy" measures representation fidelity, not fault detection
(fault detection is exactly what stage 2's contracts already cover).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Tuple

from visual_transition_benchmark import dataset as ds
from zeromodel.video.arcade_policy.rendering import (
    COOLDOWN_BLOCKED_VALUE,
    COOLDOWN_READY_VALUE,
)

from visual_transition_benchmark.compiler.candidates import RegionGeometry
from visual_transition_benchmark.compiler.contracts import VisualEvidenceRequirement
from visual_transition_benchmark.compiler.evaluate import DevelopmentSample

CANVAS_SHAPE = (ds.FRAME_HEIGHT, ds.WIDTH_PX)
READY = COOLDOWN_READY_VALUE / 255.0
BLOCKED = COOLDOWN_BLOCKED_VALUE / 255.0

TANK_BAND = RegionGeometry(
    region_id="tank_band",
    canvas_shape=CANVAS_SHAPE,
    y0=11,
    y1=14,
    x0=0,
    x1=ds.WIDTH_PX,
    cell_height=3,
    cell_width=ds.CELL_PIXELS,
)
ALIEN_BAND = RegionGeometry(
    region_id="alien_band",
    canvas_shape=CANVAS_SHAPE,
    y0=2,
    y1=5,
    x0=0,
    x1=ds.WIDTH_PX,
    cell_height=3,
    cell_width=ds.CELL_PIXELS,
)
COOLDOWN_REGION = RegionGeometry(
    region_id="cooldown_region",
    canvas_shape=CANVAS_SHAPE,
    y0=7,
    y1=9,
    x0=ds.WIDTH_PX - ds.CELL_PIXELS,
    x1=ds.WIDTH_PX,
    cell_height=1,
    cell_width=ds.CELL_PIXELS,
)


@dataclass(frozen=True)
class CompilerCase:
    name: str
    requirement: VisualEvidenceRequirement
    region: RegionGeometry
    canonical_levels: Tuple[float, ...]
    min_decoding_accuracy: float
    build_samples: Callable[[int, int], Tuple[DevelopmentSample, ...]]


def _tank_position_samples(count: int, seed: int) -> Tuple[DevelopmentSample, ...]:
    samples = []
    for i, category in enumerate(ds.ORDINARY_CATEGORIES):
        for j in range(count):
            rec = ds.build_transition(
                episode_id=f"tank-{category}",
                step_number=j,
                seed=seed + i * 1000 + j,
                category=category,
            )
            samples.append(
                DevelopmentSample(
                    sample_id=rec.transition_id,
                    frame_before=rec.frame_before,
                    frame_after=rec.frame_after,
                    true_before=(0, rec.state_before["tank_x"]),
                    true_after=(0, rec.state_after["tank_x"]),
                    is_unrelated=(
                        rec.state_before["tank_x"] == rec.state_after["tank_x"]
                    ),
                )
            )
    return tuple(samples)


def _cooldown_samples(count: int, seed: int) -> Tuple[DevelopmentSample, ...]:
    samples = []
    for i, category in enumerate(ds.ORDINARY_CATEGORIES):
        for j in range(count):
            rec = ds.build_transition(
                episode_id=f"cooldown-{category}",
                step_number=j,
                seed=seed + i * 1000 + j,
                category=category,
            )
            true_before = BLOCKED if rec.state_before["cooldown"] else READY
            true_after = BLOCKED if rec.state_after["cooldown"] else READY
            samples.append(
                DevelopmentSample(
                    sample_id=rec.transition_id,
                    frame_before=rec.frame_before,
                    frame_after=rec.frame_after,
                    true_before=true_before,
                    true_after=true_after,
                    is_unrelated=(true_before == true_after),
                )
            )
    return tuple(samples)


def _presence_samples(
    band: str, region: RegionGeometry, count: int, seed: int
) -> Tuple[DevelopmentSample, ...]:
    """Ground truth is whether *this declared region's* pixels actually
    changed -- not the privileged per-component attribution used elsewhere.
    The two can diverge (a fault's pixel edit can land inside a region's
    bounds without belonging to that component), and the presence decoder can
    only ever answer the region-pixel question, so scoring it against the
    component-attribution answer conflates two different questions."""

    samples = []
    for i, category in enumerate(ds.ALL_CATEGORIES):
        for j in range(count):
            rec = ds.build_transition(
                episode_id=f"presence-{band}-{category}",
                step_number=j,
                seed=seed + i * 1000 + j,
                category=category,
            )
            before_patch = rec.frame_before[
                region.y0 : region.y1, region.x0 : region.x1
            ]
            after_patch = rec.frame_after[region.y0 : region.y1, region.x0 : region.x1]
            changed = bool((before_patch != after_patch).any())
            samples.append(
                DevelopmentSample(
                    sample_id=rec.transition_id,
                    frame_before=rec.frame_before,
                    frame_after=rec.frame_after,
                    true_before=False,
                    true_after=changed,
                    is_unrelated=False,
                    fault_present=rec.is_faulty,
                )
            )
    return tuple(samples)


def _identity_samples(count: int, seed: int) -> Tuple[DevelopmentSample, ...]:
    """The alien sprite carries no identity marker (no dots, no label) --
    only a column position renders. This case exists to demonstrate, not
    dodge, that limitation: a synthetic privileged identity (uncorrelated
    with anything visual) is assigned per sample purely for evaluation, so a
    decoder that can only read pixels has no way to recover it regardless of
    resolution or aggregation."""

    rng = random.Random(seed)
    samples = []
    for i, category in enumerate(ds.ORDINARY_CATEGORIES):
        for j in range(count):
            rec = ds.build_transition(
                episode_id=f"identity-{category}",
                step_number=j,
                seed=seed + i * 1000 + j,
                category=category,
            )
            samples.append(
                DevelopmentSample(
                    sample_id=rec.transition_id,
                    frame_before=rec.frame_before,
                    frame_after=rec.frame_after,
                    true_before=rng.randint(0, 10_000),
                    true_after=rng.randint(0, 10_000),
                    is_unrelated=False,
                )
            )
    return tuple(samples)


def build_cases() -> Tuple[CompilerCase, ...]:
    return (
        CompilerCase(
            name="tank_presence",
            requirement=VisualEvidenceRequirement(
                domain_name="arcade",
                component_type="tank",
                property_name="presence",
                evidence_kind="presence",
                candidate_region_id="tank_band",
                comparison="changed",
            ),
            region=TANK_BAND,
            canonical_levels=(),
            min_decoding_accuracy=0.95,
            build_samples=lambda count, seed: _presence_samples(
                "tank", TANK_BAND, count, seed
            ),
        ),
        CompilerCase(
            name="tank_position",
            requirement=VisualEvidenceRequirement(
                domain_name="arcade",
                component_type="tank",
                property_name="position",
                evidence_kind="spatial_position",
                candidate_region_id="tank_band",
                comparison="equal",
            ),
            region=TANK_BAND,
            canonical_levels=(),
            min_decoding_accuracy=0.95,
            build_samples=_tank_position_samples,
        ),
        CompilerCase(
            name="tank_direction",
            requirement=VisualEvidenceRequirement(
                domain_name="arcade",
                component_type="tank",
                property_name="direction",
                evidence_kind="signed_delta",
                candidate_region_id="tank_band",
                comparison="signed_delta",
            ),
            region=TANK_BAND,
            canonical_levels=(),
            min_decoding_accuracy=0.95,
            build_samples=_tank_position_samples,
        ),
        CompilerCase(
            name="tank_movement_magnitude",
            requirement=VisualEvidenceRequirement(
                domain_name="arcade",
                component_type="tank",
                property_name="movement_magnitude",
                evidence_kind="exact_magnitude",
                candidate_region_id="tank_band",
                comparison="exact_delta",
            ),
            region=TANK_BAND,
            canonical_levels=(),
            min_decoding_accuracy=0.95,
            build_samples=_tank_position_samples,
        ),
        CompilerCase(
            name="cooldown_value",
            requirement=VisualEvidenceRequirement(
                domain_name="arcade",
                component_type="cooldown",
                property_name="cooldown_value",
                evidence_kind="numeric_value",
                candidate_region_id="cooldown_region",
                expected_value_domain=(READY, BLOCKED),
                required_precision=15 / 255,
                comparison="equal",
            ),
            region=COOLDOWN_REGION,
            canonical_levels=(READY, BLOCKED),
            min_decoding_accuracy=0.95,
            build_samples=_cooldown_samples,
        ),
        CompilerCase(
            name="alien_target_identity",
            requirement=VisualEvidenceRequirement(
                domain_name="arcade",
                component_type="alien",
                property_name="target_identity",
                evidence_kind="visible_identity",
                candidate_region_id="alien_band",
                comparison="identity_equal",
                permits_identity_marker=True,
            ),
            region=ALIEN_BAND,
            canonical_levels=(),
            min_decoding_accuracy=0.95,
            build_samples=_identity_samples,
        ),
    )
