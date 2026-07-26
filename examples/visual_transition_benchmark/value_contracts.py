"""Typed value decoding and contracts for stage 2 (value-aware ZeroModel).

Stage 1 asked "did this named component change?" Stage 2 asks "did it change
to the *correct value*?" Every value here is **decoded from pixels**, using
only the same P4A/P18A machinery stage 1 already built (per-field
``before_mean``/``after_mean`` from ``TransitionEvidenceVPMDTO``) -- nothing
here reads ``ArcadeState``/``tank_x``/``target_x``/``cooldown`` directly. A
decoded value is an *observation*, not privileged ground truth: it is exactly
what a person looking at the two frames could read off, expressed as a number
instead of a component label.

Decodable, per this environment's fixed rendering contract
(``zeromodel.video.arcade_policy.rendering``):
  - tank / alien column index: which game column (0..width-1) has the
    highest field intensity in that band.
  - alien "alive": whether any column in the alien band shows signal above a
    small noise floor.
  - cooldown level: the cooldown corner's intensity classified against the
    two canonical constants the renderer uses (``COOLDOWN_READY_VALUE`` = 40,
    ``COOLDOWN_BLOCKED_VALUE`` = 160); anything else is "out_of_domain".

Field resolution note: stage 1's field schema (``zeromodel_adapter.FIELD_SCHEMA``,
4x1px tiles) is coarse enough to dilute an exact-value read -- the cooldown
indicator is only 2px wide inside a 4px-wide tile, so a tile's raw mean mixes
2 real cooldown pixels with 2 always-zero background pixels, corrupting the
decoded intensity by 2x. Presence/absence detection (stage 1) is insensitive
to this because it only compares a tile's before/after *delta*, and dilution
cancels out of a delta. Absolute-value decoding (stage 2) is not insensitive
to it. So this module builds its own **per-pixel** P4A field schema
(``VALUE_FIELD_SCHEMA``, 1x1px tiles) purely for value decoding -- still P4A,
still ``build_transition_evidence_vpm``, just at the resolution the task
actually needs. This is the one place stage 2 needed a *finer* view of the
existing representation, not a different one.

Contracts asserted (action-conditioned only, same discipline as stage 1):
  - tank direction: sign(decoded delta_x) must match the action's sign
    (LEFT=-1, RIGHT=+1, STAY/FIRE=0).
  - tank magnitude: decoded delta_x must equal exactly -1/+1/0 (this
    environment's single-step quantum) -- catches "correct direction, wrong
    magnitude" faults that a direction-only check would miss.
  - cooldown value: this environment's cooldown is a single binary flag, so
    its post-state is *fully* determined by the action alone -- FIRE always
    ends "blocked", anything else always ends "ready". This is a strictly
    stronger, still non-privileged claim than stage 1's "cooldown changes".
  - relation: a legitimate alien substitution (identity or alive-flag change)
    can only happen alongside a FIRE, so it must coincide with
    cooldown.after == "blocked". Cross-field, evaluated regardless of
    per-field verdicts.

Known, intentional limitation carried forward from stage 1: no contract here
can name the *correct* target column/identity -- that requires the hidden
alien queue. See value_adapter.py / README for what remains unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from zeromodel.perception.fields import VPMFieldSchemaDTO, build_grid_field_schema
from zeromodel.perception.representation import encode_source_array
from zeromodel.perception.transition_evidence import (
    TransitionEvidenceVPMDTO,
    build_transition_evidence_vpm,
)
from zeromodel.video.arcade_policy.rendering import (
    CELL_PIXELS,
    COOLDOWN_BLOCKED_VALUE,
    COOLDOWN_READY_VALUE,
)

from visual_transition_benchmark import zeromodel_adapter as zm

ALIVE_THRESHOLD = 0.05  # normalized [0,1] field intensity; matches FIELD_MIN_MEAN_ABS order of magnitude
COOLDOWN_TOLERANCE = 15.0 / 255.0
COOLDOWN_LEVELS: Dict[str, float] = {
    "ready": COOLDOWN_READY_VALUE / 255.0,
    "blocked": COOLDOWN_BLOCKED_VALUE / 255.0,
}
ACTION_TANK_EXPECTED_DELTA: Dict[str, int] = {"LEFT": -1, "RIGHT": 1, "STAY": 0, "FIRE": 0}
TANK_MAGNITUDE_BOUND = 1  # this environment's single-step movement quantum


def _build_value_field_schema() -> VPMFieldSchemaDTO:
    dummy = encode_source_array(np.zeros((zm.FRAME_HEIGHT, zm.WIDTH_PX), dtype=np.uint8), zm._SPEC)
    return build_grid_field_schema(dummy, tile_width=1, tile_height=1, channel_mode="joint")


def _fine_band_for_field(y0: int, x0: int) -> str:
    """Same bands as zeromodel_adapter._band_for_field, but expressed as exact
    pixel ranges rather than tile-aligned equality checks -- the latter only
    works at zeromodel_adapter's own 4px tile width. The cooldown corner is
    2px wide (cols WIDTH_PX-3 : WIDTH_PX-1); at 1px resolution a tile-aligned
    check silently selects the wrong (background) column instead."""

    if 11 <= y0 <= 13:
        return "tank"
    if 2 <= y0 <= 4:
        return "alien"
    if 7 <= y0 <= 8 and (zm.WIDTH_PX - 3) <= x0 < (zm.WIDTH_PX - 1):
        return "cooldown"
    return "background"


VALUE_FIELD_SCHEMA: VPMFieldSchemaDTO = _build_value_field_schema()
_VALUE_FIELD_BY_ID = {field.field_id: field for field in VALUE_FIELD_SCHEMA.fields}
_VALUE_BAND_FIELD_IDS: Dict[str, Tuple[str, ...]] = {name: [] for name in ("tank", "alien", "cooldown", "background")}
for _field in VALUE_FIELD_SCHEMA.fields:
    _VALUE_BAND_FIELD_IDS[_fine_band_for_field(_field.y0, _field.x0)].append(_field.field_id)
_VALUE_BAND_FIELD_IDS = {name: tuple(ids) for name, ids in _VALUE_BAND_FIELD_IDS.items()}


def build_value_transition_evidence(
    frame_before: np.ndarray, frame_after: np.ndarray
) -> TransitionEvidenceVPMDTO:
    """Per-pixel P18A evidence -- the resolution value decoding needs."""

    before_vpm = encode_source_array(np.ascontiguousarray(frame_before, dtype=np.uint8), zm._SPEC)
    after_vpm = encode_source_array(np.ascontiguousarray(frame_after, dtype=np.uint8), zm._SPEC)
    return build_transition_evidence_vpm(
        before_vpm, after_vpm, VALUE_FIELD_SCHEMA, annotations=(), change_threshold=zm.CHANGE_THRESHOLD
    )


def _column_index(field_id: str) -> int:
    return _VALUE_FIELD_BY_ID[field_id].x0 // CELL_PIXELS


def _column_intensities(
    transition_evidence: TransitionEvidenceVPMDTO, band_name: str, which: str
) -> Dict[int, float]:
    """Mean (not max) intensity per game column across the band's rows.

    Mean matters here: the tank sprite's 5px-wide base is 1px wider than its
    own 4px column cell and bleeds one pixel into the next cell. A max
    aggregate would read that single bleed pixel as "fully lit", tying the
    bleed column with the true center column. A mean over the band's rows
    correctly reads the bleed column as mostly-dark (1 of 3 rows lit) versus
    the true column (3 of 3 rows lit).
    """

    totals: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for field_id in _VALUE_BAND_FIELD_IDS[band_name]:
        value = getattr(transition_evidence.field_evidence(field_id), which)
        column = _column_index(field_id)
        totals[column] = totals.get(column, 0.0) + value
        counts[column] = counts.get(column, 0) + 1
    return {column: totals[column] / counts[column] for column in totals}


def _decode_column(columns: Dict[int, float]) -> Optional[int]:
    if not columns:
        return None
    best_column = max(columns, key=lambda c: columns[c])
    if columns[best_column] < ALIVE_THRESHOLD:
        return None
    return best_column


def classify_cooldown_level(intensity: float) -> str:
    for name, level in COOLDOWN_LEVELS.items():
        if abs(intensity - level) <= COOLDOWN_TOLERANCE:
            return name
    return "out_of_domain"


@dataclass(frozen=True)
class TankValues:
    before_x: Optional[int]
    after_x: Optional[int]
    delta_x: Optional[int]


@dataclass(frozen=True)
class AlienValues:
    before_alive: bool
    after_alive: bool
    before_x: Optional[int]
    after_x: Optional[int]


@dataclass(frozen=True)
class CooldownValues:
    before_intensity: float
    after_intensity: float
    before_level: str
    after_level: str


@dataclass(frozen=True)
class DecodedValues:
    tank: TankValues
    alien: AlienValues
    cooldown: CooldownValues


def decode_values(transition_evidence: TransitionEvidenceVPMDTO) -> DecodedValues:
    tank_before_cols = _column_intensities(transition_evidence, "tank", "before_mean")
    tank_after_cols = _column_intensities(transition_evidence, "tank", "after_mean")
    tank_before_x = _decode_column(tank_before_cols)
    tank_after_x = _decode_column(tank_after_cols)
    delta_x = None if (tank_before_x is None or tank_after_x is None) else tank_after_x - tank_before_x
    tank = TankValues(before_x=tank_before_x, after_x=tank_after_x, delta_x=delta_x)

    alien_before_cols = _column_intensities(transition_evidence, "alien", "before_mean")
    alien_after_cols = _column_intensities(transition_evidence, "alien", "after_mean")
    alien_before_x = _decode_column(alien_before_cols)
    alien_after_x = _decode_column(alien_after_cols)
    alien = AlienValues(
        before_alive=alien_before_x is not None,
        after_alive=alien_after_x is not None,
        before_x=alien_before_x,
        after_x=alien_after_x,
    )

    cooldown_fields = _VALUE_BAND_FIELD_IDS["cooldown"]
    cooldown_before = sum(
        transition_evidence.field_evidence(fid).before_mean for fid in cooldown_fields
    ) / len(cooldown_fields)
    cooldown_after = sum(
        transition_evidence.field_evidence(fid).after_mean for fid in cooldown_fields
    ) / len(cooldown_fields)
    cooldown = CooldownValues(
        before_intensity=cooldown_before,
        after_intensity=cooldown_after,
        before_level=classify_cooldown_level(cooldown_before),
        after_level=classify_cooldown_level(cooldown_after),
    )
    return DecodedValues(tank=tank, alien=alien, cooldown=cooldown)


def _sign(value: int) -> int:
    return -1 if value < 0 else (1 if value > 0 else 0)


@dataclass(frozen=True)
class ValueContractVerdict:
    expected_delta_x: int
    tank_direction_ok: Optional[bool]
    tank_magnitude_ok: Optional[bool]
    expected_cooldown_level: str
    cooldown_value_ok: bool
    relation_violations: Tuple[str, ...]


def evaluate_contracts(action: str, values: DecodedValues) -> ValueContractVerdict:
    if action not in ACTION_TANK_EXPECTED_DELTA:
        raise ValueError(f"unsupported action: {action}")
    expected_delta = ACTION_TANK_EXPECTED_DELTA[action]

    if values.tank.delta_x is None:
        direction_ok: Optional[bool] = None
        magnitude_ok: Optional[bool] = None
    else:
        direction_ok = _sign(values.tank.delta_x) == _sign(expected_delta)
        magnitude_ok = values.tank.delta_x == expected_delta

    expected_cooldown_level = "blocked" if action == "FIRE" else "ready"
    cooldown_ok = values.cooldown.after_level == expected_cooldown_level

    violations = []
    if values.tank.delta_x is not None and abs(values.tank.delta_x) > TANK_MAGNITUDE_BOUND:
        violations.append("tank_magnitude_exceeds_single_step_bound")
    if values.cooldown.after_level == "out_of_domain":
        violations.append("cooldown_value_out_of_domain")
    alien_substituted = (
        values.alien.after_x != values.alien.before_x or values.alien.after_alive != values.alien.before_alive
    )
    if alien_substituted and values.cooldown.after_level != "blocked":
        violations.append("alien_substitution_without_cooldown_blocked")

    return ValueContractVerdict(
        expected_delta_x=expected_delta,
        tank_direction_ok=direction_ok,
        tank_magnitude_ok=magnitude_ok,
        expected_cooldown_level=expected_cooldown_level,
        cooldown_value_ok=cooldown_ok,
        relation_violations=tuple(violations),
    )
