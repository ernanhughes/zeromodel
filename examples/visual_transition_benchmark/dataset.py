"""Deterministic transition dataset for the visual-transition debugging benchmark.

Domain: the existing ``TinyArcadeShooter`` environment
(``zeromodel.video.arcade_policy.model`` / ``.rendering``). This module does not
reimplement the environment; it drives the real ``TinyArcadeShooter.step`` method
to obtain every "true" (rule-correct) transition, and calls the real
``render_state_frame`` function to obtain every rendered frame. Fault injection
only ever substitutes the *rendered* post-state (or pokes a documented pixel), it
never changes the environment's own transition rules.

Environment-to-prompt name mapping (documented once, see README.md):
  - "tank"      -> the player cannon sprite (rows 11-13)
  - "alien"     -> the current target sprite (rows 2-4); absent when no aliens remain
  - "cooldown"  -> the fixed fire-cooldown indicator (rows 7-8, rightmost cell)
  - "background"-> every remaining pixel (always 0 in a fault-free transition)
  - "projectile": this environment resolves FIRE instantaneously (hit-or-miss in the
    same step); there is no travelling projectile sprite to render. "projectile
    advances" / "projectile hits alien" collapse into the fire_hits_* categories.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from zeromodel.video.arcade_policy.model import ACTIONS, ShooterConfig, TinyArcadeShooter
from zeromodel.video.arcade_policy.rendering import (
    CELL_PIXELS,
    FRAME_HEIGHT,
    render_state_frame,
)

CONFIG = ShooterConfig()
WIDTH = CONFIG.width
WIDTH_PX = WIDTH * CELL_PIXELS
COMPONENT_NAMES: Tuple[str, ...] = ("tank", "alien", "cooldown", "background")

ORDINARY_CATEGORIES: Tuple[str, ...] = (
    "tank_moves_left",
    "tank_moves_right",
    "tank_remains_stationary_at_boundary",
    "fire_hits_advances_target",
    "fire_hits_clears_wave",
    "fire_miss_activates_cooldown",
    "cooldown_clears",
    "no_op",
)

FAULT_CATEGORIES: Tuple[str, ...] = (
    "fire_no_projectile",
    "fire_no_cooldown",
    "alien_disappears_without_hit",
    "tank_moves_wrong_action",
    "tank_moves_wrong_direction",
    "cooldown_changes_incorrectly",
    "unrelated_alien_change",
    "background_changes_unexpectedly",
    "expected_target_remains_unchanged",
    "extra_component_changes_alongside",
)

ALL_CATEGORIES: Tuple[str, ...] = ORDINARY_CATEGORIES + FAULT_CATEGORIES

# A guaranteed-background pixel, verified against rendering.py: row 6 is strictly
# between the alien band (rows 2-4) and the cooldown band (rows 7-8); column 0 is
# outside the fixed cooldown corner (cols WIDTH_PX-3 : WIDTH_PX-1). No legitimate
# transition ever writes to this pixel.
BACKGROUND_PROBE_PIXEL: Tuple[int, int] = (6, 0)
BACKGROUND_PROBE_VALUE = 90


class DatasetError(ValueError):
    """Raised when a transition cannot be constructed deterministically."""


@dataclass(frozen=True)
class ArcadeState:
    """A plain, serializable snapshot of TinyArcadeShooter's mutable state."""

    tank_x: int
    aliens: Tuple[int, ...]
    cooldown: int

    @property
    def target_x(self) -> Optional[int]:
        return self.aliens[0] if self.aliens else None

    def as_dict(self) -> dict:
        return {
            "tank_x": self.tank_x,
            "aliens": list(self.aliens),
            "target_x": self.target_x,
            "cooldown": self.cooldown,
        }


def _game_from_state(state: ArcadeState) -> TinyArcadeShooter:
    game = TinyArcadeShooter(CONFIG)
    game.tank_x = state.tank_x
    game.aliens = list(state.aliens)
    game.cooldown = state.cooldown
    game.steps = 0
    game.score = 0
    return game


def true_next_state(state: ArcadeState, action: str) -> ArcadeState:
    """Apply the real environment rule (TinyArcadeShooter.step) and read it back."""

    game = _game_from_state(state)
    game.step(action)
    return ArcadeState(tank_x=game.tank_x, aliens=tuple(game.aliens), cooldown=game.cooldown)


def render(state: ArcadeState) -> np.ndarray:
    frame = render_state_frame(state.tank_x, state.target_x, state.cooldown, width=WIDTH)
    return np.array(frame, dtype=np.uint8, copy=True)


# --------------------------------------------------------------------------- #
# Exact, formula-derived (not detected) ground-truth component masks. These are
# privileged: they use the state directly and must never be exposed to the
# ZeroModel adapter (zeromodel_adapter.py uses only static declared row-bands).
# --------------------------------------------------------------------------- #


def tank_mask(tank_x: int) -> np.ndarray:
    mask = np.zeros((FRAME_HEIGHT, WIDTH_PX), dtype=bool)
    centre = tank_x * CELL_PIXELS + CELL_PIXELS // 2
    mask[11, centre] = True
    mask[12, centre - 1 : centre + 2] = True
    mask[13, centre - 2 : centre + 3] = True
    return mask


def alien_mask(target_x: Optional[int]) -> np.ndarray:
    mask = np.zeros((FRAME_HEIGHT, WIDTH_PX), dtype=bool)
    if target_x is None:
        return mask
    centre = target_x * CELL_PIXELS + CELL_PIXELS // 2
    mask[2:4, centre - 1 : centre + 2] = True
    mask[4, centre] = True
    return mask


def cooldown_mask() -> np.ndarray:
    mask = np.zeros((FRAME_HEIGHT, WIDTH_PX), dtype=bool)
    mask[7:9, -3:-1] = True
    return mask


def background_mask(tank_x: int, target_x: Optional[int]) -> np.ndarray:
    used = tank_mask(tank_x) | alien_mask(target_x) | cooldown_mask()
    return ~used


def component_masks(tank_x: int, target_x: Optional[int]) -> dict:
    return {
        "tank": tank_mask(tank_x),
        "alien": alien_mask(target_x),
        "cooldown": cooldown_mask(),
        "background": background_mask(tank_x, target_x),
    }


def transition_component_masks(
    before_tank_x: int,
    before_target_x: Optional[int],
    after_tank_x: int,
    after_target_x: Optional[int],
) -> dict:
    """Exact partition of the canvas across one transition.

    Background must be the complement of the *combined* tank/alien footprint
    (both timesteps), not the union of two independently-complemented masks:
    a pixel vacated by the tank when it moves belongs to "tank", never to
    "background", even though it is background in the after-state alone.
    """

    tank = tank_mask(before_tank_x) | tank_mask(after_tank_x)
    alien = alien_mask(before_target_x) | alien_mask(after_target_x)
    cooldown = cooldown_mask()
    background = ~(tank | alien | cooldown)
    return {"tank": tank, "alien": alien, "cooldown": cooldown, "background": background}


def _changed_components_from_pixels(
    frame_before: np.ndarray,
    frame_after: np.ndarray,
    before_tank_x: int,
    before_target_x: Optional[int],
    after_tank_x: int,
    after_target_x: Optional[int],
) -> Tuple[str, ...]:
    masks = transition_component_masks(before_tank_x, before_target_x, after_tank_x, after_target_x)
    changed = []
    for name in COMPONENT_NAMES:
        region = masks[name]
        if region.any() and np.any(frame_before[region] != frame_after[region]):
            changed.append(name)
    return tuple(changed)


def _changed_components_from_states(before: ArcadeState, after: ArcadeState) -> Tuple[str, ...]:
    changed = []
    if before.tank_x != after.tank_x:
        changed.append("tank")
    if before.target_x != after.target_x:
        changed.append("alien")
    if before.cooldown != after.cooldown:
        changed.append("cooldown")
    return tuple(changed)


# --------------------------------------------------------------------------- #
# Transition record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TransitionRecord:
    transition_id: str
    episode_id: str
    step_number: int
    seed: int
    action: str
    category: str
    frame_before: np.ndarray
    frame_after: np.ndarray
    state_before: dict
    state_after: dict
    component_annotations: dict  # name -> bool mask, union of before/after ground truth
    expected_changed_components: Tuple[str, ...]
    observed_changed_components: Tuple[str, ...]
    fault_type: Optional[str]
    is_faulty: bool
    notes: str


def _clip(x: int) -> int:
    return max(0, min(WIDTH - 1, x))


def _pick_aliens(rng: random.Random, count: int, *, exclude: Sequence[int] = ()) -> Tuple[int, ...]:
    pool = [c for c in range(WIDTH) if c not in exclude]
    rng.shuffle(pool)
    if len(pool) < count:
        raise DatasetError("not enough distinct columns to sample aliens")
    return tuple(pool[:count])


# --------------------------------------------------------------------------- #
# Ordinary category builders: (rng, config) -> (tank_x, aliens, cooldown, action)
# --------------------------------------------------------------------------- #


def _build_tank_moves_left(rng: random.Random):
    tank_x = rng.randint(1, WIDTH - 1)
    aliens = _pick_aliens(rng, rng.choice((1, 2)))
    return tank_x, aliens, 0, "LEFT"


def _build_tank_moves_right(rng: random.Random):
    tank_x = rng.randint(0, WIDTH - 2)
    aliens = _pick_aliens(rng, rng.choice((1, 2)))
    return tank_x, aliens, 0, "RIGHT"


def _build_tank_stationary_boundary(rng: random.Random):
    if rng.random() < 0.5:
        tank_x, action = 0, "LEFT"
    else:
        tank_x, action = WIDTH - 1, "RIGHT"
    aliens = _pick_aliens(rng, rng.choice((1, 2)))
    return tank_x, aliens, 0, action


def _build_fire_hits_advances(rng: random.Random):
    aliens = _pick_aliens(rng, 2)
    tank_x = aliens[0]
    return tank_x, aliens, 0, "FIRE"


def _build_fire_hits_clears(rng: random.Random):
    aliens = _pick_aliens(rng, 1)
    tank_x = aliens[0]
    return tank_x, aliens, 0, "FIRE"


def _build_fire_miss(rng: random.Random):
    aliens = _pick_aliens(rng, 1)
    tank_x = _clip(aliens[0] + 1) if aliens[0] < WIDTH - 1 else aliens[0] - 1
    return tank_x, aliens, 0, "FIRE"


def _build_cooldown_clears(rng: random.Random):
    tank_x = rng.randint(0, WIDTH - 1)
    aliens = _pick_aliens(rng, rng.choice((1, 2)))
    return tank_x, aliens, 1, "STAY"


def _build_no_op(rng: random.Random):
    tank_x = rng.randint(0, WIDTH - 1)
    aliens = _pick_aliens(rng, rng.choice((1, 2)))
    return tank_x, aliens, 0, "STAY"


ORDINARY_BUILDERS = {
    "tank_moves_left": _build_tank_moves_left,
    "tank_moves_right": _build_tank_moves_right,
    "tank_remains_stationary_at_boundary": _build_tank_stationary_boundary,
    "fire_hits_advances_target": _build_fire_hits_advances,
    "fire_hits_clears_wave": _build_fire_hits_clears,
    "fire_miss_activates_cooldown": _build_fire_miss,
    "cooldown_clears": _build_cooldown_clears,
    "no_op": _build_no_op,
}


# --------------------------------------------------------------------------- #
# Fault builders: return (tank_x, aliens, cooldown, action) preconditions, plus
# a fault function applied to the true post-state to build the *rendered*
# post-state. Every fault function documents exactly what it alters.
# --------------------------------------------------------------------------- #


def _fault_fire_no_projectile(true_after: ArcadeState, before: ArcadeState):
    # Suppress the alien removal only; cooldown and tank render as the true state.
    return (
        true_after.tank_x,
        before.target_x,  # keep pre-hit target: alien wrongly still present
        true_after.cooldown,
        (),
        "fire hit suppressed: alien not removed despite valid hit",
    )


def _fault_fire_no_cooldown(true_after: ArcadeState, before: ArcadeState):
    return (
        true_after.tank_x,
        true_after.target_x,
        before.cooldown,  # cooldown wrongly left at pre-fire value
        (),
        "cooldown failed to activate after FIRE",
    )


def _fault_alien_disappears_without_hit(true_after: ArcadeState, before: ArcadeState):
    remaining = true_after.aliens
    next_target = remaining[1] if len(remaining) > 1 else None
    return (
        true_after.tank_x,
        next_target,  # wrongly advance the target though no hit occurred
        true_after.cooldown,
        (),
        "alien advanced/removed with no valid hit",
    )


def _fault_tank_wrong_action(true_after: ArcadeState, before: ArcadeState):
    shifted = _clip(before.tank_x + 1) if before.tank_x < WIDTH - 1 else before.tank_x - 1
    return (
        shifted,
        true_after.target_x,
        true_after.cooldown,
        (),
        "tank shifted despite a non-movement action",
    )


def _fault_tank_wrong_direction(true_after: ArcadeState, before: ArcadeState):
    wrong = _clip(before.tank_x + 1)  # commanded LEFT but rendered as RIGHT
    return (
        wrong,
        true_after.target_x,
        true_after.cooldown,
        (),
        "tank moved opposite of the commanded direction",
    )


def _fault_cooldown_incorrect(true_after: ArcadeState, before: ArcadeState):
    return (
        true_after.tank_x,
        true_after.target_x,
        1,  # toggled to blocked though no fire occurred and it should stay ready
        (),
        "cooldown indicator toggled without a firing action",
    )


def _fault_unrelated_alien_change(true_after: ArcadeState, before: ArcadeState):
    assert before.target_x is not None
    relocated = _clip(before.target_x + 1) if before.target_x < WIDTH - 1 else before.target_x - 1
    return (
        true_after.tank_x,
        relocated,
        true_after.cooldown,
        (),
        "alien mark relocated without any underlying state change",
    )


def _fault_background_unexpected(true_after: ArcadeState, before: ArcadeState):
    row, col = BACKGROUND_PROBE_PIXEL
    return (
        true_after.tank_x,
        true_after.target_x,
        true_after.cooldown,
        ((row, col, BACKGROUND_PROBE_VALUE),),
        "background pixel flipped despite no environment rule touching it",
    )


def _fault_target_unchanged(true_after: ArcadeState, before: ArcadeState):
    return (
        before.tank_x,  # tank fails to move despite a movement action
        true_after.target_x,
        true_after.cooldown,
        (),
        "tank failed to move despite a movement action",
    )


def _fault_extra_component_change(true_after: ArcadeState, before: ArcadeState):
    row, col = BACKGROUND_PROBE_PIXEL
    return (
        true_after.tank_x,  # legitimate tank movement is preserved
        true_after.target_x,
        true_after.cooldown,
        ((row, col, BACKGROUND_PROBE_VALUE),),
        "legitimate tank movement accompanied by an unexplained background change",
    )


FAULT_FUNCTIONS: dict = {
    "fire_no_projectile": _fault_fire_no_projectile,
    "fire_no_cooldown": _fault_fire_no_cooldown,
    "alien_disappears_without_hit": _fault_alien_disappears_without_hit,
    "tank_moves_wrong_action": _fault_tank_wrong_action,
    "tank_moves_wrong_direction": _fault_tank_wrong_direction,
    "cooldown_changes_incorrectly": _fault_cooldown_incorrect,
    "unrelated_alien_change": _fault_unrelated_alien_change,
    "background_changes_unexpectedly": _fault_background_unexpected,
    "expected_target_remains_unchanged": _fault_target_unchanged,
    "extra_component_changes_alongside": _fault_extra_component_change,
}


def _build_fire_no_projectile(rng: random.Random):
    aliens = _pick_aliens(rng, 2)
    return aliens[0], aliens, 0, "FIRE"


def _build_fire_no_cooldown(rng: random.Random):
    aliens = _pick_aliens(rng, 1)
    tank_x = _clip(aliens[0] + 1) if aliens[0] < WIDTH - 1 else aliens[0] - 1
    return tank_x, aliens, 0, "FIRE"


def _build_alien_disappears_without_hit(rng: random.Random):
    aliens = _pick_aliens(rng, 2)
    tank_x = _clip(aliens[0] + 1) if aliens[0] < WIDTH - 1 else aliens[0] - 1
    return tank_x, aliens, 0, "STAY"


def _build_tank_wrong_action(rng: random.Random):
    tank_x = rng.randint(1, WIDTH - 2)
    aliens = _pick_aliens(rng, rng.choice((1, 2)))
    return tank_x, aliens, 0, "STAY"


def _build_tank_wrong_direction(rng: random.Random):
    tank_x = rng.randint(1, WIDTH - 1)
    aliens = _pick_aliens(rng, rng.choice((1, 2)))
    return tank_x, aliens, 0, "LEFT"


def _build_cooldown_incorrect(rng: random.Random):
    # Isolated to STAY so the true post-state is byte-identical to the pre-state;
    # any rendered difference is attributable only to the injected fault.
    tank_x = rng.randint(0, WIDTH - 1)
    aliens = _pick_aliens(rng, rng.choice((1, 2)))
    return tank_x, aliens, 0, "STAY"


def _build_unrelated_alien_change(rng: random.Random):
    # Isolated to STAY for the same reason as _build_cooldown_incorrect.
    aliens = _pick_aliens(rng, rng.choice((1, 2)))
    tank_x = _clip(aliens[0] + 1) if aliens[0] < WIDTH - 1 else aliens[0] - 1
    return tank_x, aliens, 0, "STAY"


def _build_background_unexpected(rng: random.Random):
    return _build_no_op(rng)


def _build_target_unchanged(rng: random.Random):
    if rng.random() < 0.5:
        tank_x, action = rng.randint(1, WIDTH - 1), "LEFT"
    else:
        tank_x, action = rng.randint(0, WIDTH - 2), "RIGHT"
    aliens = _pick_aliens(rng, rng.choice((1, 2)))
    return tank_x, aliens, 0, action


def _build_extra_component_change(rng: random.Random):
    if rng.random() < 0.5:
        return _build_tank_moves_left(rng)
    return _build_tank_moves_right(rng)


FAULT_BUILDERS = {
    "fire_no_projectile": _build_fire_no_projectile,
    "fire_no_cooldown": _build_fire_no_cooldown,
    "alien_disappears_without_hit": _build_alien_disappears_without_hit,
    "tank_moves_wrong_action": _build_tank_wrong_action,
    "tank_moves_wrong_direction": _build_tank_wrong_direction,
    "cooldown_changes_incorrectly": _build_cooldown_incorrect,
    "unrelated_alien_change": _build_unrelated_alien_change,
    "background_changes_unexpectedly": _build_background_unexpected,
    "expected_target_remains_unchanged": _build_target_unchanged,
    "extra_component_changes_alongside": _build_extra_component_change,
}


def _category_seed(seed: int, episode_id: str, category: str) -> int:
    # Deterministic, distinct per (seed, episode, category); avoids reusing the
    # exact same rng stream across categories within one episode.
    digest = 0
    for part in (str(seed), episode_id, category):
        for ch in part:
            digest = (digest * 1_000_003 + ord(ch)) % (2**32)
    return digest


def build_transition(
    *,
    episode_id: str,
    step_number: int,
    seed: int,
    category: str,
) -> TransitionRecord:
    if category not in ALL_CATEGORIES:
        raise DatasetError(f"unknown category: {category}")
    rng = random.Random(_category_seed(seed, episode_id, category))
    is_faulty = category in FAULT_CATEGORIES

    if is_faulty:
        tank_x, aliens, cooldown, action = FAULT_BUILDERS[category](rng)
    else:
        tank_x, aliens, cooldown, action = ORDINARY_BUILDERS[category](rng)

    before = ArcadeState(tank_x=tank_x, aliens=aliens, cooldown=cooldown)
    true_after = true_next_state(before, action)
    frame_before = render(before)

    if is_faulty:
        render_tank, render_target, render_cooldown, extra_edits, notes = FAULT_FUNCTIONS[category](
            true_after, before
        )
        rendered_state = ArcadeState(
            tank_x=render_tank,
            aliens=(render_target,) if render_target is not None else (),
            cooldown=render_cooldown,
        )
        frame_after = render(rendered_state)
        for row, col, value in extra_edits:
            frame_after[row, col] = value
        after_tank_x, after_target_x = render_tank, render_target
        fault_type = category
    else:
        rendered_state = true_after
        frame_after = render(true_after)
        after_tank_x, after_target_x = true_after.tank_x, true_after.target_x
        fault_type = None
        notes = "ordinary transition; rendered from the true post-state"

    expected_changed = _changed_components_from_states(before, true_after)
    observed_changed = _changed_components_from_pixels(
        frame_before,
        frame_after,
        before.tank_x,
        before.target_x,
        after_tank_x,
        after_target_x,
    )
    annotations = transition_component_masks(
        before.tank_x, before.target_x, after_tank_x, after_target_x
    )

    transition_id = f"{episode_id}-{step_number:04d}"
    return TransitionRecord(
        transition_id=transition_id,
        episode_id=episode_id,
        step_number=step_number,
        seed=seed,
        action=action,
        category=category,
        frame_before=frame_before,
        frame_after=frame_after,
        state_before=before.as_dict(),
        state_after=true_after.as_dict(),
        component_annotations=annotations,
        expected_changed_components=expected_changed,
        observed_changed_components=observed_changed,
        fault_type=fault_type,
        is_faulty=is_faulty,
        notes=notes,
    )


def generate_episode(episode_id: str, seed: int) -> Tuple[TransitionRecord, ...]:
    """One transition per declared category, in fixed category order."""

    records = []
    for step_number, category in enumerate(ALL_CATEGORIES):
        records.append(
            build_transition(
                episode_id=episode_id,
                step_number=step_number,
                seed=seed,
                category=category,
            )
        )
    return tuple(records)


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    episode_ids: Tuple[str, ...]
    records: Tuple[TransitionRecord, ...]


def generate_split(*, prefix: str, episode_count: int, seed_offset: int) -> DatasetSplit:
    episode_ids = tuple(f"{prefix}-{index:04d}" for index in range(episode_count))
    records: list = []
    for index, episode_id in enumerate(episode_ids):
        records.extend(generate_episode(episode_id, seed_offset + index))
    return DatasetSplit(name=prefix, episode_ids=episode_ids, records=tuple(records))


def assert_disjoint_splits(*splits: DatasetSplit) -> None:
    seen: dict = {}
    for split in splits:
        overlap = set(split.episode_ids) & set(seen.keys())
        if overlap:
            raise DatasetError(
                f"episode ids reused across splits ({split.name} vs "
                f"{[seen[e] for e in overlap]}): {sorted(overlap)}"
            )
        for episode_id in split.episode_ids:
            seen[episode_id] = split.name
    ids_by_split = [set(split.episode_ids) for split in splits]
    for i in range(len(ids_by_split)):
        for j in range(i + 1, len(ids_by_split)):
            if ids_by_split[i] & ids_by_split[j]:
                raise DatasetError("splits are not disjoint")
