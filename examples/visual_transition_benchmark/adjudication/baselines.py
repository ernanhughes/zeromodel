from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from visual_transition_benchmark import zeromodel_adapter as component_zm
from visual_transition_benchmark.adjudication.metrics import rate
from visual_transition_benchmark.value_contracts import (
    build_value_transition_evidence,
    decode_values,
)


def static_reader_baseline(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    items = list(rows)
    accepted = [row for row in items if row["policy_executed"]]
    wrong = [row for row in accepted if not row["exact_address"]]
    return {
        "accepted_case_count": len(accepted),
        "row_accuracy": rate(
            sum(bool(row["exact_address"]) for row in accepted), len(accepted)
        ),
        "action_accuracy": rate(
            sum(bool(row["same_action"]) for row in accepted), len(accepted)
        ),
        "accepted_wrong_row_rate": rate(len(wrong), len(accepted)),
        "action_equivalent_wrong_row_rate": rate(
            sum(bool(row["same_action"]) for row in wrong), len(wrong)
        ),
    }


def raw_pixel_baseline(
    frame_before: np.ndarray, frame_after: np.ndarray
) -> dict[str, object]:
    delta = np.abs(frame_after.astype(np.int16) - frame_before.astype(np.int16))
    return {
        "changed_pixel_count": int(np.count_nonzero(delta)),
        "global_absolute_difference": int(delta.sum()),
        "can_contradict_candidate_without_regions": False,
    }


def region_pixel_signature(
    frame_before: np.ndarray, frame_after: np.ndarray
) -> tuple[str, ...]:
    changed = []
    for name, mask in component_zm.BAND_MASKS.items():
        if np.any(frame_before[mask] != frame_after[mask]):
            changed.append(name)
    return tuple(sorted(changed))


def value_signature(
    frame_before: np.ndarray, frame_after: np.ndarray
) -> dict[str, object]:
    values = decode_values(build_value_transition_evidence(frame_before, frame_after))
    return {
        "tank_delta": values.tank.delta_x,
        "cooldown_level": values.cooldown.after_level,
        "alien_after_alive": values.alien.after_alive,
        "alien_after_x": values.alien.after_x,
    }
