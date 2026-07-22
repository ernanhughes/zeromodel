from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Optional, Tuple

from zeromodel.core.artifact import LayoutRecipe, ScoreTable, build_vpm
from zeromodel.core.policy_lookup import VPMPolicyLookup


ACTIONS: Tuple[str, ...] = ("LEFT", "RIGHT", "STAY", "FIRE")


@dataclass(frozen=True)
class ShooterConfig:
    width: int = 7
    wave: Tuple[int, ...] = (0, 6, 1, 5)
    max_steps: int = 32


class TinyArcadeShooter:
    def __init__(self, config: ShooterConfig = ShooterConfig()) -> None:
        if config.width <= 1:
            raise ValueError("width must be greater than one")
        for column in config.wave:
            if not (0 <= int(column) < config.width):
                raise ValueError("wave columns must be inside the screen")
        self.config = config
        self.tank_x = config.width // 2
        self.aliens = list(int(column) for column in config.wave)
        self.cooldown = 0
        self.steps = 0
        self.score = 0

    @property
    def done(self) -> bool:
        return self.cleared or self.steps >= self.config.max_steps

    @property
    def cleared(self) -> bool:
        return len(self.aliens) == 0

    @property
    def target_x(self) -> Optional[int]:
        return self.aliens[0] if self.aliens else None

    def row_id(self) -> str:
        return state_row_id(self.tank_x, self.target_x, self.cooldown)

    def snapshot(self) -> dict[str, Any]:
        return {
            "step": self.steps,
            "tank_x": self.tank_x,
            "target_x": self.target_x,
            "cooldown": self.cooldown,
            "remaining_aliens": list(self.aliens),
            "score": self.score,
        }

    def step(self, action: str) -> None:
        if self.done:
            return
        action = str(action).upper()
        if action not in ACTIONS:
            raise ValueError("unknown action: %s" % action)

        fired = False
        if action == "LEFT":
            self.tank_x = max(0, self.tank_x - 1)
        elif action == "RIGHT":
            self.tank_x = min(self.config.width - 1, self.tank_x + 1)
        elif action == "FIRE":
            fired = True
            if (
                self.cooldown == 0
                and self.target_x is not None
                and self.tank_x == self.target_x
            ):
                self.aliens.pop(0)
                self.score += 1

        if fired and self.cooldown == 0:
            self.cooldown = 1
        elif not fired and self.cooldown > 0:
            self.cooldown -= 1
        self.steps += 1


def state_row_id(tank_x: int, target_x: Optional[int], cooldown: int) -> str:
    target = "none" if target_x is None else str(int(target_x))
    return "tank=%s|target=%s|cooldown=%s" % (int(tank_x), target, int(cooldown))


def parse_state_row_id(row_id: str) -> tuple[int, Optional[int], int]:
    values = {}
    for part in str(row_id).split("|"):
        key, value = part.split("=", 1)
        values[key] = value
    target = None if values["target"] == "none" else int(values["target"])
    return int(values["tank"]), target, int(values["cooldown"])


def _action_values(
    tank_x: int, target_x: Optional[int], cooldown: int
) -> tuple[float, ...]:
    if target_x is None:
        return (0.0, 0.0, 1.0, 0.0)
    if cooldown == 0 and tank_x == target_x:
        return (0.0, 0.0, 0.0, 1.0)
    if tank_x > target_x:
        return (1.0, 0.0, 0.1, 0.0)
    if tank_x < target_x:
        return (0.0, 1.0, 0.1, 0.0)
    return (0.0, 0.0, 1.0, 0.0)


def compile_policy_artifact(config: ShooterConfig = ShooterConfig()):
    row_ids: list[str] = []
    values: list[tuple[float, ...]] = []
    targets: tuple[Optional[int], ...] = (None,) + tuple(range(config.width))
    for tank_x in range(config.width):
        for target_x in targets:
            for cooldown in (0, 1):
                row_ids.append(state_row_id(tank_x, target_x, cooldown))
                values.append(_action_values(tank_x, target_x, cooldown))

    table = ScoreTable(
        values=values,
        row_ids=row_ids,
        metric_ids=ACTIONS,
        metadata={
            "kind": "arcade_shooter_policy",
            "world": "tiny_arcade_shooter",
            "addressing": "tank_x,target_x,cooldown",
            "slogan": "signs_not_directions",
        },
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "arcade-shooter-policy-source-order",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(
        table,
        recipe,
        provenance={
            "kind": "compiled_policy",
            "consumer": "VPMPolicyLookup",
            "compile_time_intelligence": "hand_scored_closed_world_policy",
        },
    )


def run_policy_episode(config: ShooterConfig = ShooterConfig()) -> dict[str, Any]:
    artifact = compile_policy_artifact(config)
    reader = VPMPolicyLookup(artifact, action_metric_ids=ACTIONS)
    game = TinyArcadeShooter(config)
    trace: list[dict[str, Any]] = []
    while not game.done:
        before = game.snapshot()
        row_id = game.row_id()
        decision = reader.read(row_id)
        game.step(decision.action)
        trace.append(
            {
                **before,
                "row_id": row_id,
                "action": decision.action,
                "artifact_id": decision.artifact_id,
                "source_row_index": decision.source_row_index,
                "source_metric_index": decision.source_metric_index,
                "view_row": decision.view_row,
                "view_column": decision.view_column,
            }
        )
    return {
        "artifact_id": artifact.artifact_id,
        "score": game.score,
        "cleared": game.cleared,
        "steps": game.steps,
        "trace": trace,
    }


def run_random_episode(
    config: ShooterConfig = ShooterConfig(), *, seed: int = 0
) -> dict[str, Any]:
    rng = random.Random(seed)
    game = TinyArcadeShooter(config)
    trace: list[dict[str, Any]] = []
    while not game.done:
        before = game.snapshot()
        action = rng.choice(ACTIONS)
        game.step(action)
        trace.append({**before, "action": action})
    return {
        "score": game.score,
        "cleared": game.cleared,
        "steps": game.steps,
        "trace": trace,
    }


def random_baseline_average(
    config: ShooterConfig = ShooterConfig(), *, seeds: int = 10
) -> float:
    return sum(
        run_random_episode(config, seed=seed)["score"] for seed in range(seeds)
    ) / float(seeds)
