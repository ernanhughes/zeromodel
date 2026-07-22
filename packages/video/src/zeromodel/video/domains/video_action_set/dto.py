from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import cast

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import (
    JsonValue,
    canonical_json_text,
    canonical_sha256,
)
from zeromodel.video.domains.video_action_set.contracts import (
    BENCHMARK_VERSION,
    EPISODE_PLAN_VERSION,
    GENERATOR_VERSION,
    REACHABILITY_TILE_DIGEST,
    REACHABILITY_TILE_VERSION,
    SEED_DERIVATION_VERSION,
)

EPISODE_PLAN_KEYS = (
    "version",
    "seed_derivation_version",
    "episode_id",
    "split",
    "ordinal",
    "family_label",
    "family_ordinal",
    "episode_family",
    "episode_disposition",
    "denominator_class",
    "final_observation_provenance",
    "mutation_kind",
    "source_row_id",
    "secondary_row_id",
    "family_contract",
    "family_intervention",
    "derived_seed_identity",
    "episode_seed",
    "frame_count",
    "seed_lineage",
    "frame_plans",
    "plan_digest",
)
EPISODE_COUNT_KEYS = (
    "valid",
    "frame_invalid",
    "temporal_negative",
    "information_control",
)
SEALED_SPLIT_PLAN_KEYS = (
    "version",
    "seed_derivation_version",
    "split",
    "plan_only",
    "materialization_prohibited",
    "episode_counts",
    "frame_count",
    "sealed_episode_ids",
    "episodes",
    "seed_commitment",
    "sealed_plan_digest",
)
SPLITS = frozenset({"development", "calibration", "selection", "final"})
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
FINAL_OBSERVATION_PROVENANCE = {
    "materialization_status": "prospective_materialization_prohibited",
    "observation_payload_included": False,
    "provenance": "sealed_plan_only",
}
MATERIALIZED_OBSERVATION_PROVENANCE = {
    "materialization_status": "materialized",
    "observation_payload_included": True,
    "provenance": "in_memory_generation",
}


def _require_exact_keys(
    payload: Mapping[str, object],
    keys: tuple[str, ...],
    message: str,
) -> None:
    if set(payload) != set(keys):
        raise VPMValidationError(message)


def _require_mapping(value: object, message: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise VPMValidationError(message)
    return cast(Mapping[str, object], value)


def _require_sequence(value: object, message: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise VPMValidationError(message)
    return cast(Sequence[object], value)


def _require_str(payload: Mapping[str, object], key: str, message: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise VPMValidationError(message)
    return value


def _require_optional_str(
    payload: Mapping[str, object],
    key: str,
    message: str,
) -> str | None:
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise VPMValidationError(message)
    return value


def _require_int(payload: Mapping[str, object], key: str, message: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise VPMValidationError(message)
    return value


def _require_bool(payload: Mapping[str, object], key: str, message: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise VPMValidationError(message)
    return value


def _is_sha256_identity(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _seed_int_from_digest(digest: str) -> int:
    if not _is_sha256_identity(digest):
        raise VPMValidationError("episode plan root seed lineage mismatch")
    return int(digest.removeprefix("sha256:")[:16], 16)


def _canonical_from_payload(
    payload: Mapping[str, object],
    key: str,
    message: str,
) -> CanonicalJsonDTO:
    try:
        return CanonicalJsonDTO.from_value(payload[key])
    except VPMValidationError as exc:
        raise VPMValidationError(message) from exc


def _frame_plan_tuple(value: object) -> tuple[CanonicalJsonDTO, ...]:
    frames = _require_sequence(value, "episode plan frame indexes are not contiguous")
    return tuple(CanonicalJsonDTO.from_value(frame) for frame in frames)


def _lineage_mapping(lineage: CanonicalJsonDTO) -> dict[str, JsonValue]:
    value = lineage.to_value()
    if not isinstance(value, dict):
        raise VPMValidationError("episode plan root seed lineage mismatch")
    return value


def _seed_node_mapping(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise VPMValidationError("episode plan root seed lineage mismatch")
    return value


def _validate_seed_node(
    node: dict[str, JsonValue],
    *,
    split: str,
    ordinal: int,
) -> str:
    if node.get("version") != SEED_DERIVATION_VERSION:
        raise VPMValidationError("episode plan root seed lineage mismatch")
    if node.get("split") != split or node.get("episode_ordinal") != ordinal:
        raise VPMValidationError("episode plan root seed lineage mismatch")
    root_seed = node.get("root_seed_digest")
    seed_digest = node.get("seed_digest")
    seed_int64 = node.get("seed_int64")
    if not _is_sha256_identity(root_seed) or not _is_sha256_identity(seed_digest):
        raise VPMValidationError("episode plan root seed lineage mismatch")
    if not isinstance(seed_int64, int) or isinstance(seed_int64, bool):
        raise VPMValidationError("episode plan root seed lineage mismatch")
    payload = {
        key: value
        for key, value in node.items()
        if key not in {"seed_digest", "seed_int64"}
    }
    if canonical_sha256(payload) != seed_digest:
        raise VPMValidationError("episode plan root seed lineage mismatch")
    if _seed_int_from_digest(seed_digest) != seed_int64:
        raise VPMValidationError("episode plan root seed lineage mismatch")
    return str(root_seed)


def _validate_count(value: int) -> int:
    if value < 0:
        raise VPMValidationError("episode counts cannot be negative")
    return value


def _episode_counts_from_episodes(
    episodes: Sequence[EpisodePlanDTO],
) -> EpisodeCountsDTO:
    counts = dict.fromkeys(EPISODE_COUNT_KEYS, 0)
    for episode in episodes:
        if episode.family_label not in counts:
            raise VPMValidationError("sealed plan episode counts mismatch")
        counts[episode.family_label] += 1
    return EpisodeCountsDTO(
        valid=counts["valid"],
        frame_invalid=counts["frame_invalid"],
        temporal_negative=counts["temporal_negative"],
        information_control=counts["information_control"],
    )


def _episode_ids_from_episodes(
    episodes: Sequence[EpisodePlanDTO],
) -> EpisodeIdsByFamilyDTO:
    grouped: dict[str, list[str]] = {key: [] for key in EPISODE_COUNT_KEYS}
    for episode in episodes:
        if episode.family_label not in grouped:
            raise VPMValidationError("sealed plan episode id manifest mismatch")
        grouped[episode.family_label].append(episode.episode_id)
    return EpisodeIdsByFamilyDTO(
        valid=tuple(grouped["valid"]),
        frame_invalid=tuple(grouped["frame_invalid"]),
        temporal_negative=tuple(grouped["temporal_negative"]),
        information_control=tuple(grouped["information_control"]),
    )


def _sealed_payload_without_digest(plan: SealedSplitPlanDTO) -> dict[str, object]:
    payload = plan.to_dict()
    payload.pop("sealed_plan_digest")
    return payload


def _episode_payload_without_digest(plan: EpisodePlanDTO) -> dict[str, object]:
    payload = plan.to_dict()
    payload.pop("plan_digest")
    return payload


@dataclass(frozen=True, slots=True)
class CanonicalJsonDTO:
    canonical_text: str

    def __post_init__(self) -> None:
        try:
            parsed = json.loads(self.canonical_text)
            if canonical_json_text(parsed) != self.canonical_text:
                raise ValueError
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VPMValidationError("canonical JSON text is not canonical") from exc

    @classmethod
    def from_value(cls, value: object) -> CanonicalJsonDTO:
        try:
            return cls(canonical_json_text(value))
        except (TypeError, ValueError) as exc:
            raise VPMValidationError("value is not canonical JSON compatible") from exc

    def to_value(self) -> JsonValue:
        return cast(JsonValue, json.loads(self.canonical_text))


@dataclass(frozen=True, slots=True)
class BenchmarkIdentityDTO:
    contract_commit: str
    seed_material: str
    seed_digest: str
    policy_artifact_id: str
    parent_audit_sha: str
    parent_v3_sha: str

    def __post_init__(self) -> None:
        expected = (
            "sha256:" + hashlib.sha256(self.seed_material.encode("utf-8")).hexdigest()
        )
        if self.seed_digest != expected:
            raise VPMValidationError(
                "benchmark seed digest is inconsistent with frozen seed material"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "benchmark_version": BENCHMARK_VERSION,
            "generator_version": GENERATOR_VERSION,
            "contract_commit": self.contract_commit,
            "seed_material": self.seed_material,
            "seed_digest": self.seed_digest,
            "policy_artifact_id": self.policy_artifact_id,
            "parent_audit_sha": self.parent_audit_sha,
            "parent_v3_sha": self.parent_v3_sha,
            "reachability_tile_version": REACHABILITY_TILE_VERSION,
            "reachability_tile_digest": REACHABILITY_TILE_DIGEST,
        }


@dataclass(frozen=True, slots=True)
class EpisodePlanDTO:
    version: str
    seed_derivation_version: str
    episode_id: str
    split: str
    ordinal: int
    family_label: str
    family_ordinal: int
    episode_family: str
    episode_disposition: str
    denominator_class: str
    final_observation_provenance: CanonicalJsonDTO
    mutation_kind: str | None
    source_row_id: str
    secondary_row_id: str | None
    family_contract: CanonicalJsonDTO
    family_intervention: CanonicalJsonDTO
    derived_seed_identity: str
    episode_seed: int
    frame_count: int
    seed_lineage: CanonicalJsonDTO
    frame_plans: tuple[CanonicalJsonDTO, ...]
    plan_digest: str

    def __post_init__(self) -> None:
        if self.version != EPISODE_PLAN_VERSION:
            raise VPMValidationError("unsupported episode plan version")
        if self.seed_derivation_version != SEED_DERIVATION_VERSION:
            raise VPMValidationError("unsupported seed derivation version")
        if self.split not in SPLITS:
            raise VPMValidationError("unsupported episode plan split")
        if not self.episode_id.startswith(f"{self.split}:"):
            raise VPMValidationError("episode id does not match split")
        if self.ordinal < 0 or self.family_ordinal < 0:
            raise VPMValidationError("episode plan ordinal cannot be negative")
        if self.episode_family != self.family_label:
            raise VPMValidationError("episode plan family mismatch")
        if self.frame_count != len(self.frame_plans):
            raise VPMValidationError("episode plan frame count mismatch")
        if self.frame_count < 0:
            raise VPMValidationError("episode plan frame count mismatch")
        self._validate_frame_indexes()
        if not _is_sha256_identity(self.derived_seed_identity):
            raise VPMValidationError("episode plan derived seed identity is not sha256")
        self._validate_seed_lineage()
        if canonical_sha256(_episode_payload_without_digest(self)) != self.plan_digest:
            raise VPMValidationError("episode plan digest mismatch")
        self._validate_final_observation_provenance()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EpisodePlanDTO:
        _require_exact_keys(
            payload,
            EPISODE_PLAN_KEYS,
            "episode plan payload keys mismatch",
        )
        return cls(
            version=_require_str(
                payload, "version", "unsupported episode plan version"
            ),
            seed_derivation_version=_require_str(
                payload,
                "seed_derivation_version",
                "unsupported seed derivation version",
            ),
            episode_id=_require_str(
                payload, "episode_id", "episode id does not match split"
            ),
            split=_require_str(payload, "split", "unsupported episode plan split"),
            ordinal=_require_int(
                payload, "ordinal", "episode plan ordinal cannot be negative"
            ),
            family_label=_require_str(
                payload, "family_label", "episode plan family mismatch"
            ),
            family_ordinal=_require_int(
                payload, "family_ordinal", "episode plan ordinal cannot be negative"
            ),
            episode_family=_require_str(
                payload, "episode_family", "episode plan family mismatch"
            ),
            episode_disposition=_require_str(
                payload, "episode_disposition", "episode plan payload keys mismatch"
            ),
            denominator_class=_require_str(
                payload, "denominator_class", "episode plan payload keys mismatch"
            ),
            final_observation_provenance=_canonical_from_payload(
                payload,
                "final_observation_provenance",
                "episode plan final observation provenance mismatch",
            ),
            mutation_kind=_require_optional_str(
                payload, "mutation_kind", "episode plan payload keys mismatch"
            ),
            source_row_id=_require_str(
                payload, "source_row_id", "episode plan payload keys mismatch"
            ),
            secondary_row_id=_require_optional_str(
                payload, "secondary_row_id", "episode plan payload keys mismatch"
            ),
            family_contract=_canonical_from_payload(
                payload, "family_contract", "episode plan payload keys mismatch"
            ),
            family_intervention=_canonical_from_payload(
                payload, "family_intervention", "episode plan payload keys mismatch"
            ),
            derived_seed_identity=_require_str(
                payload,
                "derived_seed_identity",
                "episode plan derived seed identity is not sha256",
            ),
            episode_seed=_require_int(
                payload, "episode_seed", "episode plan root seed lineage mismatch"
            ),
            frame_count=_require_int(
                payload, "frame_count", "episode plan frame count mismatch"
            ),
            seed_lineage=_canonical_from_payload(
                payload, "seed_lineage", "episode plan root seed lineage mismatch"
            ),
            frame_plans=_frame_plan_tuple(payload["frame_plans"]),
            plan_digest=_require_str(
                payload, "plan_digest", "episode plan digest mismatch"
            ),
        )

    @property
    def benchmark_seed_digest(self) -> str:
        root: str | None = None
        for value in _lineage_mapping(self.seed_lineage).values():
            seed_root = _validate_seed_node(
                _seed_node_mapping(value),
                split=self.split,
                ordinal=self.ordinal,
            )
            if root is None:
                root = seed_root
            elif root != seed_root:
                raise VPMValidationError("episode plan root seed lineage mismatch")
        if root is None:
            raise VPMValidationError("episode plan root seed lineage mismatch")
        return root

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "seed_derivation_version": self.seed_derivation_version,
            "episode_id": self.episode_id,
            "split": self.split,
            "ordinal": self.ordinal,
            "family_label": self.family_label,
            "family_ordinal": self.family_ordinal,
            "episode_family": self.episode_family,
            "episode_disposition": self.episode_disposition,
            "denominator_class": self.denominator_class,
            "final_observation_provenance": self.final_observation_provenance.to_value(),
            "mutation_kind": self.mutation_kind,
            "source_row_id": self.source_row_id,
            "secondary_row_id": self.secondary_row_id,
            "family_contract": self.family_contract.to_value(),
            "family_intervention": self.family_intervention.to_value(),
            "derived_seed_identity": self.derived_seed_identity,
            "episode_seed": self.episode_seed,
            "frame_count": self.frame_count,
            "seed_lineage": self.seed_lineage.to_value(),
            "frame_plans": [frame.to_value() for frame in self.frame_plans],
            "plan_digest": self.plan_digest,
        }

    def _validate_frame_indexes(self) -> None:
        for expected_index, frame in enumerate(self.frame_plans):
            value = frame.to_value()
            if (
                not isinstance(value, dict)
                or value.get("frame_index") != expected_index
            ):
                raise VPMValidationError(
                    "episode plan frame indexes are not contiguous"
                )

    def _validate_seed_lineage(self) -> None:
        root = self.benchmark_seed_digest
        concrete_seed = _seed_node_mapping(
            _lineage_mapping(self.seed_lineage).get("concrete_episode_seed")
        )
        if concrete_seed.get("seed_digest") != self.derived_seed_identity:
            raise VPMValidationError("episode plan root seed lineage mismatch")
        if concrete_seed.get("seed_int64") != self.episode_seed:
            raise VPMValidationError("episode plan root seed lineage mismatch")
        if not _is_sha256_identity(root):
            raise VPMValidationError("episode plan root seed lineage mismatch")

    def _validate_final_observation_provenance(self) -> None:
        expected = (
            FINAL_OBSERVATION_PROVENANCE
            if self.split == "final"
            else MATERIALIZED_OBSERVATION_PROVENANCE
        )
        if self.final_observation_provenance.to_value() != expected:
            raise VPMValidationError(
                "episode plan final observation provenance mismatch"
            )


@dataclass(frozen=True, slots=True)
class EpisodeCountsDTO:
    valid: int
    frame_invalid: int
    temporal_negative: int
    information_control: int

    def __post_init__(self) -> None:
        _validate_count(self.valid)
        _validate_count(self.frame_invalid)
        _validate_count(self.temporal_negative)
        _validate_count(self.information_control)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EpisodeCountsDTO:
        _require_exact_keys(payload, EPISODE_COUNT_KEYS, "episode counts keys mismatch")
        return cls(
            valid=_validate_count(
                _require_int(payload, "valid", "episode counts keys mismatch")
            ),
            frame_invalid=_validate_count(
                _require_int(payload, "frame_invalid", "episode counts keys mismatch")
            ),
            temporal_negative=_validate_count(
                _require_int(
                    payload,
                    "temporal_negative",
                    "episode counts keys mismatch",
                )
            ),
            information_control=_validate_count(
                _require_int(
                    payload,
                    "information_control",
                    "episode counts keys mismatch",
                )
            ),
        )

    @property
    def total(self) -> int:
        return (
            self.valid
            + self.frame_invalid
            + self.temporal_negative
            + self.information_control
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "valid": self.valid,
            "frame_invalid": self.frame_invalid,
            "temporal_negative": self.temporal_negative,
            "information_control": self.information_control,
        }


@dataclass(frozen=True, slots=True)
class EpisodeIdsByFamilyDTO:
    valid: tuple[str, ...]
    frame_invalid: tuple[str, ...]
    temporal_negative: tuple[str, ...]
    information_control: tuple[str, ...]

    def __post_init__(self) -> None:
        episode_ids = [
            *self.valid,
            *self.frame_invalid,
            *self.temporal_negative,
            *self.information_control,
        ]
        if any(not isinstance(episode_id, str) for episode_id in episode_ids):
            raise VPMValidationError("sealed plan episode id manifest mismatch")
        if len(set(episode_ids)) != len(episode_ids):
            raise VPMValidationError("sealed plan contains duplicate episode ids")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EpisodeIdsByFamilyDTO:
        _require_exact_keys(
            payload,
            EPISODE_COUNT_KEYS,
            "sealed plan episode id manifest mismatch",
        )
        return cls(
            valid=_string_tuple(payload["valid"]),
            frame_invalid=_string_tuple(payload["frame_invalid"]),
            temporal_negative=_string_tuple(payload["temporal_negative"]),
            information_control=_string_tuple(payload["information_control"]),
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "valid": list(self.valid),
            "frame_invalid": list(self.frame_invalid),
            "temporal_negative": list(self.temporal_negative),
            "information_control": list(self.information_control),
        }


def _string_tuple(value: object) -> tuple[str, ...]:
    items = _require_sequence(value, "sealed plan episode id manifest mismatch")
    if any(not isinstance(item, str) for item in items):
        raise VPMValidationError("sealed plan episode id manifest mismatch")
    return tuple(cast(Sequence[str], items))


@dataclass(frozen=True, slots=True)
class SealedSplitPlanDTO:
    version: str
    seed_derivation_version: str
    split: str
    plan_only: bool
    materialization_prohibited: bool
    episode_counts: EpisodeCountsDTO
    frame_count: int
    sealed_episode_ids: EpisodeIdsByFamilyDTO
    episodes: tuple[EpisodePlanDTO, ...]
    seed_commitment: str
    sealed_plan_digest: str

    def __post_init__(self) -> None:
        if self.version != EPISODE_PLAN_VERSION:
            raise VPMValidationError("unsupported episode plan version")
        if self.seed_derivation_version != SEED_DERIVATION_VERSION:
            raise VPMValidationError("unsupported seed derivation version")
        if self.split != "final":
            raise VPMValidationError("sealed plan must describe the final split")
        if self.plan_only is not True or self.materialization_prohibited is not True:
            raise VPMValidationError("sealed plan permits materialization")
        if not _is_sha256_identity(self.seed_commitment):
            raise VPMValidationError("sealed plan seed commitment mismatch")
        self._validate_episodes()
        if (
            canonical_sha256(_sealed_payload_without_digest(self))
            != self.sealed_plan_digest
        ):
            raise VPMValidationError("sealed plan digest mismatch")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SealedSplitPlanDTO:
        _require_exact_keys(
            payload,
            SEALED_SPLIT_PLAN_KEYS,
            "sealed split plan payload keys mismatch",
        )
        episodes = tuple(
            EpisodePlanDTO.from_dict(
                _require_mapping(item, "sealed plan digest mismatch")
            )
            for item in _require_sequence(
                payload["episodes"], "sealed plan digest mismatch"
            )
        )
        return cls(
            version=_require_str(
                payload, "version", "unsupported episode plan version"
            ),
            seed_derivation_version=_require_str(
                payload,
                "seed_derivation_version",
                "unsupported seed derivation version",
            ),
            split=_require_str(
                payload, "split", "sealed plan must describe the final split"
            ),
            plan_only=_require_bool(
                payload, "plan_only", "sealed plan permits materialization"
            ),
            materialization_prohibited=_require_bool(
                payload,
                "materialization_prohibited",
                "sealed plan permits materialization",
            ),
            episode_counts=EpisodeCountsDTO.from_dict(
                _require_mapping(
                    payload["episode_counts"], "sealed plan episode counts mismatch"
                )
            ),
            frame_count=_require_int(
                payload, "frame_count", "sealed plan frame count mismatch"
            ),
            sealed_episode_ids=EpisodeIdsByFamilyDTO.from_dict(
                _require_mapping(
                    payload["sealed_episode_ids"],
                    "sealed plan episode id manifest mismatch",
                )
            ),
            episodes=episodes,
            seed_commitment=_require_str(
                payload, "seed_commitment", "sealed plan seed commitment mismatch"
            ),
            sealed_plan_digest=_require_str(
                payload, "sealed_plan_digest", "sealed plan digest mismatch"
            ),
        )

    @classmethod
    def build_final(
        cls,
        *,
        episodes: Sequence[EpisodePlanDTO],
        seed_commitment: str,
    ) -> SealedSplitPlanDTO:
        episode_tuple = tuple(episodes)
        payload = {
            "version": EPISODE_PLAN_VERSION,
            "seed_derivation_version": SEED_DERIVATION_VERSION,
            "split": "final",
            "plan_only": True,
            "materialization_prohibited": True,
            "episode_counts": _episode_counts_from_episodes(episode_tuple).to_dict(),
            "frame_count": sum(episode.frame_count for episode in episode_tuple),
            "sealed_episode_ids": _episode_ids_from_episodes(episode_tuple).to_dict(),
            "episodes": [episode.to_dict() for episode in episode_tuple],
            "seed_commitment": seed_commitment,
        }
        return cls.from_dict(
            payload | {"sealed_plan_digest": canonical_sha256(payload)}
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "seed_derivation_version": self.seed_derivation_version,
            "split": self.split,
            "plan_only": self.plan_only,
            "materialization_prohibited": self.materialization_prohibited,
            "episode_counts": self.episode_counts.to_dict(),
            "frame_count": self.frame_count,
            "sealed_episode_ids": self.sealed_episode_ids.to_dict(),
            "episodes": [episode.to_dict() for episode in self.episodes],
            "seed_commitment": self.seed_commitment,
            "sealed_plan_digest": self.sealed_plan_digest,
        }

    def _validate_episodes(self) -> None:
        episode_ids = [episode.episode_id for episode in self.episodes]
        if len(set(episode_ids)) != len(episode_ids):
            raise VPMValidationError("sealed plan contains duplicate episode ids")
        for episode in self.episodes:
            if episode.split != "final":
                raise VPMValidationError("sealed plan contains a non-final episode")
            if episode.benchmark_seed_digest != self.seed_commitment:
                raise VPMValidationError("sealed plan seed commitment mismatch")
        if self.episode_counts != _episode_counts_from_episodes(self.episodes):
            raise VPMValidationError("sealed plan episode counts mismatch")
        if self.sealed_episode_ids != _episode_ids_from_episodes(self.episodes):
            raise VPMValidationError("sealed plan episode id manifest mismatch")
        if self.frame_count != sum(episode.frame_count for episode in self.episodes):
            raise VPMValidationError("sealed plan frame count mismatch")


__all__ = [
    "BenchmarkIdentityDTO",
    "CanonicalJsonDTO",
    "EpisodeCountsDTO",
    "EpisodeIdsByFamilyDTO",
    "EpisodePlanDTO",
    "SealedSplitPlanDTO",
]
