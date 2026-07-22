from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import inspect, select, text

from zeromodel.core.artifact import VPMValidationError
from zeromodel.core.matrix_blob import MatrixBlob
from zeromodel.persistence.sqlalchemy import (
    SqlAlchemyVideoActionSetStore,
    build_sqlite_runtime,
    create_database_engine,
    create_schema,
    create_session_factory,
    sqlite_database_url,
)
from zeromodel.persistence.sqlalchemy.db.orm.video_action_set import (
    MatrixBlobORM,
    SealedSplitPlanORM,
)
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    BENCHMARK_VERSION,
    GENERATOR_VERSION,
    OBSERVATION_OPERATION_CHAIN_VERSION,
)
from zeromodel.video.domains.video_action_set.dto import (
    BenchmarkIdentityDTO,
    EpisodePlanDTO,
    SealedSplitPlanDTO,
)
from zeromodel.video.domains.video_action_set.episode_planning import make_episode_plan
from zeromodel.video.domains.video_action_set.observation_dto import ObservationDTO
from zeromodel.video.domains.video_action_set.store import (
    EPISODE_PLAN_CONFLICT_MESSAGE,
    MATRIX_BLOB_CONFLICT_MESSAGE,
    UNKNOWN_BENCHMARK_IDENTITY_MESSAGE,
)
from zeromodel.video.stores.video_action_set_memory import InMemoryVideoActionSetStore

REPO_ROOT = Path(__file__).resolve().parents[3]


def _identity(seed_material: str = "sqlalchemy-isolation-seed") -> BenchmarkIdentityDTO:
    seed_digest = "sha256:" + hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    return BenchmarkIdentityDTO(
        contract_commit="f251ea80d028f73fdd843fcf0ca22b4173b72b08",
        seed_material=seed_material,
        seed_digest=seed_digest,
        policy_artifact_id="artifact:"
        + canonical_sha256({"policy": seed_material})[-64:],
        parent_audit_sha="c827a36b6498990f2d7eb10e8ec4fc6a584fb502",
        parent_v3_sha="0b6a4698633e55e99326488d4dbf77b1c266c560",
    )


def _episode_plan(
    identity: BenchmarkIdentityDTO,
    *,
    ordinal: int = 0,
    split: str = "development",
    source_row_id: str = "row:left",
) -> EpisodePlanDTO:
    return EpisodePlanDTO.from_dict(
        make_episode_plan(
            identity,
            split=split,
            ordinal=ordinal,
            family_label="valid",
            family_ordinal=ordinal,
            source_row_id=source_row_id,
            row_actions={"row:left": "left", "row:right": "right"},
        )
    )


def _redigest_episode(payload: dict[str, object]) -> dict[str, object]:
    payload = dict(payload)
    payload.pop("plan_digest", None)
    return payload | {"plan_digest": canonical_sha256(payload)}


def _operation(final_digest: str | None) -> dict[str, object]:
    parameters = {"event_type": "gap_unknown" if final_digest is None else "frame"}
    payload = {
        "index": 0,
        "operation": "emit_observation",
        "operation_version": OBSERVATION_OPERATION_CHAIN_VERSION,
        "input_digests": [final_digest],
        "parameters": parameters,
        "parameter_digest": canonical_sha256(parameters),
        "output_digest": final_digest,
    }
    return payload | {"operation_digest": canonical_sha256(payload)}


def _observation(plan: EpisodePlanDTO, *, sequence_number: int = 0) -> ObservationDTO:
    frame_id = f"{plan.split}:{plan.episode_id}:frame-{sequence_number:02d}"
    chain_payload = {
        "version": OBSERVATION_OPERATION_CHAIN_VERSION,
        "operations": [_operation(None)],
        "final_emitted_digest": None,
    }
    chain = chain_payload | {"operation_chain_digest": canonical_sha256(chain_payload)}
    materialized = ObservationDTO.from_record(
        {
            "benchmark_version": BENCHMARK_VERSION,
            "generator_version": GENERATOR_VERSION,
            "split": plan.split,
            "episode_id": plan.episode_id,
            "clip_id": f"{plan.split}:{plan.episode_id}:clip",
            "frame_id": frame_id,
            "sequence_number": sequence_number,
            "event_type": "gap_unknown",
            "family": "bounded_translation",
            "expected_disposition": "valid",
            "episode_family": plan.episode_family,
            "episode_disposition": plan.episode_disposition,
            "frame_disposition": "gap_no_payload",
            "denominator_class": plan.denominator_class,
            "expected_row": plan.source_row_id,
            "expected_action": "left",
            "actual_executed_action": None,
            "action_known": False,
            "gap_declaration": {"reason": "package-isolation-test"},
            "observation_pixel_digest": None,
            "metadata": {
                "episode_seed": plan.episode_seed,
                "seed_digest": plan.benchmark_seed_digest,
                "derived_seed_identity": plan.derived_seed_identity,
                "episode_plan_digest": plan.plan_digest,
                "frame_seed_identity": plan.frame_plans[0].to_value()[
                    "frame_seed_identity"
                ],
                "observation_operation_chain": chain,
            },
        }
    )
    if hasattr(materialized, "observation"):
        return materialized.observation
    return materialized


def _sql_store(path: Path) -> tuple[SqlAlchemyVideoActionSetStore, object, object]:
    engine = create_database_engine(sqlite_database_url(path))
    create_schema(engine)
    session_factory = create_session_factory(engine)
    return SqlAlchemyVideoActionSetStore(session_factory), session_factory, engine


def test_import_has_narrow_public_surface_and_no_database_side_effects(
    tmp_path: Path,
) -> None:
    script = """
import json, sys
from pathlib import Path
import zeromodel.persistence.sqlalchemy as zsql
print(json.dumps({
    "exports": sorted(zsql.__all__),
    "cwd_entries": sorted(p.name for p in Path.cwd().iterdir()),
    "forbidden_loaded": [
        name for name in ("zeromodel.analysis", "zeromodel.vision", "torch",
                         "torchvision", "transformers", "PIL")
        if name in sys.modules
    ],
}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        str(REPO_ROOT / path)
        for path in (
            "packages/core/src",
            "packages/observation/src",
            "packages/video/src",
            "packages/sqlalchemy/src",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["cwd_entries"] == []
    assert payload["forbidden_loaded"] == []
    assert payload["exports"] == [
        "FINALIZATION_SCHEMA_VERSION",
        "SqlAlchemyVideoActionSetStore",
        "build_finalization_sqlite_runtime",
        "build_sqlite_runtime",
        "create_database_engine",
        "create_schema",
        "create_session_factory",
        "initialize_finalization_authority",
        "sqlite_database_url",
        "verify_finalization_authority",
    ]


def test_schema_creation_is_explicit_and_enables_sqlite_foreign_keys(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "runtime.sqlite"
    engine = create_database_engine(sqlite_database_url(database_path))
    assert inspect(engine).get_table_names() == []

    create_schema(engine)

    table_names = set(inspect(engine).get_table_names())
    assert "video_action_set_benchmark_identity" in table_names
    assert "video_action_set_episode_plan" in table_names
    assert "matrix_blob" in table_names
    assert "video_action_set_observation" in table_names
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1


def test_sql_store_matches_in_memory_store_for_identity_plans_and_filters(
    tmp_path: Path,
) -> None:
    sql_store, _session_factory, _engine = _sql_store(tmp_path / "parity.sqlite")
    memory_store = InMemoryVideoActionSetStore()
    identity = _identity()
    first = _episode_plan(identity, ordinal=0)
    second = _episode_plan(identity, ordinal=1, source_row_id="row:right")

    for store in (memory_store, sql_store):
        with pytest.raises(
            VPMValidationError, match=UNKNOWN_BENCHMARK_IDENTITY_MESSAGE
        ):
            store.save_episode_plan(first)
        assert store.save_identity(identity) == identity
        assert store.save_episode_plan(first) == first
        assert store.save_episode_plans((second,)) == (second,)
        assert store.list_episode_plans(
            benchmark_seed_digest=identity.seed_digest,
            split="development",
        ) == (first, second)

    conflict_payload = first.to_dict()
    conflict_payload["source_row_id"] = "row:changed"
    conflict = EpisodePlanDTO.from_dict(_redigest_episode(conflict_payload))
    with pytest.raises(VPMValidationError, match=EPISODE_PLAN_CONFLICT_MESSAGE):
        sql_store.save_episode_plan(conflict)


def test_matrix_blob_dedup_conflict_and_observation_operation_filters(
    tmp_path: Path,
) -> None:
    store, session_factory, _engine = _sql_store(tmp_path / "blob.sqlite")
    identity = _identity()
    plan = _episode_plan(identity)
    blob = MatrixBlob.from_array(
        [[1, 2], [3, 4]], dtype="uint8", metadata={"kind": "test"}
    )
    conflicting = MatrixBlob.from_array(
        [[4, 3], [2, 1]],
        dtype="uint8",
        metadata={"kind": "changed"},
    )
    object.__setattr__(conflicting, "blob_id", blob.blob_id)

    store.save_identity(identity)
    store.save_episode_plan(plan)
    assert store.save_matrix_blob(blob) == blob
    assert store.save_matrix_blob(blob) == blob
    with pytest.raises(VPMValidationError, match=MATRIX_BLOB_CONFLICT_MESSAGE):
        store.save_matrix_blob(conflicting)

    observation = _observation(plan)
    assert store.save_observation(observation, matrix_blob=None) == observation
    assert store.list_observations_by_operation(operation="emit_observation") == (
        observation,
    )
    assert store.list_observations(split="development", has_pixels=False) == (
        observation,
    )
    with session_factory() as session:
        assert len(session.scalars(select(MatrixBlobORM)).all()) == 1


def test_reopen_and_tamper_validation_for_persistent_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "reopen.sqlite"
    store, session_factory, _engine = _sql_store(database_path)
    identity = _identity()
    plan = _episode_plan(identity, split="final")
    sealed = SealedSplitPlanDTO.build_final(
        episodes=(plan,),
        seed_commitment=identity.seed_digest,
    )
    store.save_identity(identity)
    store.save_episode_plan(plan)
    store.save_sealed_split_plan(sealed)

    reopened_engine = create_database_engine(sqlite_database_url(database_path))
    reopened = SqlAlchemyVideoActionSetStore(create_session_factory(reopened_engine))
    assert reopened.get_identity(identity.seed_digest) == identity
    assert reopened.get_episode_plan(plan.episode_id) == plan
    assert (
        reopened.get_sealed_split_plan(
            seed_commitment=identity.seed_digest,
            split="final",
        )
        == sealed
    )

    with session_factory.begin() as session:
        row = session.get(SealedSplitPlanORM, (identity.seed_digest, "final"))
        assert row is not None
        row.sealed_plan_digest = "sha256:" + "0" * 64

    with pytest.raises(VPMValidationError, match="sealed plan digest mismatch"):
        reopened.get_sealed_split_plan(
            seed_commitment=identity.seed_digest,
            split="final",
        )


def test_build_sqlite_runtime_respects_initialize_schema_flag(tmp_path: Path) -> None:
    database_path = tmp_path / "runtime.sqlite"
    build_sqlite_runtime(sqlite_database_url(database_path), initialize_schema=False)
    assert not database_path.exists()

    runtime = build_sqlite_runtime(
        sqlite_database_url(database_path), initialize_schema=True
    )
    identity = _identity()
    runtime.video_action_set.save_identity(identity)
    assert runtime.video_action_set.get_identity(identity.seed_digest) == identity
