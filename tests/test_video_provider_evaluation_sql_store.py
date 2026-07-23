from __future__ import annotations

import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError

from test_video_episode_plan_rmdto import plan_dto
from test_video_episode_plan_sql_store import sql_identity
from test_video_observation_rmdto import _pixels, sample_record
from test_video_provider_evaluation_rmdto import (
    POLICY_ARTIFACT_ID,
    exact_fixture,
    imperfect_fixture,
    sample_case,
    sample_configuration,
    sample_decision,
)

from zeromodel.core.artifact import VPMValidationError
from zeromodel.persistence.sqlalchemy.db.orm.provider_evaluation import (
    ProviderEvaluationCaseORM,
    ProviderEvaluationConfigurationORM,
    ProviderEvaluationRunORM,
)
from zeromodel.persistence.sqlalchemy.db.runtime import build_sqlite_runtime
from zeromodel.persistence.sqlalchemy.db.session import (
    create_database_engine,
    create_schema,
    create_session_factory,
)
from zeromodel.persistence.sqlalchemy.db.stores.video_action_set import (
    SqlAlchemyVideoActionSetStore,
)
from zeromodel.video.domains.video_action_set.observation_dto import ObservationDTO
from zeromodel.video.domains.video_action_set.provider_evaluation_dto import (
    CASE_OUTCOME_ACTION_CHANGING,
    CASE_OUTCOME_EXACT,
    ProviderEvaluationCaseContext,
    ProviderEvaluationCaseDTO,
    build_provider_evaluation_run,
)
from zeromodel.video.domains.video_action_set.store import (
    PROVIDER_EVALUATION_CASE_CONFLICT_MESSAGE,
    UNKNOWN_OBSERVATION_FOR_PROVIDER_EVALUATION_MESSAGE,
)
from zeromodel.video.stores.video_action_set_memory import InMemoryVideoActionSetStore


pytestmark = pytest.mark.integration


def build_store():
    engine = create_database_engine("sqlite:///:memory:")
    create_schema(engine)
    session_factory = create_session_factory(engine)
    return SqlAlchemyVideoActionSetStore(session_factory), session_factory, engine


def _save_identity_plan_and_observations(store, count: int) -> tuple[str, ...]:
    identity = sql_identity("sql-provider-evaluation-rmdto-seed")
    plan = plan_dto(identity=identity, split="development", frame_count=count)
    store.save_identity(identity)
    store.save_episode_plan(plan)
    frame_ids: list[str] = []
    for index in range(count):
        pixels = _pixels(offset=index * 7)
        record = sample_record(plan=plan, sequence_number=index, pixels=pixels)
        materialized = ObservationDTO.from_record(record)
        store.save_observation(
            materialized.observation, matrix_blob=materialized.matrix_blob
        )
        frame_ids.append(materialized.observation.frame_id)
    return tuple(frame_ids)


def test_sql_schema_creation_registers_provider_evaluation_tables() -> None:
    engine = create_database_engine("sqlite:///:memory:")
    assert "provider_evaluation_runs" not in inspect(engine).get_table_names()

    create_schema(engine)
    table_names = set(inspect(engine).get_table_names())

    assert "provider_evaluation_configurations" in table_names
    assert "provider_evaluation_runs" in table_names
    assert "provider_evaluation_cases" in table_names


def test_sql_provider_evaluation_round_trip() -> None:
    store, session_factory, _engine = build_store()
    frame_ids = _save_identity_plan_and_observations(store, 8)
    cases = imperfect_fixture(frame_ids)
    materialized = build_provider_evaluation_run(
        fixture_identity="sql-round-trip",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="unlabelled",
        cases=cases,
    )

    saved = store.save_provider_evaluation_run(materialized)
    assert saved == materialized

    read_store = SqlAlchemyVideoActionSetStore(session_factory)
    reloaded = read_store.get_materialized_provider_evaluation_run(saved.run.run_id)
    assert reloaded == materialized
    assert read_store.get_provider_evaluation_run(saved.run.run_id) == materialized.run

    with session_factory() as session:
        assert (
            session.scalar(select(func.count()).select_from(ProviderEvaluationRunORM))
            == 1
        )
        assert (
            session.scalar(select(func.count()).select_from(ProviderEvaluationCaseORM))
            == 8
        )
        assert (
            session.scalar(
                select(func.count()).select_from(ProviderEvaluationConfigurationORM)
            )
            == 1
        )


def test_sql_provider_evaluation_idempotent_save() -> None:
    store, _session_factory, _engine = build_store()
    frame_ids = _save_identity_plan_and_observations(store, 2)
    cases = exact_fixture(frame_ids)
    materialized = build_provider_evaluation_run(
        fixture_identity="sql-idempotent",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=cases,
    )
    first = store.save_provider_evaluation_run(materialized)
    second = store.save_provider_evaluation_run(materialized)
    assert first == second


def test_sql_provider_evaluation_unknown_observation_rolls_back_completely() -> None:
    store, session_factory, _engine = build_store()
    _frame_ids = _save_identity_plan_and_observations(store, 1)
    bad_case = sample_case(
        case_ordinal=0, frame_id="does-not-exist", outcome="rejected"
    )
    materialized = build_provider_evaluation_run(
        fixture_identity="sql-unknown-obs",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=[bad_case],
    )

    with pytest.raises(
        VPMValidationError, match=UNKNOWN_OBSERVATION_FOR_PROVIDER_EVALUATION_MESSAGE
    ):
        store.save_provider_evaluation_run(materialized)

    with session_factory() as session:
        assert (
            session.scalar(select(func.count()).select_from(ProviderEvaluationRunORM))
            == 0
        )
        assert (
            session.scalar(select(func.count()).select_from(ProviderEvaluationCaseORM))
            == 0
        )
        assert (
            session.scalar(
                select(func.count()).select_from(ProviderEvaluationConfigurationORM)
            )
            == 0
        )


def test_sql_provider_evaluation_duplicate_case_across_runs_rejected() -> None:
    store, _session_factory, _engine = build_store()
    frame_ids = _save_identity_plan_and_observations(store, 1)
    configuration = sample_configuration()
    shared_case = ProviderEvaluationCaseDTO.build(
        case_ordinal=0,
        frame_id=frame_ids[0],
        context=ProviderEvaluationCaseContext(
            policy_artifact_id=POLICY_ARTIFACT_ID,
            provider_configuration_id=configuration.provider_configuration_id,
        ),
        expected_state={"tank_column": 0},
        expected_decision=sample_decision("r0", "STAY"),
        accepted=True,
        predicted_state={"tank_column": 0},
        predicted_decision=sample_decision("r0", "STAY"),
    )
    run_a = build_provider_evaluation_run(
        fixture_identity="sql-run-a",
        provider_configuration=configuration,
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=[shared_case],
    )
    store.save_provider_evaluation_run(run_a)

    run_b = build_provider_evaluation_run(
        fixture_identity="sql-run-b-different",
        provider_configuration=configuration,
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=[shared_case],
    )
    with pytest.raises(
        VPMValidationError, match=PROVIDER_EVALUATION_CASE_CONFLICT_MESSAGE
    ):
        store.save_provider_evaluation_run(run_b)


def test_sqlite_foreign_keys_reject_orphan_provider_evaluation_case() -> None:
    _store, session_factory, engine = build_store()

    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1

    orphan = ProviderEvaluationCaseORM(
        case_id="sha256:" + "a" * 64,
        run_id="sha256:" + "b" * 64,
        case_ordinal=0,
        frame_id="missing-frame",
        policy_artifact_id=POLICY_ARTIFACT_ID,
        provider_configuration_id="sha256:" + "c" * 64,
        accepted=False,
        exact_state_match=False,
        action_match=False,
        outcome="rejected",
        expected_row_id="r0",
        expected_action="STAY",
        predicted_row_id=None,
        predicted_action=None,
        rejection_reason="test",
        provider_confidence=None,
        provider_latency_us=None,
        provider_raw_response_digest=None,
        provider_raw_response_text=None,
        expected_state_json="{}",
        predicted_state_json=None,
        expected_decision_trace_json="{}",
        predicted_decision_trace_json=None,
        factor_matches_json="{}",
        provider_response_metadata_json="{}",
        metadata_json="{}",
    )
    with pytest.raises(IntegrityError):
        with session_factory.begin() as session:
            session.add(orphan)


def test_sql_provider_evaluation_filters_use_sql_predicates() -> None:
    store, _session_factory, _engine = build_store()
    frame_ids = _save_identity_plan_and_observations(store, 8)
    cases = imperfect_fixture(frame_ids)
    materialized = build_provider_evaluation_run(
        fixture_identity="sql-filters",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="unlabelled",
        cases=cases,
    )
    saved = store.save_provider_evaluation_run(materialized)

    assert len(store.list_provider_evaluation_runs(provider_kind="fake")) == 1
    assert store.list_provider_evaluation_runs(provider_kind="ollama") == ()
    assert store.list_provider_evaluation_runs(
        model_digest=materialized.run.provider_configuration.model_digest
    ) == (saved.run,)

    exact_cases = store.list_provider_evaluation_cases(
        run_id=saved.run.run_id, outcome=CASE_OUTCOME_EXACT
    )
    assert len(exact_cases) == 3
    changing_cases = store.list_provider_evaluation_cases(
        run_id=saved.run.run_id, outcome=CASE_OUTCOME_ACTION_CHANGING
    )
    assert len(changing_cases) == 1
    assert [case.case_ordinal for case in changing_cases] == [7]


def test_sql_read_rejects_tampered_run_counts() -> None:
    store, session_factory, _engine = build_store()
    frame_ids = _save_identity_plan_and_observations(store, 2)
    cases = exact_fixture(frame_ids)
    materialized = build_provider_evaluation_run(
        fixture_identity="sql-tamper-run",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=cases,
    )
    store.save_provider_evaluation_run(materialized)

    with session_factory.begin() as session:
        row = session.get(ProviderEvaluationRunORM, materialized.run.run_id)
        assert row is not None
        row.exact_count = 999

    read_store = SqlAlchemyVideoActionSetStore(session_factory)
    with pytest.raises(
        VPMValidationError, match="provider evaluation run digest mismatch"
    ):
        read_store.get_provider_evaluation_run(materialized.run.run_id)


def test_sql_read_rejects_tampered_case_outcome() -> None:
    store, session_factory, _engine = build_store()
    frame_ids = _save_identity_plan_and_observations(store, 1)
    cases = exact_fixture(frame_ids)
    materialized = build_provider_evaluation_run(
        fixture_identity="sql-tamper-case",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=cases,
    )
    store.save_provider_evaluation_run(materialized)

    with session_factory.begin() as session:
        row = session.get(ProviderEvaluationCaseORM, cases[0].case_id)
        assert row is not None
        row.outcome = "action_changing"

    read_store = SqlAlchemyVideoActionSetStore(session_factory)
    with pytest.raises(VPMValidationError, match="case outcome mismatch"):
        read_store.list_provider_evaluation_cases(run_id=materialized.run.run_id)


def test_provider_evaluation_store_query_parity_between_memory_and_sql() -> None:
    sql_store, _session_factory, _engine = build_store()
    memory_store = InMemoryVideoActionSetStore()
    identity = sql_identity("sql-memory-parity-provider-evaluation-seed")
    plan = plan_dto(identity=identity, split="development", frame_count=8)

    frame_ids: list[str] = []
    for store in (sql_store, memory_store):
        store.save_identity(identity)
        store.save_episode_plan(plan)
    for index in range(8):
        pixels = _pixels(offset=index * 7)
        record = sample_record(plan=plan, sequence_number=index, pixels=pixels)
        materialized_observation = ObservationDTO.from_record(record)
        for store in (sql_store, memory_store):
            store.save_observation(
                materialized_observation.observation,
                matrix_blob=materialized_observation.matrix_blob,
            )
        frame_ids.append(materialized_observation.observation.frame_id)

    cases = imperfect_fixture(tuple(frame_ids))
    materialized_run = build_provider_evaluation_run(
        fixture_identity="parity",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="unlabelled",
        cases=cases,
    )
    for store in (sql_store, memory_store):
        store.save_provider_evaluation_run(materialized_run)

    assert (
        sql_store.list_provider_evaluation_runs()
        == memory_store.list_provider_evaluation_runs()
    )
    assert sql_store.list_provider_evaluation_cases(
        run_id=materialized_run.run.run_id
    ) == memory_store.list_provider_evaluation_cases(run_id=materialized_run.run.run_id)
    assert sql_store.get_materialized_provider_evaluation_run(
        materialized_run.run.run_id
    ) == memory_store.get_materialized_provider_evaluation_run(
        materialized_run.run.run_id
    )


def test_sqlite_runtime_composes_provider_evaluation_store() -> None:
    runtime = build_sqlite_runtime("sqlite:///:memory:", initialize_schema=True)
    facade = runtime.video_action_set
    identity = sql_identity("sql-runtime-provider-evaluation-seed")
    plan = plan_dto(identity=identity, split="development", frame_count=1)
    facade.save_identity(identity)
    facade.save_episode_plan(plan)
    record = sample_record(plan=plan, pixels=_pixels())
    observation = facade.save_observation_record(record)

    cases = [
        sample_case(
            case_ordinal=0,
            frame_id=observation.frame_id,
            outcome=CASE_OUTCOME_EXACT,
        )
    ]
    materialized_run = build_provider_evaluation_run(
        fixture_identity="runtime-sqlite",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="labelled",
        cases=cases,
    )
    saved = facade.save_provider_evaluation_run(materialized_run)
    assert (
        facade.get_materialized_provider_evaluation_run(saved.run.run_id)
        == materialized_run
    )
