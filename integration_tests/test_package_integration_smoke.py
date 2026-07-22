from __future__ import annotations

from pathlib import Path

from zeromodel.analysis import PolicyPropertyChecker, PolicyPropertySpec
from zeromodel.core import LayoutRecipe, ScoreTable, build_vpm
from zeromodel.core.matrix_blob import MatrixBlob
from zeromodel.observation import ImageObservation
from zeromodel.persistence.sqlalchemy import (
    SqlAlchemyVideoActionSetStore,
    create_database_engine,
    create_schema,
    create_session_factory,
    sqlite_database_url,
)
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.video.arcade_policy import (
    ACTIONS,
    CELL_PIXELS,
    FRAME_HEIGHT,
    ShooterConfig,
    compile_policy_artifact,
    enumerate_visual_frames,
)
from zeromodel.video.domains.video_action_set.dto import (
    BenchmarkIdentityDTO,
    EpisodePlanDTO,
)
from zeromodel.video.domains.video_action_set.episode_planning import make_episode_plan
from zeromodel.video.stores.video_action_set_memory import InMemoryVideoActionSetStore
from zeromodel.vision import (
    DeterministicVisualAddressProvider,
    VisualFeatureSpec,
    VisualPolicyReader,
    VisualSignReader,
    build_visual_index,
)


def _small_artifact():
    scores = ScoreTable(
        values=[[0.9, 0.1], [0.4, 0.9], [0.7, 0.2]],
        row_ids=["state=b", "state=a", "state=c"],
        metric_ids=["quality", "uncertainty"],
        metadata={"source": "unit-test"},
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "quality-first",
            "row_order": {
                "kind": "lexicographic",
                "keys": [
                    {"metric_id": "quality", "direction": "desc"},
                    {"metric_id": "uncertainty", "direction": "asc"},
                ],
                "tie_break": "row_id",
            },
            "column_order": {
                "kind": "explicit",
                "metric_ids": ["quality", "uncertainty"],
            },
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(scores, recipe)


def _identity() -> BenchmarkIdentityDTO:
    seed_material = "integration-smoke-seed"
    import hashlib

    return BenchmarkIdentityDTO(
        contract_commit="892511ea8e65a8704cf41d3fa217971cc4c6a36f",
        seed_material=seed_material,
        seed_digest="sha256:" + hashlib.sha256(seed_material.encode()).hexdigest(),
        policy_artifact_id="policy:integration-smoke",
        parent_audit_sha="f251ea80d028f73fdd843fcf0ca22b4173b72b08",
        parent_v3_sha="0b6a4698633e55e99326488d4dbf77b1c266c560",
    )


def _episode(identity: BenchmarkIdentityDTO) -> EpisodePlanDTO:
    return EpisodePlanDTO.from_dict(
        make_episode_plan(
            identity,
            split="final",
            ordinal=0,
            family_label="valid",
            family_ordinal=0,
            source_row_id="row:left",
            row_actions={"row:left": "LEFT", "row:right": "RIGHT"},
        )
    )


def test_core_analysis_observation_vision_video_sql_smoke(tmp_path: Path) -> None:
    artifact = _small_artifact()
    assert (
        artifact.artifact_id
        == "3ce8dd265b949b3b26ebcd602c8b572c248b25c5bafdd13b459a1ab739533e4a"
    )

    checker = PolicyPropertyChecker(
        artifact,
        action_metric_ids=("quality", "uncertainty"),
        evidence_metric_ids=(),
    )
    tautology = PolicyPropertySpec.from_dict(
        {
            "id": "winner_is_itself",
            "version": "1",
            "description": "Bounded integration tautology.",
            "assert": {"eq": [{"var": "winner"}, {"var": "winner"}]},
        }
    )
    assert checker.check([tautology]).passed is True

    config = ShooterConfig(width=3, wave=(0, 2), max_steps=8)
    policy = compile_policy_artifact(config)
    visual_frames = dict(enumerate_visual_frames(config))
    visual_build = build_visual_index(
        policy,
        visual_frames,
        VisualFeatureSpec(
            input_height=FRAME_HEIGHT,
            input_width=config.width * CELL_PIXELS,
            target_height=8,
            target_width=config.width * 2,
            quantization_levels=16,
        ),
        threshold_fraction=0.25,
        margin_fraction=0.75,
        name="integration-smoke-visual-index",
    )
    provider = DeterministicVisualAddressProvider(
        VisualSignReader(
            visual_build.artifact,
            policy,
            action_metric_ids=ACTIONS,
            value_source="raw",
            tie_break="metric_order",
        )
    )
    row_id = str(policy.source.row_ids[0])
    observation = ImageObservation(visual_frames[row_id], source_id="frame-0")
    visual_decision = provider.read(observation)
    assert visual_decision.accepted
    policy_decision = VisualPolicyReader(
        provider,
        VPMPolicyLookup(policy, action_metric_ids=ACTIONS),
    ).read(observation)
    assert policy_decision.action

    identity = _identity()
    episode = _episode(identity)
    blob = MatrixBlob.from_array([[1, 2], [3, 4]], dtype="uint8")
    for store in (InMemoryVideoActionSetStore(),):
        store.save_identity(identity)
        store.save_episode_plan(episode)
        assert store.save_matrix_blob(blob) == blob

    path = tmp_path / "integration.sqlite3"
    engine = create_database_engine(sqlite_database_url(path))
    create_schema(engine)
    session_factory = create_session_factory(engine)
    sql_store = SqlAlchemyVideoActionSetStore(session_factory)
    sql_store.save_identity(identity)
    sql_store.save_episode_plan(episode)
    sql_store.save_matrix_blob(blob)

    reopened = SqlAlchemyVideoActionSetStore(
        create_session_factory(create_database_engine(sqlite_database_url(path)))
    )
    assert reopened.get_identity(identity.seed_digest) == identity
    assert reopened.get_episode_plan(episode.episode_id) == episode
    assert reopened.get_matrix_blob(blob.blob_id) == blob
