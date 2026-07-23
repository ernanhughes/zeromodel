"""Tests for the controlled PNG representation benchmark: provenance,
provider isolation, comparison/classification, and Store integration
(memory + SQLite). No Ollama - `--backend fake` / synthetic DTOs only.
"""

from __future__ import annotations

import hashlib

import pytest
from zeromodel.video.domains.video_action_set.provider_evaluation_dto import (
    ProviderConfigurationDTO,
    ProviderEvaluationCaseContext,
    ProviderEvaluationCaseDTO,
    ProviderResponseEvidence,
    build_provider_evaluation_run,
)
from zeromodel.video.runtime import build_runtime

import examples.local_model_zero_arcade_test as arcade
from examples.arcade_png_interventions import (
    COOLDOWN_DUAL_VARIANT,
    LABELLED_VARIANT,
    UNLABELLED_VARIANT,
    apply_recipe,
    build_recipe,
)
from examples.arcade_png_representation_comparison import (
    COOLDOWN_TARGET_METRICS,
    GENERIC_TARGET_METRICS,
    classify_variant,
    validate_comparable_runs,
)
from examples.arcade_png_representation_runner import (
    build_intervention_observation,
    build_provider_configuration,
    build_scripted_replies_for_variant,
    find_resumable_run,
    run_variant,
)

POLICY_ARTIFACT_ID = "sha256:" + "11" * 32


def _decision_trace(*, artifact_id: str, row_id: str, action: str) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "row_id": row_id,
        "action": action,
        "metric_id": action,
        "value": 1.0,
        "source_row_index": 0,
        "source_metric_index": 0,
        "view_row": 0,
        "view_column": 0,
        "candidates": [],
        "evidence": {},
    }


def _config(**overrides: object) -> ProviderConfigurationDTO:
    fields: dict[str, object] = {
        "provider_kind": "fake",
        "model_name": "model-a",
        "model_digest": "sha256:" + hashlib.sha256(b"model-a").hexdigest(),
        "runtime_name": "in-process-fake",
        "protocol_version": "proto/v1",
        "prompt_digest": "sha256:" + hashlib.sha256(b"prompt").hexdigest(),
        "seed": 0,
    }
    fields.update(overrides)
    return ProviderConfigurationDTO.build(**fields)


def _build_case(
    *,
    ordinal: int,
    context: ProviderEvaluationCaseContext,
    expected_state: dict[str, object],
    expected_action: str,
    accepted: bool,
    predicted_state: dict[str, object] | None = None,
    predicted_action: str | None = None,
    rejection_reason: str | None = None,
    frame_id: str = "development:fixture:frame-00",
) -> ProviderEvaluationCaseDTO:
    expected_decision = _decision_trace(
        artifact_id=context.policy_artifact_id,
        row_id="row-expected",
        action=expected_action,
    )
    if not accepted:
        return ProviderEvaluationCaseDTO.build(
            case_ordinal=ordinal,
            frame_id=frame_id,
            context=context,
            expected_state=expected_state,
            expected_decision=expected_decision,
            accepted=False,
            evidence=ProviderResponseEvidence(
                rejection_reason=rejection_reason or "test_reject"
            ),
        )
    predicted_decision = _decision_trace(
        artifact_id=context.policy_artifact_id,
        row_id="row-predicted",
        action=predicted_action,
    )
    return ProviderEvaluationCaseDTO.build(
        case_ordinal=ordinal,
        frame_id=frame_id,
        context=context,
        expected_state=expected_state,
        expected_decision=expected_decision,
        accepted=True,
        predicted_state=predicted_state,
        predicted_decision=predicted_decision,
        evidence=ProviderResponseEvidence(provider_latency_us=1_000),
    )


def _simple_run(
    *,
    representation_mode: str,
    provider_configuration: ProviderConfigurationDTO,
    policy_artifact_id: str = POLICY_ARTIFACT_ID,
    fixture_identity: str = "fixture",
    case_mode: str = "mode",
    exact_count: int = 0,
    equivalent_count: int = 0,
    changing_count: int = 0,
    rejected_count: int = 0,
):
    context = ProviderEvaluationCaseContext(
        policy_artifact_id=policy_artifact_id,
        provider_configuration_id=provider_configuration.provider_configuration_id,
    )
    cases = []
    ordinal = 0
    for _ in range(exact_count):
        cases.append(
            _build_case(
                ordinal=ordinal,
                context=context,
                expected_state={"x": 0},
                expected_action="STAY",
                accepted=True,
                predicted_state={"x": 0},
                predicted_action="STAY",
            )
        )
        ordinal += 1
    for _ in range(equivalent_count):
        cases.append(
            _build_case(
                ordinal=ordinal,
                context=context,
                expected_state={"x": 0},
                expected_action="STAY",
                accepted=True,
                predicted_state={"x": 1},
                predicted_action="STAY",
            )
        )
        ordinal += 1
    for _ in range(changing_count):
        cases.append(
            _build_case(
                ordinal=ordinal,
                context=context,
                expected_state={"x": 0},
                expected_action="STAY",
                accepted=True,
                predicted_state={"x": 1},
                predicted_action="FIRE",
            )
        )
        ordinal += 1
    for _ in range(rejected_count):
        cases.append(
            _build_case(
                ordinal=ordinal,
                context=context,
                expected_state={"x": 0},
                expected_action="STAY",
                accepted=False,
            )
        )
        ordinal += 1
    return build_provider_evaluation_run(
        fixture_identity=fixture_identity,
        provider_configuration=provider_configuration,
        policy_artifact_id=policy_artifact_id,
        case_mode=case_mode,
        representation_mode=representation_mode,
        cases=cases,
    )


def _factor_run(
    *,
    representation_mode: str,
    provider_configuration: ProviderConfigurationDTO,
    cooldown_correct_count: int,
    cooldown_wrong_count: int,
    policy_artifact_id: str = POLICY_ARTIFACT_ID,
    fixture_identity: str = "fixture",
    case_mode: str = "mode",
):
    """Cases whose exact/action outcome never changes (always
    action_equivalent) but whose ``cooldown`` factor correctness varies -
    isolates a factor-specific classification signal from the generic one."""
    context = ProviderEvaluationCaseContext(
        policy_artifact_id=policy_artifact_id,
        provider_configuration_id=provider_configuration.provider_configuration_id,
    )
    cooldown_pattern = [True] * cooldown_correct_count + [False] * cooldown_wrong_count
    cases = []
    for ordinal, cooldown_matches in enumerate(cooldown_pattern):
        predicted_cooldown = 0 if cooldown_matches else 1
        cases.append(
            _build_case(
                ordinal=ordinal,
                context=context,
                expected_state={"tank_column": 0, "cooldown": 0},
                expected_action="STAY",
                accepted=True,
                predicted_state={"tank_column": 1, "cooldown": predicted_cooldown},
                predicted_action="STAY",
            )
        )
    return build_provider_evaluation_run(
        fixture_identity=fixture_identity,
        provider_configuration=provider_configuration,
        policy_artifact_id=policy_artifact_id,
        case_mode=case_mode,
        representation_mode=representation_mode,
        cases=cases,
    )


class TestComparability:
    def test_representation_only_change_stays_comparable(self) -> None:
        config = _config()
        baseline = _simple_run(
            representation_mode=UNLABELLED_VARIANT,
            provider_configuration=config,
            exact_count=8,
        )
        candidate = _simple_run(
            representation_mode=COOLDOWN_DUAL_VARIANT,
            provider_configuration=config,
            exact_count=8,
        )
        validate_comparable_runs((baseline, candidate))  # must not raise

    @pytest.mark.parametrize(
        "field",
        [
            "provider_configuration_id",
            "model_digest",
            "prompt_digest",
            "protocol_version",
            "policy_artifact_id",
            "fixture_identity",
            "case_mode",
        ],
    )
    def test_each_fixed_identity_dimension_rejects_comparison(self, field: str) -> None:
        base_config = _config()
        base_run = _simple_run(
            representation_mode=UNLABELLED_VARIANT,
            provider_configuration=base_config,
            exact_count=4,
        )
        if field == "provider_configuration_id":
            candidate_config = _config(metadata={"note": "different"})
            candidate_run = _simple_run(
                representation_mode=LABELLED_VARIANT,
                provider_configuration=candidate_config,
                exact_count=4,
            )
        elif field == "model_digest":
            candidate_config = _config(
                model_digest="sha256:" + hashlib.sha256(b"other").hexdigest()
            )
            candidate_run = _simple_run(
                representation_mode=LABELLED_VARIANT,
                provider_configuration=candidate_config,
                exact_count=4,
            )
        elif field == "prompt_digest":
            candidate_config = _config(
                prompt_digest="sha256:" + hashlib.sha256(b"other-prompt").hexdigest()
            )
            candidate_run = _simple_run(
                representation_mode=LABELLED_VARIANT,
                provider_configuration=candidate_config,
                exact_count=4,
            )
        elif field == "protocol_version":
            candidate_config = _config(protocol_version="proto/v2")
            candidate_run = _simple_run(
                representation_mode=LABELLED_VARIANT,
                provider_configuration=candidate_config,
                exact_count=4,
            )
        elif field == "policy_artifact_id":
            candidate_run = _simple_run(
                representation_mode=LABELLED_VARIANT,
                provider_configuration=base_config,
                policy_artifact_id="sha256:" + "22" * 32,
                exact_count=4,
            )
        elif field == "fixture_identity":
            candidate_run = _simple_run(
                representation_mode=LABELLED_VARIANT,
                provider_configuration=base_config,
                fixture_identity="other-fixture",
                exact_count=4,
            )
        else:
            candidate_run = _simple_run(
                representation_mode=LABELLED_VARIANT,
                provider_configuration=base_config,
                case_mode="other-mode",
                exact_count=4,
            )
        with pytest.raises(Exception, match=field.split("_")[0]):
            validate_comparable_runs((base_run, candidate_run))


class TestClassification:
    def test_no_material_change_when_identical(self) -> None:
        config = _config()
        baseline = _simple_run(
            representation_mode=UNLABELLED_VARIANT,
            provider_configuration=config,
            exact_count=8,
        )
        candidate = _simple_run(
            representation_mode=COOLDOWN_DUAL_VARIANT,
            provider_configuration=config,
            exact_count=8,
        )
        result = classify_variant(
            baseline=baseline,
            candidate=candidate,
            target_metrics=GENERIC_TARGET_METRICS,
        )
        assert result.label == "no_material_change"

    def test_advance_when_exact_count_improves(self) -> None:
        config = _config()
        baseline = _simple_run(
            representation_mode=UNLABELLED_VARIANT,
            provider_configuration=config,
            exact_count=6,
            equivalent_count=2,
        )
        candidate = _simple_run(
            representation_mode=COOLDOWN_DUAL_VARIANT,
            provider_configuration=config,
            exact_count=8,
        )
        result = classify_variant(
            baseline=baseline,
            candidate=candidate,
            target_metrics=GENERIC_TARGET_METRICS,
        )
        assert result.label == "advance"
        assert "exact_count" in result.reasoning

    def test_regression_when_action_changing_increases(self) -> None:
        config = _config()
        baseline = _simple_run(
            representation_mode=UNLABELLED_VARIANT,
            provider_configuration=config,
            exact_count=8,
        )
        candidate = _simple_run(
            representation_mode=COOLDOWN_DUAL_VARIANT,
            provider_configuration=config,
            exact_count=7,
            changing_count=1,
        )
        result = classify_variant(
            baseline=baseline,
            candidate=candidate,
            target_metrics=GENERIC_TARGET_METRICS,
        )
        assert result.label == "regression"
        assert "action_changing" in result.reasoning

    def test_regression_when_rejected_increases(self) -> None:
        config = _config()
        baseline = _simple_run(
            representation_mode=UNLABELLED_VARIANT,
            provider_configuration=config,
            exact_count=8,
        )
        candidate = _simple_run(
            representation_mode=COOLDOWN_DUAL_VARIANT,
            provider_configuration=config,
            exact_count=7,
            rejected_count=1,
        )
        result = classify_variant(
            baseline=baseline,
            candidate=candidate,
            target_metrics=GENERIC_TARGET_METRICS,
        )
        assert result.label == "regression"

    def test_incompatible_when_policy_differs(self) -> None:
        config = _config()
        baseline = _simple_run(
            representation_mode=UNLABELLED_VARIANT,
            provider_configuration=config,
            exact_count=8,
        )
        candidate = _simple_run(
            representation_mode=COOLDOWN_DUAL_VARIANT,
            provider_configuration=config,
            policy_artifact_id="sha256:" + "33" * 32,
            exact_count=8,
        )
        result = classify_variant(
            baseline=baseline,
            candidate=candidate,
            target_metrics=GENERIC_TARGET_METRICS,
        )
        assert result.label == "incompatible"

    def test_family_target_metric_detects_factor_improvement_generic_does_not(
        self,
    ) -> None:
        config = _config()
        baseline = _factor_run(
            representation_mode=UNLABELLED_VARIANT,
            provider_configuration=config,
            cooldown_correct_count=2,
            cooldown_wrong_count=6,
        )
        candidate = _factor_run(
            representation_mode=COOLDOWN_DUAL_VARIANT,
            provider_configuration=config,
            cooldown_correct_count=6,
            cooldown_wrong_count=2,
        )
        generic_result = classify_variant(
            baseline=baseline,
            candidate=candidate,
            target_metrics=GENERIC_TARGET_METRICS,
        )
        cooldown_result = classify_variant(
            baseline=baseline,
            candidate=candidate,
            target_metrics=COOLDOWN_TARGET_METRICS,
        )
        assert generic_result.label == "no_material_change"
        assert cooldown_result.label == "advance"
        assert "cooldown" in cooldown_result.reasoning


class TestProviderIsolation:
    def test_predict_receives_only_bytes_and_fixed_render_mode(self) -> None:
        recorded: list[tuple[bytes, str]] = []
        recipe = build_recipe(UNLABELLED_VARIANT)
        states = arcade.smoke_states()[:2]
        steps_by_state = [
            apply_recipe(recipe, arcade.render(state, recipe.base_render_mode))
            for state in states
        ]
        replies = build_scripted_replies_for_variant(states, steps_by_state)

        class _SpyProvider:
            def predict(self, image: bytes, render_mode: str) -> arcade.ProviderReply:
                recorded.append((image, render_mode))
                digest = "sha256:" + hashlib.sha256(image).hexdigest()
                return replies[digest]

        runtime = build_runtime()
        facade = runtime.video_action_set
        identity = arcade._build_benchmark_identity(
            model="fake", artifact_id=POLICY_ARTIFACT_ID, stamp="isolation-test"
        )
        facade.save_identity(identity)
        plan = arcade._build_episode_plan(
            identity=identity,
            episode_id="development:isolation-test",
            frame_count=len(states),
        )
        facade.save_episode_plan(plan)
        reader, artifact_hex = arcade.policy_reader()
        policy_artifact_id = f"sha256:{artifact_hex}"
        provider_configuration = build_provider_configuration(
            backend="fake", model="fake", seed=0
        )

        run_variant(
            variant_id=UNLABELLED_VARIANT,
            recipe=recipe,
            states=states,
            provider=_SpyProvider(),
            provider_configuration=provider_configuration,
            policy_artifact_id=policy_artifact_id,
            reader=reader,
            facade=facade,
            identity=identity,
            plan=plan,
            fixture_identity="isolation-fixture",
            case_mode="isolation-mode",
            confidence_threshold=0.0,
            metadata={},
        )

        assert len(recorded) == len(states)
        for image, render_mode in recorded:
            assert isinstance(image, bytes)
            assert render_mode == "unlabelled"
            assert not hasattr(image, "tank_column")
            assert not hasattr(image, "row_id")


class TestProvenance:
    def _build(self, variant_id: str):
        recipe = build_recipe(variant_id)
        state = arcade.ArcadeState(3, 3, 1)
        steps = apply_recipe(recipe, arcade.render(state, recipe.base_render_mode))
        identity = arcade._build_benchmark_identity(
            model="fake", artifact_id=POLICY_ARTIFACT_ID, stamp="provenance-test"
        )
        plan = arcade._build_episode_plan(
            identity=identity, episode_id="development:provenance-test", frame_count=1
        )
        observation = build_intervention_observation(
            identity=identity,
            plan=plan,
            frame_index=0,
            recipe=recipe,
            steps=steps,
            truth_row_id=state.row_id,
            truth_action="STAY",
        )
        return recipe, steps, observation

    def test_chain_is_ordered_and_contiguous(self) -> None:
        _, _, observation = self._build(COOLDOWN_DUAL_VARIANT)
        chain = observation.observation.operation_chain
        assert [op.index for op in chain.operations] == list(
            range(len(chain.operations))
        )

    def test_first_operation_has_no_input_digest(self) -> None:
        _, _, observation = self._build(COOLDOWN_DUAL_VARIANT)
        chain = observation.observation.operation_chain
        assert chain.operations[0].input_digests == (None,)

    def test_each_operation_output_feeds_next_input(self) -> None:
        _, _, observation = self._build(COOLDOWN_DUAL_VARIANT)
        chain = observation.observation.operation_chain
        for previous, following in zip(
            chain.operations[:-1], chain.operations[1:], strict=True
        ):
            assert following.input_digests == (previous.output_digest,)

    def test_final_chain_output_matches_observation_pixel_digest(self) -> None:
        _, _, observation = self._build(COOLDOWN_DUAL_VARIANT)
        chain = observation.observation.operation_chain
        assert chain.final_emitted_digest == chain.operations[-1].output_digest
        assert (
            chain.final_emitted_digest
            == observation.observation.observation_pixel_digest
        )

    def test_reference_variant_has_single_render_operation(self) -> None:
        _, _, observation = self._build(UNLABELLED_VARIANT)
        chain = observation.observation.operation_chain
        assert len(chain.operations) == 1
        assert chain.operations[0].operation == "render_frame"

    def test_matrix_blob_matches_observation_reference(self) -> None:
        _, _, observation = self._build(COOLDOWN_DUAL_VARIANT)
        assert observation.matrix_blob.blob_id == observation.observation.matrix_blob_id

    def test_metadata_links_recipe_and_source_fixture(self) -> None:
        recipe, steps, observation = self._build(COOLDOWN_DUAL_VARIANT)
        metadata = observation.observation.metadata.to_value()
        assert metadata["recipe_id"] == recipe.recipe_id
        assert metadata["variant_id"] == COOLDOWN_DUAL_VARIANT
        assert (
            observation.observation.expected_row == arcade.ArcadeState(3, 3, 1).row_id
        )
        assert metadata["source_full_resolution_image_sha256"] == (
            "sha256:" + hashlib.sha256(steps[0][1]).hexdigest()
        )
        assert metadata["final_full_resolution_image_sha256"] == (
            "sha256:" + hashlib.sha256(steps[-1][1]).hexdigest()
        )


class TestResume:
    def _run_once(
        self,
        facade,
        *,
        variant_id: str,
        states,
        provider_configuration,
        policy_artifact_id,
        reader,
        run_label=None,
    ):
        recipe = build_recipe(variant_id)
        label = run_label or variant_id
        identity = arcade._build_benchmark_identity(
            model="fake", artifact_id=policy_artifact_id, stamp=f"resume-{label}"
        )
        facade.save_identity(identity)
        plan = arcade._build_episode_plan(
            identity=identity,
            episode_id=f"development:resume-{label}",
            frame_count=len(states),
        )
        facade.save_episode_plan(plan)
        steps_by_state = [
            apply_recipe(recipe, arcade.render(state, recipe.base_render_mode))
            for state in states
        ]
        provider = arcade.ScriptedProvider(
            build_scripted_replies_for_variant(states, steps_by_state)
        )
        saved_run, _records = run_variant(
            variant_id=variant_id,
            recipe=recipe,
            states=states,
            provider=provider,
            provider_configuration=provider_configuration,
            policy_artifact_id=policy_artifact_id,
            reader=reader,
            facade=facade,
            identity=identity,
            plan=plan,
            fixture_identity="resume-fixture",
            case_mode="resume-mode",
            confidence_threshold=0.0,
            metadata={},
        )
        return saved_run

    def test_idempotent_resume_returns_the_same_run(self) -> None:
        runtime = build_runtime()
        facade = runtime.video_action_set
        reader, artifact_hex = arcade.policy_reader()
        policy_artifact_id = f"sha256:{artifact_hex}"
        provider_configuration = build_provider_configuration(
            backend="fake", model="fake", seed=0
        )
        states = arcade.smoke_states()[:2]

        first = self._run_once(
            facade,
            variant_id=UNLABELLED_VARIANT,
            states=states,
            provider_configuration=provider_configuration,
            policy_artifact_id=policy_artifact_id,
            reader=reader,
        )
        found = find_resumable_run(
            facade=facade,
            fixture_identity="resume-fixture",
            provider_configuration_id=provider_configuration.provider_configuration_id,
            policy_artifact_id=policy_artifact_id,
            case_mode="resume-mode",
            variant_id=UNLABELLED_VARIANT,
        )
        assert found is not None
        assert found.run.run_id == first.run.run_id
        assert found == first

    def test_resume_finds_nothing_for_unrun_variant(self) -> None:
        runtime = build_runtime()
        facade = runtime.video_action_set
        found = find_resumable_run(
            facade=facade,
            fixture_identity="resume-fixture",
            provider_configuration_id="sha256:" + "44" * 32,
            policy_artifact_id=POLICY_ARTIFACT_ID,
            case_mode="resume-mode",
            variant_id=UNLABELLED_VARIANT,
        )
        assert found is None

    def test_resume_rejects_multiple_candidates(self) -> None:
        runtime = build_runtime()
        facade = runtime.video_action_set
        reader, artifact_hex = arcade.policy_reader()
        policy_artifact_id = f"sha256:{artifact_hex}"
        provider_configuration = build_provider_configuration(
            backend="fake", model="fake", seed=0
        )

        self._run_once(
            facade,
            variant_id=UNLABELLED_VARIANT,
            states=arcade.smoke_states()[:1],
            provider_configuration=provider_configuration,
            policy_artifact_id=policy_artifact_id,
            reader=reader,
            run_label="candidate-a",
        )
        self._run_once(
            facade,
            variant_id=UNLABELLED_VARIANT,
            states=arcade.smoke_states()[1:2],
            provider_configuration=provider_configuration,
            policy_artifact_id=policy_artifact_id,
            reader=reader,
            run_label="candidate-b",
        )
        with pytest.raises(RuntimeError, match="refusing to guess"):
            find_resumable_run(
                facade=facade,
                fixture_identity="resume-fixture",
                provider_configuration_id=provider_configuration.provider_configuration_id,
                policy_artifact_id=policy_artifact_id,
                case_mode="resume-mode",
                variant_id=UNLABELLED_VARIANT,
            )


@pytest.mark.integration
class TestSqliteStoreIntegration:
    def test_run_persists_and_reloads_through_sqlite(self, tmp_path) -> None:
        from zeromodel.persistence.sqlalchemy.db.runtime import build_sqlite_runtime

        db_path = tmp_path / "arcade-png-benchmark-test.db"
        runtime = build_sqlite_runtime(f"sqlite:///{db_path}", initialize_schema=True)
        facade = runtime.video_action_set
        reader, artifact_hex = arcade.policy_reader()
        policy_artifact_id = f"sha256:{artifact_hex}"
        provider_configuration = build_provider_configuration(
            backend="fake", model="fake", seed=0
        )
        states = arcade.smoke_states()[:2]

        recipe = build_recipe(COOLDOWN_DUAL_VARIANT)
        identity = arcade._build_benchmark_identity(
            model="fake", artifact_id=policy_artifact_id, stamp="sqlite-test"
        )
        facade.save_identity(identity)
        plan = arcade._build_episode_plan(
            identity=identity,
            episode_id="development:sqlite-test",
            frame_count=len(states),
        )
        facade.save_episode_plan(plan)
        steps_by_state = [
            apply_recipe(recipe, arcade.render(state, recipe.base_render_mode))
            for state in states
        ]
        provider = arcade.ScriptedProvider(
            build_scripted_replies_for_variant(states, steps_by_state)
        )
        saved_run, _records = run_variant(
            variant_id=COOLDOWN_DUAL_VARIANT,
            recipe=recipe,
            states=states,
            provider=provider,
            provider_configuration=provider_configuration,
            policy_artifact_id=policy_artifact_id,
            reader=reader,
            facade=facade,
            identity=identity,
            plan=plan,
            fixture_identity="sqlite-fixture",
            case_mode="sqlite-mode",
            confidence_threshold=0.0,
            metadata={},
        )

        reloaded = facade.get_materialized_provider_evaluation_run(saved_run.run.run_id)
        assert reloaded == saved_run
        assert reloaded.summary.attempted_count == len(states)
        for case in reloaded.cases:
            observation = facade.get_materialized_observation(case.frame_id)
            assert observation is not None
            assert len(observation.observation.operation_chain.operations) == 2
