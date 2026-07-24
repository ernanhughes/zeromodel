from __future__ import annotations

import dataclasses

import numpy as np

from zeromodel.perception import (
    DiscreteActionSchemaDTO,
    OperationalDriftPolicyDTO,
    PromotionPolicyDTO,
    RecordedInteractionDTO,
    SourceImageEncoderSpecDTO,
    SqlitePerceptionModelLifecycleStore,
    SqlitePerceptionProductionLedgerStore,
    TemporalSourceVPMDTO,
    TemporalWindowSpecDTO,
    activate_promoted_model,
    bind_comparison_report_to_partition,
    build_dataset_manifest,
    build_dataset_partition,
    build_grid_field_schema,
    build_model_lifecycle_snapshot,
    build_operational_reference_profile,
    compare_single_and_temporal_inference,
    diagnose_operational_health,
    encode_discrete_action,
    encode_source_array,
    evaluate_partition_owned_model_on_test,
    fit_source_target_translator,
    fit_temporal_translator,
    promote_partition_owned_model,
    record_production_inference,
    record_production_outcome,
    register_promoted_model,
    resolve_active_promoted_model,
    run_promoted_inference,
    supersede_active_model,
)
from zeromodel.perception.compatibility import (
    assess_rollback_compatibility,
    build_model_compatibility_contract,
    rollback_compatible_model,
)


def _pipeline_fixture():
    source_spec = SourceImageEncoderSpecDTO(color_space="L")
    montage_spec = SourceImageEncoderSpecDTO(
        color_space="L",
        max_width=2,
        max_height=1,
        max_pixels=2,
        version="p16g-temporal-montage/1",
    )
    action_schema = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])
    window_spec = TemporalWindowSpecDTO(frame_count=2)

    interactions = []
    current_sources = {}
    temporal_sources = []
    for index in range(90):
        action = "LEFT" if index % 2 == 0 else "RIGHT"
        previous_value = 0 if action == "LEFT" else 255
        current_value = 32 + index
        previous = encode_source_array(
            np.asarray([[previous_value]], dtype=np.uint8), source_spec
        )
        current = encode_source_array(
            np.asarray([[current_value]], dtype=np.uint8), source_spec
        )
        target = encode_discrete_action(action, action_schema)
        interaction = RecordedInteractionDTO.from_vpms(
            sequence_id=f"episode-{index:03d}",
            step_index=1,
            source=current,
            target=target,
        )
        montage = encode_source_array(
            np.asarray([[previous_value, current_value]], dtype=np.uint8), montage_spec
        )
        interactions.append(interaction)
        current_sources[current.source_vpm_id] = current
        temporal_sources.append(
            TemporalSourceVPMDTO(
                temporal_source_id=f"sha256:p16g-temporal-{index:03d}",
                temporal_window_spec_id=window_spec.temporal_window_spec_id,
                sequence_id=interaction.sequence_id,
                target_interaction_id=interaction.interaction_id,
                target_step_index=1,
                action_label=action,
                frame_source_vpm_ids=(previous.source_vpm_id, current.source_vpm_id),
                frame_pixel_digests=(previous.pixel_digest, current.pixel_digest),
                current_source_vpm_id=current.source_vpm_id,
                current_pixel_digest=current.pixel_digest,
                montage_source_vpm=montage,
            )
        )

    manifest = build_dataset_manifest(
        interactions,
        source_encoder_spec_ids=(source_spec.encoder_spec_id,),
        split_seed="p16g/full-vertical-integration",
    )
    partitions = {
        split: build_dataset_partition(manifest, split)
        for split in ("train", "validation", "test")
    }
    by_id = {item.target_interaction_id: item for item in temporal_sources}
    split_temporal = {
        split: tuple(by_id[item_id] for item_id in partitions[split].interaction_ids)
        for split in partitions
    }
    single_schema = build_grid_field_schema(
        next(iter(current_sources.values())), tile_width=1, tile_height=1
    )
    temporal_schema = build_grid_field_schema(
        temporal_sources[0].montage_source_vpm, tile_width=1, tile_height=1
    )
    single = fit_source_target_translator(
        manifest,
        current_sources,
        single_schema,
        action_schema,
        training_split="train",
    )
    temporal = fit_temporal_translator(
        split_temporal["train"],
        window_spec,
        temporal_schema,
        action_schema,
        training_split="train",
    )
    validation = compare_single_and_temporal_inference(
        single,
        temporal,
        single_schema,
        temporal_schema,
        split_temporal["validation"],
        current_sources,
        action_schema,
        split="validation",
    )
    owned_validation = bind_comparison_report_to_partition(
        validation, partitions["validation"]
    )
    return (
        action_schema,
        source_spec,
        window_spec,
        manifest,
        partitions,
        current_sources,
        split_temporal,
        single_schema,
        temporal_schema,
        single,
        temporal,
        owned_validation,
    )


def test_full_pipeline_survives_sqlite_restart_and_preserves_lineage(tmp_path) -> None:
    (
        action_schema,
        source_spec,
        window_spec,
        manifest,
        partitions,
        current_sources,
        split_temporal,
        single_schema,
        temporal_schema,
        single,
        temporal,
        owned_validation,
    ) = _pipeline_fixture()

    _, earlier = promote_partition_owned_model(
        owned_validation,
        policy=PromotionPolicyDTO(minimum_accuracy_gain=-0.75),
    )
    _, current = promote_partition_owned_model(
        owned_validation,
        policy=PromotionPolicyDTO(minimum_accuracy_gain=-0.50),
    )
    assert earlier.promoted_model_id != current.promoted_model_id
    assert earlier.model_kind == current.model_kind

    owned_test = evaluate_partition_owned_model_on_test(
        partitions["test"],
        current,
        action_schema,
        split_temporal["test"],
        current_sources,
        single_translator=single,
        temporal_translator=temporal,
        single_field_schema=single_schema,
        temporal_field_schema=temporal_schema,
    )
    reference = build_operational_reference_profile(owned_test.test_report)

    lifecycle_path = tmp_path / "p16g-lifecycle.sqlite3"
    production_path = tmp_path / "p16g-production.sqlite3"
    with SqlitePerceptionModelLifecycleStore(lifecycle_path) as lifecycle:
        earlier_entry = register_promoted_model(
            lifecycle,
            earlier,
            registered_by="p16g",
            registration_reason="earlier governed candidate",
        )
        activate_promoted_model(
            lifecycle,
            earlier.promoted_model_id,
            actor="p16g-operator",
            reason="initial activation",
        )
        current_entry = register_promoted_model(
            lifecycle,
            current,
            registered_by="p16g",
            registration_reason="current governed candidate",
            test_evaluation=owned_test.test_report,
        )
        _, pointer = supersede_active_model(
            lifecycle,
            current.promoted_model_id,
            actor="p16g-operator",
            reason="validation-owned promotion",
        )

        with SqlitePerceptionProductionLedgerStore(production_path) as production:
            for temporal_source in split_temporal["test"]:
                source = current_sources[temporal_source.current_source_vpm_id]
                result = run_promoted_inference(
                    current,
                    action_schema,
                    single_translator=single,
                    temporal_translator=temporal,
                    single_field_schema=single_schema,
                    temporal_field_schema=temporal_schema,
                    source=source if current.model_kind == "single_frame" else None,
                    temporal_source=(
                        temporal_source if current.model_kind == "temporal" else None
                    ),
                )
                record = record_production_inference(
                    production, pointer, current_entry, result
                )
                record_production_outcome(
                    production,
                    record.record_id,
                    observed_action=temporal_source.action_label,
                    source="p16g-ground-truth",
                )

    with SqlitePerceptionModelLifecycleStore(lifecycle_path) as lifecycle:
        assert resolve_active_promoted_model(lifecycle) == current
        snapshot = build_model_lifecycle_snapshot(lifecycle)
        assert snapshot.active_pointer.revision == 2
        assert tuple(item.transition_kind for item in snapshot.transitions) == (
            "activate",
            "supersede",
        )

        current_contract = build_model_compatibility_contract(
            current,
            action_schema_id=action_schema.action_schema_id,
            source_encoder_spec_id=source_spec.encoder_spec_id,
            field_schema_id=(
                temporal_schema.field_schema_id
                if current.model_kind == "temporal"
                else single_schema.field_schema_id
            ),
            inference_semantics_version="p16g-runtime/1",
            deployment_slot="primary",
        )
        earlier_contract = build_model_compatibility_contract(
            earlier,
            action_schema_id=action_schema.action_schema_id,
            source_encoder_spec_id=source_spec.encoder_spec_id,
            field_schema_id=(
                temporal_schema.field_schema_id
                if earlier.model_kind == "temporal"
                else single_schema.field_schema_id
            ),
            inference_semantics_version="p16g-runtime/1",
            deployment_slot="primary",
        )
        assessment = assess_rollback_compatibility(
            current_contract, earlier_contract
        )
        assert assessment.status == "compatible"
        rollback_compatible_model(
            lifecycle,
            earlier.promoted_model_id,
            current_contract=current_contract,
            target_contract=earlier_contract,
            actor="p16g-operator",
            reason="verified compatible rollback",
        )

    with SqlitePerceptionProductionLedgerStore(production_path) as production:
        report = diagnose_operational_health(
            reference,
            production,
            start_sequence_number=1,
            policy=OperationalDriftPolicyDTO(
                minimum_reference_count=1,
                minimum_inference_count=1,
                minimum_labeled_count=1,
                minimum_accepted_labeled_count=1,
                minimum_label_coverage=1.0,
            ),
        )
        assert report.overall_status in {"healthy", "drifted"}
        assert report.inference_record_ids == tuple(
            item.record_id for item in production.list_inferences()
        )
        assert report.outcome_ids == tuple(
            item.outcome_id for item in production.list_outcomes()
        )

    assert manifest.dataset_id == partitions["train"].dataset_id
    assert owned_validation.dataset_id == manifest.dataset_id
    assert owned_test.dataset_id == manifest.dataset_id
    assert owned_test.test_report.promoted_model_id == current.promoted_model_id
    assert window_spec.temporal_window_spec_id == temporal.temporal_window_spec_id
