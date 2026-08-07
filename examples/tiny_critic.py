from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import numpy as np

from zeromodel.artifacts import ArtifactRef, InMemoryArtifactStore, canonical_json_bytes
from zeromodel.core.matrix_blob import MatrixBlob
from zeromodel.critic import (
    CriticContractDTO,
    CriticEvaluationResultDTO,
    CriticEvaluationSetDTO,
    CriticFeatureBatchDTO,
    CriticFeatureDTO,
    CriticFeatureSpecDTO,
    CriticFitSpecDTO,
    CriticLabelBatchDTO,
    CriticScoreRequestDTO,
    CriticThresholdContractDTO,
    build_critic_score_receipt,
    build_critic_score_vpm,
    budget_selection_metrics,
    compile_critic_readout,
    evaluate_binary_critic,
    evaluate_promotion,
    export_portable_critic,
    load_critic_evaluation_result_aggregate,
    load_critic_readout_aggregate,
    rank_by_critic,
    replay_critic_score,
    score_critic,
    score_portable,
    store_critic_evaluation_result,
    store_critic_evaluation_set,
)
from zeromodel.critic.evaluation import grouped_selection_metrics
from zeromodel.critic.persistence import (
    load_critic_feature_batch_aggregate,
    load_critic_label_batch_aggregate,
    store_critic_contract,
    store_critic_feature_batch,
    store_critic_feature_spec,
    store_critic_label_batch,
    store_matrix_blob,
)
from zeromodel.critic.promotion import CriticPromotionPolicyDTO


TRAINING_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "id": "train-00",
        "stability": 0.92,
        "coverage": 0.88,
        "uncertainty": 0.08,
        "consistency": 0.86,
        "label": 1,
    },
    {
        "id": "train-01",
        "stability": 0.84,
        "coverage": 0.79,
        "uncertainty": 0.18,
        "consistency": 0.74,
        "label": 1,
    },
    {
        "id": "train-02",
        "stability": 0.88,
        "coverage": 0.93,
        "uncertainty": 0.14,
        "consistency": 0.90,
        "label": 1,
    },
    {
        "id": "train-03",
        "stability": 0.76,
        "coverage": 0.82,
        "uncertainty": 0.22,
        "consistency": 0.70,
        "label": 1,
    },
    {
        "id": "train-04",
        "stability": 0.81,
        "coverage": 0.68,
        "uncertainty": 0.20,
        "consistency": 0.77,
        "label": 1,
    },
    {
        "id": "train-05",
        "stability": 0.69,
        "coverage": 0.86,
        "uncertainty": 0.24,
        "consistency": 0.72,
        "label": 1,
    },
    {
        "id": "train-06",
        "stability": 0.91,
        "coverage": 0.62,
        "uncertainty": 0.30,
        "consistency": 0.83,
        "label": 1,
    },
    {
        "id": "train-07",
        "stability": 0.64,
        "coverage": 0.91,
        "uncertainty": 0.26,
        "consistency": 0.68,
        "label": 1,
    },
    {
        "id": "train-08",
        "stability": 0.18,
        "coverage": 0.25,
        "uncertainty": 0.86,
        "consistency": 0.30,
        "label": 0,
    },
    {
        "id": "train-09",
        "stability": 0.28,
        "coverage": 0.32,
        "uncertainty": 0.78,
        "consistency": 0.35,
        "label": 0,
    },
    {
        "id": "train-10",
        "stability": 0.22,
        "coverage": 0.45,
        "uncertainty": 0.74,
        "consistency": 0.28,
        "label": 0,
    },
    {
        "id": "train-11",
        "stability": 0.36,
        "coverage": 0.28,
        "uncertainty": 0.82,
        "consistency": 0.41,
        "label": 0,
    },
    {
        "id": "train-12",
        "stability": 0.48,
        "coverage": 0.42,
        "uncertainty": 0.64,
        "consistency": 0.46,
        "label": 0,
    },
    {
        "id": "train-13",
        "stability": 0.58,
        "coverage": 0.30,
        "uncertainty": 0.58,
        "consistency": 0.63,
        "label": 0,
    },
    {
        "id": "train-14",
        "stability": 0.35,
        "coverage": 0.74,
        "uncertainty": 0.55,
        "consistency": 0.38,
        "label": 0,
    },
    {
        "id": "train-15",
        "stability": 0.70,
        "coverage": 0.34,
        "uncertainty": 0.46,
        "consistency": 0.52,
        "label": 0,
    },
    {
        "id": "train-16",
        "stability": 0.73,
        "coverage": 0.58,
        "uncertainty": 0.42,
        "consistency": 0.65,
        "label": 1,
    },
    {
        "id": "train-17",
        "stability": 0.52,
        "coverage": 0.67,
        "uncertainty": 0.50,
        "consistency": 0.55,
        "label": 0,
    },
)

EVALUATION_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "id": "eval-00",
        "stability": 0.87,
        "coverage": 0.84,
        "uncertainty": 0.16,
        "consistency": 0.82,
        "label": 1,
        "group": "clear-success",
    },
    {
        "id": "eval-01",
        "stability": 0.78,
        "coverage": 0.76,
        "uncertainty": 0.21,
        "consistency": 0.75,
        "label": 1,
        "group": "clear-success",
    },
    {
        "id": "eval-02",
        "stability": 0.24,
        "coverage": 0.29,
        "uncertainty": 0.81,
        "consistency": 0.33,
        "label": 0,
        "group": "clear-failure",
    },
    {
        "id": "eval-03",
        "stability": 0.33,
        "coverage": 0.36,
        "uncertainty": 0.72,
        "consistency": 0.44,
        "label": 0,
        "group": "clear-failure",
    },
    {
        "id": "eval-04",
        "stability": 0.91,
        "coverage": 0.48,
        "uncertainty": 0.38,
        "consistency": 0.86,
        "label": 1,
        "group": "borderline-a",
    },
    {
        "id": "eval-05",
        "stability": 0.44,
        "coverage": 0.88,
        "uncertainty": 0.39,
        "consistency": 0.52,
        "label": 0,
        "group": "borderline-a",
    },
    {
        "id": "eval-06",
        "stability": 0.62,
        "coverage": 0.66,
        "uncertainty": 0.43,
        "consistency": 0.58,
        "label": 1,
        "group": "borderline-b",
    },
    {
        "id": "eval-07",
        "stability": 0.57,
        "coverage": 0.54,
        "uncertainty": 0.48,
        "consistency": 0.61,
        "label": 0,
        "group": "borderline-b",
    },
    {
        "id": "eval-08",
        "stability": 0.80,
        "coverage": 0.43,
        "uncertainty": 0.34,
        "consistency": 0.71,
        "label": 1,
        "group": "mixed",
    },
    {
        "id": "eval-09",
        "stability": 0.41,
        "coverage": 0.60,
        "uncertainty": 0.57,
        "consistency": 0.49,
        "label": 0,
        "group": "mixed",
    },
)

FEATURE_IDS = ("stability", "coverage", "uncertainty", "consistency")


@dataclass(frozen=True, slots=True)
class FixtureBatchRefs:
    feature_ref: ArtifactRef
    label_ref: ArtifactRef
    values: np.ndarray
    labels: np.ndarray
    item_refs: tuple[ArtifactRef, ...]
    group_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TinyCriticRun:
    store: InMemoryArtifactStore
    feature_spec: CriticFeatureSpecDTO
    contract: CriticContractDTO
    fit_spec: CriticFitSpecDTO
    training: FixtureBatchRefs
    evaluation: FixtureBatchRefs
    readout_ref: ArtifactRef
    evaluation_set_ref: ArtifactRef
    evaluation_result_ref: ArtifactRef
    score_result_ref: ArtifactRef
    request_ref: ArtifactRef
    receipt_ref: ArtifactRef
    vpm_artifact_id: str
    replay_result_id: str
    portable_payload: str
    metrics: Mapping[str, Any]
    budget_rows: tuple[Mapping[str, Any], ...]
    promotion: Any


def build_fixture_feature_spec() -> CriticFeatureSpecDTO:
    return CriticFeatureSpecDTO(
        features=(
            CriticFeatureDTO(
                "stability", "higher is associated with success in this fixture"
            ),
            CriticFeatureDTO(
                "coverage", "higher is associated with success in this fixture"
            ),
            CriticFeatureDTO(
                "uncertainty",
                "lower is associated with success in this fixture",
                directionality=-1,
            ),
            CriticFeatureDTO(
                "consistency", "higher is associated with success in this fixture"
            ),
        )
    )


def build_fixture_contract() -> CriticContractDTO:
    return CriticContractDTO(
        critic_id="tiny-fixture-success",
        version="v1",
        target_id="synthetic-success",
        positive_label="successful",
        negative_label="failed",
        score_semantics="Similarity to the successful rows in this deterministic capability fixture.",
        intended_uses=("ranking", "triage"),
        prohibited_uses=(
            "semantic truth",
            "universal quality",
            "reasoning correctness",
            "text quality",
        ),
    )


def _matrix(records: tuple[dict[str, Any], ...]) -> np.ndarray:
    return np.asarray(
        [[float(record[key]) for key in FEATURE_IDS] for record in records],
        dtype=np.float64,
    )


def _labels(records: tuple[dict[str, Any], ...]) -> np.ndarray:
    return np.asarray([float(record["label"]) for record in records], dtype=np.float64)


def _item_refs(
    store: InMemoryArtifactStore, split: str, records: tuple[dict[str, Any], ...]
) -> tuple[ArtifactRef, ...]:
    return tuple(
        store.put(
            "example.tiny_critic.item",
            canonical_json_bytes({"split": split, "id": record["id"]}),
            {"split": split, "fixture": "tiny_critic"},
        )
        for record in records
    )


def store_fixture_batch(
    store: InMemoryArtifactStore,
    *,
    split: str,
    records: tuple[dict[str, Any], ...],
    feature_spec_ref: ArtifactRef,
    contract_ref: ArtifactRef,
) -> FixtureBatchRefs:
    values = _matrix(records)
    labels = _labels(records)
    item_refs = _item_refs(store, split, records)
    group_ids = tuple(str(record.get("group", record["id"])) for record in records)
    feature_batch = CriticFeatureBatchDTO(
        feature_spec_ref=feature_spec_ref,
        values_blob_ref=store_matrix_blob(
            store,
            MatrixBlob.from_array(
                values,
                dtype="float64",
                metadata={"split": split, "role": "critic_features"},
            ),
        ),
        item_refs=item_refs,
        values_shape=values.shape,
        values_dtype="float64",
        metadata={"split": split},
    )
    label_batch = CriticLabelBatchDTO(
        critic_contract_ref=contract_ref,
        item_refs=item_refs,
        labels_blob_ref=store_matrix_blob(
            store,
            MatrixBlob.from_array(
                labels,
                dtype="float64",
                metadata={"split": split, "role": "critic_labels"},
            ),
        ),
        labels_shape=labels.shape,
        metadata={"split": split},
    )
    return FixtureBatchRefs(
        feature_ref=store_critic_feature_batch(store, feature_batch),
        label_ref=store_critic_label_batch(store, label_batch),
        values=values,
        labels=labels,
        item_refs=item_refs,
        group_ids=group_ids,
    )


def independent_portable_score(
    payload: str, row: np.ndarray
) -> dict[str, float | None]:
    data = json.loads(payload)
    x = np.asarray(row, dtype=np.float64)
    direction = np.asarray(data["directionality"], dtype=np.float64)
    center = np.asarray(data["center"], dtype=np.float64)
    scale = np.asarray(data["scale"], dtype=np.float64)
    coefficients = np.asarray(data["coefficients"], dtype=np.float64)
    z = (x * direction - center) / scale
    logit = float(z @ coefficients + float(data["intercept"]))
    score = (
        float(1.0 / (1.0 + np.exp(-logit)))
        if logit >= 0
        else float(np.exp(logit) / (1.0 + np.exp(logit)))
    )
    calibrated = None
    if data.get("calibration") and data["calibration"].get("method") == "platt":
        params = data["calibration"]["parameters"]
        c_logit = float(params.get("a", 1.0)) * logit + float(params.get("b", 0.0))
        calibrated = (
            float(1.0 / (1.0 + np.exp(-c_logit)))
            if c_logit >= 0
            else float(np.exp(c_logit) / (1.0 + np.exp(c_logit)))
        )
    return {"logit": logit, "score": score, "calibrated_probability": calibrated}


def run_tiny_critic_fixture() -> TinyCriticRun:
    store = InMemoryArtifactStore()
    feature_spec = build_fixture_feature_spec()
    contract = build_fixture_contract()
    spec_ref = store_critic_feature_spec(store, feature_spec)
    contract_ref = store_critic_contract(store, contract)
    training = store_fixture_batch(
        store,
        split="training",
        records=TRAINING_RECORDS,
        feature_spec_ref=spec_ref,
        contract_ref=contract_ref,
    )
    evaluation = store_fixture_batch(
        store,
        split="held-out",
        records=EVALUATION_RECORDS,
        feature_spec_ref=spec_ref,
        contract_ref=contract_ref,
    )
    assert {ref.artifact_id for ref in training.item_refs}.isdisjoint(
        {ref.artifact_id for ref in evaluation.item_refs}
    )
    fit_spec = CriticFitSpecDTO(l2_penalty=0.1, max_iterations=80)
    readout, readout_ref = compile_critic_readout(
        store=store,
        features=load_critic_feature_batch_aggregate(training.feature_ref, store),
        labels=load_critic_label_batch_aggregate(training.label_ref, store),
        fit_spec=fit_spec,
    )
    threshold = CriticThresholdContractDTO(reject_below=0.40, accept_at_or_above=0.60)
    request = CriticScoreRequestDTO(
        readout_ref=readout_ref,
        feature_batch_ref=evaluation.feature_ref,
        threshold_contract=threshold,
        explanation_depth=4,
    )
    result, result_ref, request_ref = score_critic(store=store, request=request)
    if result_ref is None:
        raise RuntimeError("result was not persisted")
    scores = np.asarray([item.score for item in result.items], dtype=np.float64)
    metrics = evaluate_binary_critic(evaluation.labels, scores, bin_count=5)
    budget_rows = tuple(
        budget_selection_metrics(evaluation.labels, scores, [0.25, 0.50, 0.75])
    )
    group_results = grouped_selection_metrics(
        evaluation.group_ids, evaluation.labels, scores
    )
    evaluation_set = CriticEvaluationSetDTO(
        feature_batch_ref=evaluation.feature_ref,
        label_batch_ref=evaluation.label_ref,
        split_id="tiny-critic-held-out-v1",
        group_ids=evaluation.group_ids,
        evaluation_contract={
            "threshold_contract": threshold.to_dict(),
            "metrics": ["accuracy", "auroc", "brier", "ece"],
        },
        metadata={"fixture": "deterministic synthetic capability fixture"},
    )
    evaluation_set_ref = store_critic_evaluation_set(store, evaluation_set)
    evaluation_result = CriticEvaluationResultDTO(
        readout_ref=readout_ref,
        evaluation_set_ref=evaluation_set_ref,
        score_result_ref=result_ref,
        metrics=metrics,
        baseline_metrics=baseline_metrics(evaluation.labels, evaluation.values),
        budget_results=budget_rows,
        group_results=group_results,
    )
    evaluation_result_ref = store_critic_evaluation_result(store, evaluation_result)
    load_critic_evaluation_result_aggregate(evaluation_result_ref, store)
    aggregate = load_critic_readout_aggregate(readout_ref, store)
    portable = export_portable_critic(aggregate)
    vpm = build_critic_score_vpm(result=result)
    _, receipt_ref = build_critic_score_receipt(
        store=store, request_ref=request_ref, result_ref=result_ref
    )
    replayed = replay_critic_score(store=store, receipt_ref=receipt_ref)
    promotion = evaluate_promotion(
        current_metrics={"auroc": 0.5, "ece": 0.25},
        candidate_metrics={
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        },
        policy=CriticPromotionPolicyDTO(
            min_candidate_auroc=0.70, max_candidate_ece=0.30, min_auroc_gain=0.05
        ),
    )
    assert (
        score_portable(portable, evaluation.values[0].tolist())["score"]
        == independent_portable_score(portable, evaluation.values[0])["score"]
    )
    return TinyCriticRun(
        store=store,
        feature_spec=feature_spec,
        contract=contract,
        fit_spec=fit_spec,
        training=training,
        evaluation=evaluation,
        readout_ref=readout_ref,
        evaluation_set_ref=evaluation_set_ref,
        evaluation_result_ref=evaluation_result_ref,
        score_result_ref=result_ref,
        request_ref=request_ref,
        receipt_ref=receipt_ref,
        vpm_artifact_id=vpm.artifact_id,
        replay_result_id=replayed.result_id,
        portable_payload=portable,
        metrics=metrics,
        budget_rows=budget_rows,
        promotion=promotion,
    )


def baseline_metrics(labels: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    coverage_scores = values[:, 1]
    directed = values.copy()
    directed[:, 2] = 1.0 - directed[:, 2]
    aggregate_scores = directed.mean(axis=1)
    from zeromodel.critic import evaluate_binary_critic

    return {
        "random_expected_positive_rate": float(np.mean(labels)),
        "coverage_only": evaluate_binary_critic(labels, coverage_scores, bin_count=5),
        "direction_corrected_mean": evaluate_binary_critic(
            labels, aggregate_scores, bin_count=5
        ),
    }


def main() -> None:
    run = run_tiny_critic_fixture()
    print("training-rows:", len(run.training.item_refs))
    print("held-out-rows:", len(run.evaluation.item_refs))
    print("readout:", run.readout_ref.artifact_id)
    print("evaluation-set:", run.evaluation_set_ref.artifact_id)
    print("evaluation-result:", run.evaluation_result_ref.artifact_id)
    print("auroc:", run.metrics["auroc"])
    print("accuracy:", run.metrics["accuracy"])
    print("portable-bytes:", len(run.portable_payload.encode("utf-8")))
    print(
        "ranked:",
        rank_by_critic(
            __import__(
                "zeromodel.critic.persistence"
            ).critic.persistence.load_critic_score_result(
                run.store, run.score_result_ref
            )
        )[:3],
    )
    print("vpm:", run.vpm_artifact_id)
    print("replay:", run.replay_result_id)
    print("promotion:", run.promotion.recommended, "; ".join(run.promotion.reasons))


if __name__ == "__main__":
    main()
