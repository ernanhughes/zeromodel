from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from zeromodel.artifacts import canonical_json_bytes

from zeromodel.critic.errors import CriticPayloadTooLargeError, CriticValidationError
from zeromodel.critic.linear import CompiledCriticReadout, stable_sigmoid
from zeromodel.critic.persistence import ResolvedCriticReadoutAggregate
from zeromodel.critic.scoring import compiled_from_aggregate

PORTABLE_SCHEMA_VERSION = "zeromodel-portable-critic/v1"


def export_portable_critic(
    aggregate: ResolvedCriticReadoutAggregate, *, limit_bytes: int | None = None
) -> str:
    runtime = compiled_from_aggregate(aggregate)
    payload = {
        "schema_version": PORTABLE_SCHEMA_VERSION,
        "critic_contract_id": runtime.contract_id,
        "feature_spec_id": runtime.feature_spec_id,
        "feature_ids": list(runtime.feature_ids),
        "directionality": list(runtime.directionality),
        "center": [float(v) for v in runtime.center],
        "scale": [float(v) for v in runtime.scale],
        "coefficients": [float(v) for v in runtime.coefficients],
        "intercept": float(runtime.intercept),
        "calibration": None
        if aggregate.calibration is None
        else aggregate.calibration.to_dict(),
        "score_contract": {
            "score": "sigmoid(logit)",
            "calibration_input": "logit",
            "semantics": aggregate.contract.score_semantics,
            "positive_label": aggregate.contract.positive_label,
            "negative_label": aggregate.contract.negative_label,
        },
    }
    encoded = canonical_json_bytes(payload)
    max_bytes = (
        aggregate.fit_spec.portable_payload_limit_bytes
        if limit_bytes is None
        else int(limit_bytes)
    )
    if len(encoded) > max_bytes:
        raise CriticPayloadTooLargeError(
            f"portable critic payload is {len(encoded)} bytes, limit is {max_bytes}"
        )
    return encoded.decode("utf-8")


def load_portable_critic(payload: str | bytes | Mapping[str, Any]) -> Mapping[str, Any]:
    data = (
        json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
        if isinstance(payload, (str, bytes))
        else dict(payload)
    )
    if data.get("schema_version") != PORTABLE_SCHEMA_VERSION:
        raise CriticValidationError("unsupported portable critic schema")
    return data


def score_portable(
    payload: str | bytes | Mapping[str, Any],
    values: Mapping[str, float | None] | list[float],
) -> dict[str, float | None]:
    data = load_portable_critic(payload)
    if isinstance(values, Mapping):
        # Portable payloads do not carry missing-value policy;
        # producers must supply complete compatible rows.
        resolved_values: list[float] = []

        for feature_id in data["feature_ids"]:
            value = values[feature_id]
            if value is None:
                raise CriticValidationError(
                    f"portable value for feature {feature_id!r} must not be missing"
                )
            resolved_values.append(float(value))

        row = np.asarray(resolved_values, dtype=np.float64)
    else:
        row = np.asarray(values, dtype=np.float64)
    if (
        row.ndim != 1
        or row.size != len(data["feature_ids"])
        or not np.isfinite(row).all()
    ):
        raise CriticValidationError("portable values must be a finite compatible row")
    direction = np.asarray(data["directionality"], dtype=np.float64)
    center = np.asarray(data["center"], dtype=np.float64)
    scale = np.asarray(data["scale"], dtype=np.float64)
    coefficients = np.asarray(data["coefficients"], dtype=np.float64)
    z = (row * direction - center) / scale
    logit = float(z @ coefficients + float(data["intercept"]))
    score = float(stable_sigmoid(np.asarray([logit]))[0])
    calibrated = None
    calibration = data.get("calibration")
    if calibration and calibration.get("method") == "platt":
        params = calibration.get("parameters") or {}
        calibrated = float(
            stable_sigmoid(
                np.asarray(
                    [float(params.get("a", 1.0)) * logit + float(params.get("b", 0.0))]
                )
            )[0]
        )
    return {"logit": logit, "score": score, "calibrated_probability": calibrated}


def runtime_to_portable(runtime: CompiledCriticReadout) -> str:
    payload = {
        "schema_version": PORTABLE_SCHEMA_VERSION,
        "critic_contract_id": runtime.contract_id,
        "feature_spec_id": runtime.feature_spec_id,
        "feature_ids": list(runtime.feature_ids),
        "directionality": list(runtime.directionality),
        "center": [float(v) for v in runtime.center],
        "scale": [float(v) for v in runtime.scale],
        "coefficients": [float(v) for v in runtime.coefficients],
        "intercept": float(runtime.intercept),
        "calibration": runtime.calibration,
        "score_contract": {"score": "sigmoid(logit)", "calibration_input": "logit"},
    }
    return canonical_json_bytes(payload).decode("utf-8")
