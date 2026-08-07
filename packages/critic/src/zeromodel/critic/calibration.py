from __future__ import annotations

import numpy as np

from zeromodel.critic.dto import CriticCalibrationDTO
from zeromodel.critic.linear import CompiledCriticReadout, stable_sigmoid


def fit_platt_calibration(
    runtime: CompiledCriticReadout,
    features: object,
    labels: object,
    *,
    feature_spec_id: str,
    calibration_set_ref: object,
    max_iterations: int = 50,
) -> CriticCalibrationDTO:
    logits = runtime.logit_many(features, feature_spec_id=feature_spec_id)
    y = np.asarray(labels, dtype=np.float64)
    x = logits[:, None]
    design = np.column_stack([x, np.ones_like(logits)])
    beta = np.asarray([1.0, 0.0], dtype=np.float64)
    for _ in range(max_iterations):
        p = stable_sigmoid(design @ beta)
        w = np.maximum(p * (1.0 - p), 1e-12)
        gradient = design.T @ (p - y)
        hessian = (design.T * w) @ design
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta -= step
        if np.max(np.abs(step)) <= 1e-8:
            break
    return CriticCalibrationDTO(
        method="platt",
        parameters={"a": float(beta[0]), "b": float(beta[1]), "input": "logit"},
        calibration_set_ref=calibration_set_ref,  # type: ignore[arg-type]
        metrics={},
    )
