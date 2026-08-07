from __future__ import annotations

import json

import numpy as np
import pytest

from zeromodel.core.artifact import VPMValidationError
from zeromodel.critic import (
    CriticFeatureDTO,
    CriticFeatureSpecDTO,
    CompiledCriticReadout,
)
from zeromodel.critic.dto import canonical_dto_bytes
from zeromodel.critic.linear import features_from_mapping, stable_sigmoid


def test_feature_spec_identity_round_trip_and_mutation() -> None:
    spec = CriticFeatureSpecDTO(
        features=(
            CriticFeatureDTO("a", "A"),
            CriticFeatureDTO(
                "b",
                "B",
                directionality=-1,
                required=False,
                missing_policy="constant",
                missing_value=0.0,
            ),
        )
    )
    clone = CriticFeatureSpecDTO.from_dict(json.loads(canonical_dto_bytes(spec)))
    assert clone.feature_spec_id == spec.feature_spec_id
    changed_order = CriticFeatureSpecDTO(features=tuple(reversed(spec.features)))
    assert changed_order.feature_spec_id != spec.feature_spec_id
    changed_direction = CriticFeatureSpecDTO(
        features=(CriticFeatureDTO("a", "A", directionality=-1), spec.features[1])
    )
    assert changed_direction.feature_spec_id != spec.feature_spec_id


def test_invalid_supplied_id_fails() -> None:
    spec = CriticFeatureSpecDTO(features=(CriticFeatureDTO("a", "A"),))
    data = spec.to_dict()
    data["feature_spec_id"] = "wrong"
    with pytest.raises(VPMValidationError):
        CriticFeatureSpecDTO.from_dict(data)


def test_missing_feature_policy() -> None:
    spec = CriticFeatureSpecDTO(
        features=(
            CriticFeatureDTO("required", "required"),
            CriticFeatureDTO(
                "optional",
                "optional",
                required=False,
                missing_policy="constant",
                missing_value=2.0,
            ),
        )
    )
    assert features_from_mapping({"required": 1.0}, spec).tolist() == [1.0, 2.0]
    with pytest.raises(VPMValidationError):
        features_from_mapping({"optional": 1.0}, spec)


def test_logistic_fit_is_deterministic_and_explanations_sum(critic_fixture) -> None:
    spec = critic_fixture["spec"]
    contract = critic_fixture["contract"]
    values = critic_fixture["values"]
    labels = critic_fixture["labels_array"]
    first = CompiledCriticReadout.fit(
        values,
        labels,
        feature_spec=spec,
        contract_id=contract.critic_contract_id,
        l2_penalty=0.1,
        max_iterations=80,
        tolerance=1e-8,
        class_weighting="balanced",
    )
    second = CompiledCriticReadout.fit(
        values,
        labels,
        feature_spec=spec,
        contract_id=contract.critic_contract_id,
        l2_penalty=0.1,
        max_iterations=80,
        tolerance=1e-8,
        class_weighting="balanced",
    )
    np.testing.assert_allclose(first.coefficients, second.coefficients)
    assert first.score_one(
        values[0], feature_spec_id=spec.feature_spec_id
    ) > first.score_one(values[3], feature_spec_id=spec.feature_spec_id)
    contributions = first.contributions_one(
        values[0], feature_spec_id=spec.feature_spec_id
    )
    assert sum(
        item.contribution for item in contributions
    ) + first.intercept == pytest.approx(
        first.logit_one(values[0], feature_spec_id=spec.feature_spec_id)
    )


def test_stable_extreme_sigmoid() -> None:
    scores = stable_sigmoid(np.asarray([-1000.0, 0.0, 1000.0]))
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(1.0)
