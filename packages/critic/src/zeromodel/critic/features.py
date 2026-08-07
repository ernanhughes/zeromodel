from __future__ import annotations

from typing import Mapping

import numpy as np

from zeromodel.critic.dto import CriticFeatureSpecDTO
from zeromodel.critic.linear import features_from_mapping


def build_feature_row(
    values: Mapping[str, float | None], spec: CriticFeatureSpecDTO
) -> np.ndarray:
    return features_from_mapping(values, spec)
