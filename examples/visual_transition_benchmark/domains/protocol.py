"""Domain-neutral seam (stage 3, cross-domain replication).

This module must never import anything from ``domains/arcade`` or
``domains/warehouse``, and must never mention a component name specific to
either domain (no "tank", "robot", "cooldown", "battery", ...). That is the
test: if this file needs a domain-specific name to work, the seam has failed.

``DomainTransition``'s field names deliberately mirror
``visual_transition_benchmark.dataset.TransitionRecord`` so that stage 1's
existing, already domain-agnostic presence-level metrics
(``metrics.component_multilabel_metrics``, ``metrics.unexpected_change_summary``,
``metrics.missing_change_summary``, ``metrics.false_implicated_components``,
``metrics.field_precision_recall``) run **unchanged** against transitions from
either domain -- they were already only using component-name strings, never
arcade-specific identifiers.

Value-level ground truth and decoded values use one small, shared vocabulary
(below) instead of domain-specific field names, so ``cross_domain_metrics.py``
can score "direction", "magnitude", "value", "relation", and "identity"
capability classes identically for both domains without knowing what a tank
or a crate is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, Tuple

import numpy as np

# Shared value-ground-truth / decoded-value vocabulary. A domain sets a key
# only when that capability class applies to a given transition; an absent
# key means "not applicable to this transition", not "wrong".
#   direction_expected_sign / direction_decoded_sign   : int in {-1, 0, 1}
#   magnitude_expected_delta / magnitude_decoded_delta : int (exact)
#   value_expected_level / value_decoded_level         : str (categorical level)
#   relation_expected_satisfied / relation_decoded_satisfied : bool
#   identity_expected_id / identity_decoded_id         : str


@dataclass(frozen=True)
class ComponentSchema:
    domain_name: str
    component_names: Tuple[str, ...]
    canvas_shape: Tuple[int, int]  # (height, width)


@dataclass(frozen=True)
class TransitionContract:
    contract_id: str
    component: str
    kind: str  # "presence_stable" | "presence_change" | "direction" | "magnitude" | "value" | "relation" | "identity"
    description: str


@dataclass(frozen=True)
class DomainTransition:
    transition_id: str
    domain_name: str
    episode_id: str
    step_number: int
    seed: int
    action: str
    category: str
    frame_before: np.ndarray
    frame_after: np.ndarray
    expected_changed_components: Tuple[str, ...]
    observed_changed_components: Tuple[str, ...]
    fault_type: Optional[str]
    is_faulty: bool
    expected_contracts: Tuple[TransitionContract, ...]
    value_ground_truth: Mapping[str, object] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class AnalysisMetadata:
    transition_id: str
    step_number: int


@dataclass(frozen=True)
class ComponentAnalysisResult:
    predicted_region_mask: np.ndarray
    predicted_fields: Tuple[str, ...]
    predicted_components: Tuple[str, ...]
    expected_components: Tuple[str, ...]
    unexpected_components: Tuple[str, ...]
    missing_components: Tuple[str, ...]
    evidence_scores: Mapping[str, float]
    diagnostics: Mapping[str, object]


class ComponentAnalyzer(Protocol):
    def analyze(
        self, frame_before: np.ndarray, frame_after: np.ndarray, action: str, metadata: AnalysisMetadata
    ) -> ComponentAnalysisResult: ...


@dataclass(frozen=True)
class ValueAnalysisResult:
    decoded: Mapping[str, object]
    value_flags: Tuple[str, ...]
    diagnostics: Mapping[str, object]


class ValueAnalyzer(Protocol):
    def analyze(
        self, frame_before: np.ndarray, frame_after: np.ndarray, action: str, metadata: AnalysisMetadata
    ) -> ValueAnalysisResult: ...


class VisualTransitionDomain(Protocol):
    name: str

    def generate_episode(self, *, seed: int, episode_id: str) -> Tuple[DomainTransition, ...]: ...

    def render(self, state: object) -> np.ndarray: ...

    def component_schema(self) -> ComponentSchema: ...

    def contracts_for_action(self, action: str) -> Tuple[TransitionContract, ...]: ...

    def build_component_analyzer(self) -> ComponentAnalyzer: ...

    def build_value_analyzer(self) -> ValueAnalyzer: ...
