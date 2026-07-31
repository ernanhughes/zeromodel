"""Benchmark-owned static visual address adjudication experiment."""

from visual_transition_benchmark.adjudication.adjudicator import (
    AddressTransitionAdjudicationResult,
    CandidateAdjudicationResult,
    RuntimeAdjudicationInput,
    adjudicate_address_transition,
)
from visual_transition_benchmark.adjudication.corpus import (
    AddressAliasCase,
    build_case_corpus,
)

__all__ = [
    "AddressAliasCase",
    "AddressTransitionAdjudicationResult",
    "CandidateAdjudicationResult",
    "RuntimeAdjudicationInput",
    "adjudicate_address_transition",
    "build_case_corpus",
]
