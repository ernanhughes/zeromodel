from __future__ import annotations

from zeromodel.core.artifact import VPMValidationError


class SearchValidationError(VPMValidationError):
    """Raised when a Search DTO or runtime input violates its contract."""


class RepresentationMismatchError(SearchValidationError):
    """Raised when representation identities or dimensions are incompatible."""


class RelationContractMismatchError(SearchValidationError):
    """Raised when relation contracts, coordinates, or dimensions disagree."""


class RelationReadoutIntegrityError(SearchValidationError):
    """Raised when a persisted readout aggregate fails closure validation."""


class SearchReplayMismatchError(SearchValidationError):
    """Raised when replay cannot reproduce the receipt-bound result."""


class CandidateGenerationError(SearchValidationError):
    """Raised for unsupported or invalid candidate generation configuration."""
