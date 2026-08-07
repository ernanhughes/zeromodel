from __future__ import annotations

from zeromodel.core.artifact import VPMValidationError


class CriticValidationError(VPMValidationError):
    """Raised when a Critic DTO or runtime input violates its contract."""


class CriticFeatureSchemaMismatchError(CriticValidationError):
    """Raised when feature schema identities or dimensions are incompatible."""


class CriticContractMismatchError(CriticValidationError):
    """Raised when critic contracts or label bindings disagree."""


class CriticReadoutIntegrityError(CriticValidationError):
    """Raised when a persisted critic aggregate fails closure validation."""


class CriticPayloadTooLargeError(CriticValidationError):
    """Raised when a portable executable payload exceeds its declared limit."""


class CriticCalibrationError(CriticValidationError):
    """Raised when calibration configuration is invalid."""


class CriticReplayMismatchError(CriticValidationError):
    """Raised when replay cannot reproduce the receipt-bound score result."""


class CriticEvaluationError(CriticValidationError):
    """Raised when evaluation inputs are invalid."""
