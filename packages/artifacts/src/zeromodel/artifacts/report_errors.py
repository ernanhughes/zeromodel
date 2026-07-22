from __future__ import annotations

from zeromodel.core.artifact import VPMValidationError


class ReportAdaptationError(VPMValidationError):
    """Raised when an adapted report (or its contract) is malformed."""


class ReportCompilationError(VPMValidationError):
    """Raised when a well-formed adapted report cannot be compiled into a
    canonical artifact (e.g. a declared missing-value policy this compiler
    does not yet support, or a digest/kind mismatch on load)."""
