from __future__ import annotations

from zeromodel.core.artifact import VPMValidationError


class HierarchyCompilationError(VPMValidationError):
    """Raised when source artifacts cannot be compiled into a hierarchy."""


class HierarchyClosureError(VPMValidationError):
    """Raised when a hierarchy fails structural closure validation."""
