"""The adapter protocol - the ingress boundary for domain reports.

Concrete adapters live in the external application (e.g.
`writer.integrations.zeromodel.AIArtifactReportAdapter`), never in this
package. An adapter must not write directly to an `ArtifactStore`,
construct rendered images, or calculate artifact identity itself - it only
translates one typed domain report into a neutral `AdaptedReportDTO`;
`compile_report` (see `report_compiler.py`) owns validation and identity.
"""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar, runtime_checkable

from zeromodel.artifacts.report_dto import AdaptedReportDTO, ReportAdapterContractDTO

ReportT = TypeVar("ReportT", contravariant=True)


@runtime_checkable
class ReportAdapter(Protocol, Generic[ReportT]):
    """Domain-owned translation from one typed report into neutral ZeroModel form."""

    def contract(self) -> ReportAdapterContractDTO:
        """Return the stable contract governing this adapter's translation."""
        ...

    def adapt(self, report: ReportT) -> AdaptedReportDTO:
        """Convert one domain report into the neutral ZeroModel report form."""
        ...
