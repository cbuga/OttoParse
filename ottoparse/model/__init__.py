"""Phase 1 data model: the hand-off contract to Phase 2."""

from ottoparse.model.flags import FlagCode, IssueFlag, Severity
from ottoparse.model.record import (
    PageRecord,
    ProcessingStep,
    Provenance,
    SourcePath,
)

__all__ = [
    "FlagCode",
    "IssueFlag",
    "Severity",
    "PageRecord",
    "ProcessingStep",
    "Provenance",
    "SourcePath",
]
