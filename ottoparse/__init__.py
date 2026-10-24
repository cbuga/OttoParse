"""OttoParse: Phase 1 intake & pre-processing for late-Ottoman printed documents.

Public API
----------
The common case is a single call::

    from ottoparse import Pipeline
    records = Pipeline().process("archive_scan.pdf")  # -> list[PageRecord]

Every stage is swappable; see :class:`Pipeline` and the ``intake`` / ``imaging``
sub-packages for the available strategies.
"""

from ottoparse.core.config import PipelineConfig
from ottoparse.model.flags import FlagCode, IssueFlag, Severity
from ottoparse.model.record import (
    PageRecord,
    ProcessingStep,
    Provenance,
    SourcePath,
)
from ottoparse.pipeline import Pipeline, default_image_steps

__version__ = "0.2.0"

__all__ = [
    "Pipeline",
    "default_image_steps",
    "PipelineConfig",
    "PageRecord",
    "Provenance",
    "ProcessingStep",
    "SourcePath",
    "IssueFlag",
    "FlagCode",
    "Severity",
    "__version__",
]
