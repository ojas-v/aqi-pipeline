"""Custom exception hierarchy.

Why: catching one base class (``PipelineError``) in the orchestrator lets us
fail gracefully with a clear message, while still allowing callers to catch
specific failure modes (e.g. only schema problems) when they need to.
"""


class PipelineError(Exception):
    """Base class for every error raised intentionally by this project."""


class DataIngestionError(PipelineError):
    """Raised when the raw file is missing, unreadable, empty or malformed."""


class SchemaValidationError(PipelineError):
    """Raised when the ingested data does not match the expected schema."""


class ModelTrainingError(PipelineError):
    """Raised when model selection, fitting or evaluation fails."""
