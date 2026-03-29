class DataPipelineError(Exception):
    """Base exception for all pipeline-related errors."""
    pass

class DataIngestionError(DataPipelineError):
    """Raised when data retrieval from source fails."""
    pass

class DataValidationError(DataPipelineError):
    """Raised when data fails schema or quality checks."""
    pass

class ModelTrainingError(DataPipelineError):
    """Raised when ML model training fails or is non-convergent."""
    pass
