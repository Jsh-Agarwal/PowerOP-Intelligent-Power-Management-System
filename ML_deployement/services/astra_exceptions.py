from ..utils.exceptions import HVACSystemError

class AstraDBError(HVACSystemError):
    """Base exception for Astra DB errors."""
    pass

class AstraConnectionError(AstraDBError):
    """Raised when connection to Astra DB fails."""
    def __init__(self, message: str = "Failed to connect to Astra DB"):
        super().__init__(message)

class AstraQueryError(AstraDBError):
    """Raised when a query fails."""
    def __init__(self, query: str, error: Exception):
        self.query = query
        self.error = error
        super().__init__(f"Query failed: {query}. Error: {str(error)}")

class AstraBatchError(AstraDBError):
    """Raised when a batch operation fails."""
    def __init__(self, error: Exception):
        self.error = error
        super().__init__(f"Batch operation failed: {str(error)}")

class AstraModelError(AstraDBError):
    """Raised when data model validation fails."""
    def __init__(self, model: str, field: str, value: str):
        self.model = model
        self.field = field
        self.value = value
        super().__init__(f"Invalid {field} '{value}' for model {model}")
