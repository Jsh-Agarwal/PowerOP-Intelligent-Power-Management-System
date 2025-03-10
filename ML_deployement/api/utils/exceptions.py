class ModelError(Exception):
    """Exception raised for errors in the ML model."""
    pass

class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass

class ServiceError(Exception):
    """Exception raised for service errors."""
    pass

class DatabaseError(Exception):
    """Exception raised for database errors."""
    pass
