from ..utils.exceptions import HVACSystemError

class GroqServiceError(HVACSystemError):
    """Base exception for Groq SLM service errors."""
    pass

class GroqAPIError(GroqServiceError):
    """Raised when Groq API request fails."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"Groq API error ({status_code}): {message}")

class GroqRateLimitError(GroqServiceError):
    """Raised when API rate limit is exceeded."""
    def __init__(self, reset_time: int = 60):
        self.reset_time = reset_time
        super().__init__(f"Rate limit exceeded. Retry after {reset_time} seconds")

class GroqContextError(GroqServiceError):
    """Raised when context generation fails."""
    pass

class GroqResponseParsingError(GroqServiceError):
    """Raised when response parsing fails."""
    pass
