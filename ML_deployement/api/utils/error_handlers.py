from fastapi import HTTPException
from typing import Optional, Dict, Any
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

class APIError(HTTPException):
    def __init__(
        self, 
        status_code: int, 
        detail: str, 
        error_code: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code
        self.extra = extra or {}
        self.traceback = traceback.format_exc()
        
        error_detail = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": detail,
            "error_code": error_code or "UNKNOWN_ERROR",
            "traceback": self.traceback,
            **self.extra
        }
        
        super().__init__(
            status_code=status_code,
            detail=error_detail
        )

def handle_api_error(error: Exception, operation: str) -> APIError:
    """Convert any exception to APIError with detailed information"""
    logger.error(f"Error during {operation}: {str(error)}")
    logger.debug(traceback.format_exc())
    
    if isinstance(error, APIError):
        return error
        
    return APIError(
        status_code=500,
        detail=str(error),
        error_code=error.__class__.__name__,
        extra={
            "operation": operation,
            "error_type": error.__class__.__name__
        }
    )
