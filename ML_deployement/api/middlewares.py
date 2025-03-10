import logging
import time
from fastapi import Request
import json

logger = logging.getLogger("api")

async def log_requests_middleware(request: Request, call_next):
    """Log request and response details."""
    start_time = time.time()
    
    # Extract request details
    request_details = {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "client": request.client.host if request.client else None,
    }
    
    # Log request
    logger.info(
        "Incoming request",
        extra={
            "request": request_details
        }
    )

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Extract response details
        response_details = {
            "status_code": response.status_code,
            "process_time": f"{process_time:.3f}s",
            "headers": dict(response.headers)
        }

        # Log response
        logger.info(
            "Request completed",
            extra={
                "request": {
                    "method": request.method,
                    "url": str(request.url)
                },
                "response": response_details
            }
        )

        return response
        
    except Exception as e:
        logger.error(
            f"Request failed: {str(e)}",
            extra={
                "request": request_details,
                "error": str(e)
            },
            exc_info=True
        )
        raise
