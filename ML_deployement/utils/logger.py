import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    max_size: int = 1024 * 1024,  # 1MB
    backup_count: int = 5
) -> logging.Logger:
    """Configure logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file specified
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def setup_detailed_logger(name: str) -> logging.Logger:
    """Setup detailed logger with both file and console output."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Use ASCII symbols instead of Unicode for Windows compatibility
    SUCCESS_SYMBOL = "+"
    FAILURE_SYMBOL = "x"
    
    class WindowsCompatibleFormatter(logging.Formatter):
        def format(self, record):
            # Replace Unicode symbols with ASCII
            if hasattr(record, 'msg'):
                record.msg = record.msg.replace('✓', SUCCESS_SYMBOL).replace('✗', FAILURE_SYMBOL)
            return super().format(record)
    
    # Detailed formatter
    formatter = WindowsCompatibleFormatter(
        '%(asctime)s - %(name)s - %(levelname)s\n'
        'File: %(filename)s - Line: %(lineno)d\n'
        'Message: %(message)s\n'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File Handler
    file_handler = RotatingFileHandler(
        log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'  # Ensure UTF-8 encoding for file output
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def log_request_response(logger: logging.Logger, method: str, url: str, 
                        request_data: dict = None, response_data: dict = None, 
                        status_code: int = None, error: Exception = None):
    """Log detailed request/response information."""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "url": url,
        "request": request_data,
        "response": response_data,
        "status_code": status_code,
        "error": str(error) if error else None
    }
    
    logger.info(
        f"\n{'='*50}\n"
        f"API {method} {url}\n"
        f"Request Data: {json.dumps(request_data, indent=2) if request_data else 'None'}\n"
        f"Response Status: {status_code}\n"
        f"Response Data: {json.dumps(response_data, indent=2) if response_data else 'None'}\n"
        f"Error: {error}\n"
        f"{'='*50}\n"
    )
    
    # Also log to file in JSON format for easier parsing
    logger.debug(json.dumps(log_data, indent=2))

def setup_request_logger():
    """Setup logger for API requests."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("request_logger")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler(
        log_dir / f"requests_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s\nRequest: %(request)s\nResponse: %(response)s\n'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

request_logger = setup_request_logger()
