import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logging(console_level=logging.WARNING, file_level=logging.DEBUG):
    """Configure logging with minimal console output and detailed file logs."""
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create a timestamp-based log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"api_test_{timestamp}.log"

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s\n'
        '%(data)s' if '%(data)s' else '',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(message)s'  # Simplified console output
    )

    # File handler with custom filter
    class ExtraFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, 'data'):
                record.data = ''
            return True

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(ExtraFilter())

    # Console handler for minimal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(min(console_level, file_level))
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return log_file
