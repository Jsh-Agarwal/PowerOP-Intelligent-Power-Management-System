import logging
from datetime import datetime
import sys

def setup_logger():
    logger = logging.getLogger("api")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(f"logs/api_{datetime.now().strftime('%Y%m%d')}.log")
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s\n'
        'File: %(filename)s - Line: %(lineno)d\n'
        'Message: %(message)s\n'
    )
    
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logger()
request_logger = setup_logger()
