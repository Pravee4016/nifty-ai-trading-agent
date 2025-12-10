"""
Centralized Logging Configuration
"""
import logging
import os
import sys
from config.settings import LOG_DIR, DEBUG_MODE

def setup_logging(name: str = __name__) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.
    """
    log_format = "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
    log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
    
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "trading_agent.log")
    
    # Get the root logger or custom logger
    logger = logging.getLogger() # Root logger
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.setLevel(log_level)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
        
        # File Handler
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    # Return logger for the specific module
    return logging.getLogger(name)
