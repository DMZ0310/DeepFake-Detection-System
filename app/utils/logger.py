"""
DeepFake Detection System - Logging Utility
=============================================
Centralized logging configuration.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str = None, level: str = "INFO") -> logging.Logger:
    """
    Create and configure a logger.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional file path for log output
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:  # Avoid duplicate handlers
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    
    return logger
