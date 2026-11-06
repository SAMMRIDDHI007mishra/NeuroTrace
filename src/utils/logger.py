"""
NeuroTrace Structured Logging
=============================
Provides formatted logging with timestamp, level, and caller module context.
"""

import logging
import sys

def setup_logger(name: str = "NeuroTrace", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a consistent logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
