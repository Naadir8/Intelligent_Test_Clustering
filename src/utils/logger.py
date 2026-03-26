"""Logging configuration module.

Provides a centralized logger setup for the entire application,
including console and optional file logging with consistent formatting.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs"
) -> logging.Logger:
    """Configure and return the main application logger.

    This function initializes a logger with a unified format and attaches
    handlers for console output and optionally file logging. Existing handlers
    are cleared to prevent duplicate log entries when the function is called
    multiple times.

    Args:
        level (str): Logging level as a string
            (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_to_file (bool): If True, logs will also be written to a file.
        log_dir (str): Directory where log files will be stored.

    Returns:
        logging.Logger: Configured logger instance.

    Side Effects:
        - Creates a directory for log files if it does not exist.
        - Writes logs to stdout and optionally to a file.
        - Clears existing logger handlers to avoid duplication.

    Notes:
        - The logger name is fixed to "intelligent_test_clustering".
        - Log file name is "app.log".
        - Safe to call multiple times due to handler reset.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("intelligent_test_clustering")
    logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Formatter: time | level | name | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_path / "app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logger()