"""Centralized logging configuration for the project.

This module provides a configurable logger with console output,
rotating file logging, and consistent formatting across the application.
It is designed to be safely re-initialized with different configurations.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logger(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,      # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """Configure and return the main application logger.

    This function initializes a named logger with console output and
    optional rotating file logging. Existing handlers are cleared to
    prevent duplicate log entries when reconfiguring the logger.

    Args:
        level (str): Logging level
            ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_to_file (bool): If True, enables rotating file logging.
        log_dir (str): Directory where log files are stored.
        max_bytes (int): Maximum size (in bytes) of a single log file
            before rotation occurs.
        backup_count (int): Number of rotated backup log files to retain.

    Returns:
        logging.Logger: Configured logger instance.

    Side Effects:
        - Creates the log directory if it does not exist.
        - Writes logs to stdout and optionally to rotating log files.
        - Clears existing logger handlers before attaching new ones.

    Notes:
        - Logger name is fixed to "intelligent_test_clustering".
        - Log file name is "app.log".
        - Uses RotatingFileHandler for log rotation.
        - Safe to call multiple times (idempotent configuration).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("intelligent_test_clustering")
    logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_path / "app.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logger initialized with level: {level.upper()}")
    return logger


# Global logger instance
logger = setup_logger()