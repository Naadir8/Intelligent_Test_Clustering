"""Centralized logging and error tracking utilities.

This module provides:
- A configurable application-wide logger
- Standardized error logging with unique identifiers
- Optional contextual metadata for debugging

It is designed to ensure consistent logging behavior and improve
traceability of runtime errors.
"""

import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs"
) -> logging.Logger:
    """Configure and return the main application logger.

    Initializes a named logger with console output and optional file logging.
    Existing handlers are cleared to prevent duplicate log entries when
    reconfiguring the logger.

    Args:
        level (str): Logging level
            ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_to_file (bool): If True, enables file logging.
        log_dir (str): Directory where log files are stored.

    Returns:
        logging.Logger: Configured logger instance.

    Side Effects:
        - Creates the log directory if it does not exist.
        - Writes logs to stdout and optionally to a file.
        - Clears existing logger handlers before attaching new ones.

    Notes:
        - Logger name is fixed to "intelligent_test_clustering".
        - Log file name is "app.log".
        - Safe to call multiple times (idempotent configuration).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("intelligent_test_clustering")
    logger.setLevel(numeric_level)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_path / "app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logger()


def log_error(
    message: str,
    exc_info: bool = True,
    error_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> str:
    """Log an error with a unique identifier and optional context.

    This function standardizes error logging by attaching a unique error ID
    to each logged event. It optionally includes structured context data
    to aid in debugging and traceability.

    Args:
        message (str): Human-readable error description.
        exc_info (bool): If True, includes the full exception traceback.
        error_id (Optional[str]): Custom error identifier. If not provided,
            a unique ID is generated automatically.
        extra (Optional[Dict[str, Any]]): Additional contextual information
            (e.g., parameters, file paths, runtime state).

    Returns:
        str: The error identifier associated with this log entry.

    Side Effects:
        - Writes an error log entry via the global logger.
        - May include stack trace depending on `exc_info`.

    Notes:
        - Generated error IDs follow the format: "ERR-XXXXXXXX".
        - Context is serialized as a string and appended to the log message.
        - Designed for use in exception handling blocks.
    """
    if error_id is None:
        error_id = f"ERR-{uuid.uuid4().hex[:8].upper()}"

    context = f"[Error ID: {error_id}]"

    if extra:
        context += f" | Context: {extra}"

    logger.error(f"{context} {message}", exc_info=exc_info)
    return error_id