"""Centralized logging, error handling, and localization utilities.

This module provides:
- Configurable application-wide logger
- Rotating file logging for production use
- Localized error messages (i18n support)
- Standardized error logging with unique identifiers

It ensures consistent logging behavior and improves traceability
and user-facing error communication.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

# Simple localization dictionary (can be extended to external files later)
MESSAGES = {
    "en": {
        "critical_error": "A critical error occurred. Please check the logs for details.",
        "file_not_found": "Required file not found. Please check if the data directory exists.",
        "embedding_failed": "Failed to generate embeddings. Check model availability and memory.",
        "clustering_failed": "Clustering process failed. Please try again with different parameters."
    },
    "uk": {
        "critical_error": "Виникла критична помилка. Перевірте логи для деталей.",
        "file_not_found": "Не знайдено необхідний файл. Перевірте наявність папки data.",
        "embedding_failed": "Не вдалося згенерувати ембедінги. Перевірте модель та пам'ять.",
        "clustering_failed": "Процес кластеризації завершився помилкою. Спробуйте інші параметри."
    }
}


def setup_logger(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> logging.Logger:
    """Configure and return the main application logger.

    Initializes a named logger with console output and optional rotating
    file logging. Existing handlers are cleared to prevent duplicate logs.

    Args:
        level (str): Logging level
            ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_to_file (bool): If True, enables rotating file logging.
        log_dir (str): Directory where log files are stored.
        max_bytes (int): Maximum size (in bytes) of a log file before rotation.
        backup_count (int): Number of rotated backup files to retain.

    Returns:
        logging.Logger: Configured logger instance.

    Side Effects:
        - Creates the log directory if it does not exist.
        - Writes logs to stdout and rotating log files.
        - Clears existing handlers before reconfiguration.

    Notes:
        - Logger name is fixed to "intelligent_test_clustering".
        - Log file name is "app.log".
        - Uses size-based rotation via RotatingFileHandler.
        - Safe to call multiple times (idempotent).
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

        file_handler = RotatingFileHandler(
            filename=log_path / "app.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()


def log_error(
    message_key: str,
    lang: str = "en",
    exc_info: bool = True,
    extra: Optional[Dict[str, Any]] = None
) -> str:
    """Log an error with localization and contextual metadata.

    Retrieves a localized error message based on a message key and language,
    attaches a unique error ID, and logs the error with optional context.

    Args:
        message_key (str): Key from the MESSAGES dictionary.
        lang (str): Language code ("en" or "uk").
        exc_info (bool): If True, includes full exception traceback.
        extra (Optional[Dict[str, Any]]): Additional context
            (e.g., file paths, parameters, runtime state).

    Returns:
        str: Generated unique error identifier (e.g., "ERR-XXXXXXXX").

    Side Effects:
        - Writes an error entry to the global logger.
        - May include stack trace depending on `exc_info`.

    Notes:
        - Falls back to English if language or key is not found.
        - Context is appended as a string to the log message.
        - Intended for use in exception handling blocks.
    """
    message = MESSAGES.get(lang, MESSAGES["en"]).get(
        message_key, "An unknown error occurred."
    )

    error_id = f"ERR-{__import__('uuid').uuid4().hex[:8].upper()}"

    context = f"[Error ID: {error_id}]"

    if extra:
        context += f" | Context: {extra}"

    logger.error(f"{context} {message}", exc_info=exc_info)
    return error_id