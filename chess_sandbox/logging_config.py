"""Centralized logging configuration for chess_sandbox."""

import logging
import sys


def setup_logging(name: str | None = None) -> logging.Logger:
    """
    Configure and return a logger with simple formatting.

    Uses StreamHandler to stderr with print-like formatting (module name only).

    Args:
        name: Logger name (typically __name__ from the calling module).
              If None, returns the root logger.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("%(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
