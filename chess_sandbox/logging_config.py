"""Centralized logging configuration for chess_sandbox."""

import logging
import sys

# Global flag to ensure handler is only initialized once
_LOGGING_CONFIGURED = False


def setup_logging(name: str | None = None) -> logging.Logger:
    """
    Configure and return a logger with simple formatting.

    Uses StreamHandler to stderr with print-like formatting (module name only).
    Handler is configured once globally on first call.

    Args:
        name: Logger name (typically __name__ from the calling module).
              If None, returns the root logger.

    Returns:
        Configured logger instance.
    """
    global _LOGGING_CONFIGURED

    # Configure handler once globally
    if not _LOGGING_CONFIGURED:
        # Get the root logger for chess_sandbox package
        root_logger = logging.getLogger("chess_sandbox")
        root_logger.setLevel(logging.INFO)

        # Add handler to root logger
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("%(name)s: %(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

        # Prevent propagation to avoid duplicate logs
        root_logger.propagate = False

        _LOGGING_CONFIGURED = True

    # Return child logger that inherits configuration
    return logging.getLogger(name) if name else logging.getLogger("chess_sandbox")
