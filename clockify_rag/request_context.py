"""Request context management for tracing requests across the pipeline.

Provides request ID (correlation ID) tracking using Python's contextvars module,
which is thread-safe and works with both sync and async code.

Usage:
    # Set request ID (in API/CLI entry point)
    from clockify_rag.request_context import set_request_id, get_request_id

    request_id = set_request_id()  # Generates new UUID4
    # OR
    set_request_id("custom-request-id")  # Use provided ID

    # Get request ID (in any downstream function)
    request_id = get_request_id()  # Returns None if not set

    # Use in logging
    logger.info(f"Processing query | request_id={get_request_id()}")
"""

import contextvars
import uuid
from typing import Optional

# Context variable for request ID (thread-safe)
_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)


def generate_request_id() -> str:
    """Generate a new request ID (UUID4).

    Returns:
        New UUID4 string (e.g., "a1b2c3d4-e5f6-7890-abcd-ef1234567890")
    """
    return str(uuid.uuid4())


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set request ID in current context.

    Args:
        request_id: Request ID to set. If None, generates a new UUID4.

    Returns:
        The request ID that was set

    Example:
        >>> req_id = set_request_id()
        >>> req_id
        'a1b2c3d4-e5f6-7890-abcd-ef1234567890'

        >>> set_request_id("my-custom-id")
        'my-custom-id'
    """
    if request_id is None:
        request_id = generate_request_id()
    _request_id_var.set(request_id)
    return request_id


def get_request_id() -> Optional[str]:
    """Get request ID from current context.

    Returns:
        Request ID if set, None otherwise

    Example:
        >>> set_request_id("test-123")
        'test-123'
        >>> get_request_id()
        'test-123'
    """
    return _request_id_var.get()


def clear_request_id() -> None:
    """Clear request ID from current context.

    Useful for cleanup in tests.
    """
    _request_id_var.set(None)


def format_request_id_for_log(prefix: str = "request_id") -> str:
    """Format request ID for structured logging.

    Args:
        prefix: Prefix for the log field (default: "request_id")

    Returns:
        Formatted string like "request_id=abc123" or empty string if no request ID

    Example:
        >>> set_request_id("test-123")
        'test-123'
        >>> format_request_id_for_log()
        'request_id=test-123'
        >>> clear_request_id()
        >>> format_request_id_for_log()
        ''
    """
    request_id = get_request_id()
    if request_id:
        return f"{prefix}={request_id}"
    return ""


def log_with_request_id(logger_instance, level: str, message: str, **kwargs) -> None:
    """Log message with request ID automatically included.

    Args:
        logger_instance: Logger instance (from logging.getLogger)
        level: Log level ("debug", "info", "warning", "error", "critical")
        message: Log message
        **kwargs: Additional key-value pairs to include in log

    Example:
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> set_request_id("test-123")
        >>> log_with_request_id(logger, "info", "Processing query", query="test")
        # Logs: "Processing query | request_id=test-123 | query=test"
    """
    request_id = get_request_id()

    # Build structured log message
    parts = [message]

    if request_id:
        parts.append(f"request_id={request_id}")

    for key, value in kwargs.items():
        parts.append(f"{key}={value}")

    formatted_message = " | ".join(parts)

    # Log at the specified level
    log_func = getattr(logger_instance, level.lower())
    log_func(formatted_message)
