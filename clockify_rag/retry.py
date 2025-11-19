"""Retry logic with exponential backoff for transient failures.

Provides decorators and utilities for retrying operations that may fail
due to transient errors (network issues, temporary unavailability, etc.).

Features:
- Exponential backoff with jitter
- Configurable retry attempts and delays
- Retry on specific exception types
- Request ID logging for traceability
- Thread-safe

Usage:
    from clockify_rag.retry import retry_with_backoff

    @retry_with_backoff(max_attempts=3, base_delay=1.0, max_delay=10.0)
    def fetch_embeddings(text):
        # May raise transient errors
        return api_client.embed(text)
"""

import functools
import logging
import random
import time
from typing import Callable, Type, Tuple, Optional, Any

from .request_context import get_request_id

logger = logging.getLogger(__name__)


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> float:
    """Calculate exponential backoff delay with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exponential_base: Base for exponential calculation (default: 2.0)
        jitter: Add random jitter to prevent thundering herd (default: True)

    Returns:
        Delay in seconds before next retry

    Example:
        >>> exponential_backoff(0, base_delay=1.0)  # First retry
        ~1.0 seconds (with jitter: 0.5-1.5s)
        >>> exponential_backoff(1, base_delay=1.0)  # Second retry
        ~2.0 seconds (with jitter: 1.0-3.0s)
        >>> exponential_backoff(2, base_delay=1.0)  # Third retry
        ~4.0 seconds (with jitter: 2.0-6.0s)
    """
    # Calculate exponential delay: base_delay * (exponential_base ^ attempt)
    delay = min(base_delay * (exponential_base ** attempt), max_delay)

    # Add jitter to prevent thundering herd problem
    # Jitter range: [delay * 0.5, delay * 1.5]
    if jitter:
        jitter_range = delay * 0.5
        delay = delay + random.uniform(-jitter_range, jitter_range)
        # Ensure delay is non-negative
        delay = max(0.1, delay)

    return delay


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
):
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including initial) (default: 3)
        base_delay: Base delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        retry_on: Tuple of exception types to retry on (default: all exceptions)
        on_retry: Optional callback function called before each retry
                  Signature: (exception, attempt, delay) -> None

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_with_backoff(max_attempts=3, base_delay=1.0)
        ... def fetch_data():
        ...     return api.get("/data")

        >>> # Custom retry callback
        >>> def log_retry(exc, attempt, delay):
        ...     print(f"Retry {attempt} after {delay}s due to {exc}")
        >>>
        >>> @retry_with_backoff(max_attempts=5, on_retry=log_retry)
        ... def fetch_with_logging():
        ...     return api.get("/data")
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            request_id = get_request_id()
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    # Attempt the function call
                    result = func(*args, **kwargs)

                    # Log success after retry
                    if attempt > 0:
                        logger.info(
                            f"Retry succeeded | request_id={request_id} | "
                            f"function={func.__name__} | attempt={attempt + 1}/{max_attempts}"
                        )

                    return result

                except retry_on as exc:
                    last_exception = exc

                    # Check if we should retry
                    if attempt + 1 >= max_attempts:
                        # No more retries, raise the exception
                        logger.error(
                            f"Retry exhausted | request_id={request_id} | "
                            f"function={func.__name__} | attempts={max_attempts} | error={exc}"
                        )
                        raise

                    # Calculate backoff delay
                    delay = exponential_backoff(
                        attempt,
                        base_delay=base_delay,
                        max_delay=max_delay,
                        exponential_base=exponential_base,
                        jitter=True,
                    )

                    # Log retry attempt
                    logger.warning(
                        f"Retry scheduled | request_id={request_id} | "
                        f"function={func.__name__} | attempt={attempt + 1}/{max_attempts} | "
                        f"delay={delay:.2f}s | error={str(exc)[:100]}"
                    )

                    # Call custom retry callback if provided
                    if on_retry:
                        try:
                            on_retry(exc, attempt + 1, delay)
                        except Exception as callback_exc:
                            logger.warning(
                                f"Retry callback failed | request_id={request_id} | "
                                f"error={callback_exc}"
                            )

                    # Wait before retrying
                    time.sleep(delay)

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Retry logic error in {func.__name__}")

        return wrapper

    return decorator


def retry_with_fallback(
    primary_func: Callable,
    fallback_func: Callable,
    max_attempts: int = 2,
    base_delay: float = 1.0,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """Execute a function with retry, then fall back to alternative if all retries fail.

    Args:
        primary_func: Primary function to execute
        fallback_func: Fallback function to execute if primary fails
        max_attempts: Maximum attempts for primary function (default: 2)
        base_delay: Base delay for primary function retries (default: 1.0)
        retry_on: Exception types to retry on (default: all exceptions)

    Returns:
        Result from primary_func (if successful) or fallback_func (if primary fails)

    Example:
        >>> def fetch_from_primary():
        ...     return primary_api.get("/data")
        >>>
        >>> def fetch_from_fallback():
        ...     return fallback_api.get("/data")
        >>>
        >>> result = retry_with_fallback(
        ...     fetch_from_primary,
        ...     fetch_from_fallback,
        ...     max_attempts=3
        ... )
    """
    request_id = get_request_id()

    # Try primary function with retries
    @retry_with_backoff(max_attempts=max_attempts, base_delay=base_delay, retry_on=retry_on)
    def _primary_with_retry():
        return primary_func()

    try:
        return _primary_with_retry()
    except retry_on as primary_exc:
        # Primary failed, try fallback
        logger.warning(
            f"Primary function failed, using fallback | request_id={request_id} | "
            f"primary={primary_func.__name__} | fallback={fallback_func.__name__} | "
            f"error={str(primary_exc)[:100]}"
        )

        try:
            result = fallback_func()
            logger.info(
                f"Fallback succeeded | request_id={request_id} | "
                f"fallback={fallback_func.__name__}"
            )
            return result
        except Exception as fallback_exc:
            logger.error(
                f"Fallback also failed | request_id={request_id} | "
                f"fallback={fallback_func.__name__} | error={fallback_exc}"
            )
            # Raise the fallback exception with context
            raise fallback_exc from primary_exc


class RetryConfig:
    """Configuration for retry behavior.

    Provides centralized retry configuration that can be adjusted per operation type.
    """

    # Embedding API retries (network-dependent)
    EMBEDDING_MAX_ATTEMPTS = 3
    EMBEDDING_BASE_DELAY = 1.0
    EMBEDDING_MAX_DELAY = 10.0

    # LLM API retries (may take longer)
    LLM_MAX_ATTEMPTS = 2
    LLM_BASE_DELAY = 2.0
    LLM_MAX_DELAY = 30.0

    # FAISS operations (fast, few retries needed)
    FAISS_MAX_ATTEMPTS = 2
    FAISS_BASE_DELAY = 0.5
    FAISS_MAX_DELAY = 5.0

    # Index build operations (expensive, minimal retries)
    INDEX_BUILD_MAX_ATTEMPTS = 1  # No retry (too expensive)
    INDEX_BUILD_BASE_DELAY = 0.0
    INDEX_BUILD_MAX_DELAY = 0.0

    # Network operations (general)
    NETWORK_MAX_ATTEMPTS = 3
    NETWORK_BASE_DELAY = 1.0
    NETWORK_MAX_DELAY = 15.0

    @classmethod
    def update_from_env(cls):
        """Update retry configuration from environment variables.

        Environment variables:
        - RAG_RETRY_EMBEDDING_MAX_ATTEMPTS
        - RAG_RETRY_LLM_MAX_ATTEMPTS
        - RAG_RETRY_NETWORK_MAX_ATTEMPTS
        etc.
        """
        import os

        def _get_int_env(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, default))
            except ValueError:
                logger.warning(f"Invalid {key} value, using default: {default}")
                return default

        def _get_float_env(key: str, default: float) -> float:
            try:
                return float(os.environ.get(key, default))
            except ValueError:
                logger.warning(f"Invalid {key} value, using default: {default}")
                return default

        cls.EMBEDDING_MAX_ATTEMPTS = _get_int_env(
            "RAG_RETRY_EMBEDDING_MAX_ATTEMPTS", cls.EMBEDDING_MAX_ATTEMPTS
        )
        cls.LLM_MAX_ATTEMPTS = _get_int_env(
            "RAG_RETRY_LLM_MAX_ATTEMPTS", cls.LLM_MAX_ATTEMPTS
        )
        cls.NETWORK_MAX_ATTEMPTS = _get_int_env(
            "RAG_RETRY_NETWORK_MAX_ATTEMPTS", cls.NETWORK_MAX_ATTEMPTS
        )

        cls.EMBEDDING_BASE_DELAY = _get_float_env(
            "RAG_RETRY_EMBEDDING_BASE_DELAY", cls.EMBEDDING_BASE_DELAY
        )
        cls.LLM_BASE_DELAY = _get_float_env(
            "RAG_RETRY_LLM_BASE_DELAY", cls.LLM_BASE_DELAY
        )


# Initialize from environment on module import
RetryConfig.update_from_env()
