"""
Robust error handling for RAG and LLM operations.

This module provides retry decorators, safe wrapper functions, and error
handling utilities for LLM invocations and RAG search operations. It includes
support for exponential backoff, token tracking, and graceful degradation.

Key Components:
    - retry_on_failure: Decorator for automatic retry with exponential backoff
    - safe_rag_search: Safe wrapper for RAG search operations with fallback
    - safe_llm_invoke: Safe wrapper for LLM invocations with retry and tracking

Dependencies:
    - utils.token_tracker: Optional token usage tracking
    - utils.exceptions: Custom exception types
    - tools.patient_tools: Patient information retrieval

Example:
    >>> from utils.error_handling import retry_on_failure, safe_llm_invoke
    >>> @retry_on_failure(max_retries=3, delay=2.0)
    ... def my_function():
    ...     # Function will auto-retry on failure
    ...     pass
"""

import logging
import time
from typing import Callable, Any, Optional, List, Dict, Tuple
from functools import wraps

# Import custom exceptions
from .exceptions import GenerationError, RAGError, LLMError, ParseError

# Import token tracker for usage tracking
try:
    from .token_tracker import get_token_tracker
    TOKEN_TRACKING_AVAILABLE = True
except ImportError:
    TOKEN_TRACKING_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 2.0,
    exponential_backoff: bool = True,
    exceptions: Tuple[type, ...] = (Exception,)
) -> Callable:
    """
    Decorator to retry function on failure with exponential backoff.

    This decorator will automatically retry a function when it raises an exception,
    with configurable retry attempts, delay, and backoff strategy.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 2.0)
        exponential_backoff: Whether to exponentially increase delay (default: True)
        exceptions: Tuple of exception types to catch and retry (default: (Exception,))

    Returns:
        Decorated function with retry logic

    Raises:
        Last exception after all retry attempts are exhausted

    Example:
        >>> @retry_on_failure(max_retries=3, delay=1.0)
        ... def unreliable_function():
        ...     # May fail occasionally
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}: {str(e)}"
                        )
                        logger.info(f"Retrying in {current_delay:.1f} seconds...")
                        time.sleep(current_delay)

                        if exponential_backoff:
                            current_delay *= 2
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts",
                            exc_info=True
                        )

            # All retries exhausted
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def safe_rag_search(
    query: str,
    max_references: int = 3,
    note_types: Optional[List[str]] = None,
    fallback_result: Optional[Any] = None
) -> Any:
    """
    Safely perform RAG search with error handling and fallback.

    This function wraps RAG search operations with exception handling,
    providing graceful degradation when searches fail.

    Args:
        query: Search query string
        max_references: Maximum number of source references to retrieve (default: 3)
        note_types: Optional list of note types to filter by
        fallback_result: Result to return on complete failure (default: empty PatientInfoResult)

    Returns:
        PatientInfoResult with search results, or fallback result on failure

    Raises:
        RAGError: Only if fallback_result is None and search fails

    Example:
        >>> result = safe_rag_search("diabetes diagnosis", max_references=5)
        >>> print(result.content)
    """
    from tools.patient_tools import get_patient_info_with_sources, PatientInfoResult

    try:
        logger.debug(f"Performing RAG search for query: '{query[:100]}...'")
        result = get_patient_info_with_sources(
            query,
            max_references=max_references,
            note_types=note_types
        )
        logger.debug(f"RAG search successful, found {len(result.sources)} sources")
        return result

    except Exception as e:
        logger.error(
            f"RAG search failed for query '{query[:50]}...': {str(e)}",
            exc_info=True
        )

        # Return empty fallback result
        if fallback_result is None:
            fallback_result = PatientInfoResult(
                content="Ingen information kunne hentes fra patientjournal (teknisk fejl)",
                sources=[],
                max_references=0,
                search_method="failed"
            )

        return fallback_result


def safe_llm_invoke(
    prompt: str,
    llm_instance: Any,
    max_retries: int = 3,
    fallback_response: str = "LLM kunne ikke generere svar",
    operation: str = "unknown"
) -> str:
    """
    Safely invoke LLM with retry logic and token usage tracking.

    This function wraps LLM invocations with automatic retry on failure,
    exponential backoff, and optional token usage tracking.

    Args:
        prompt: LLM prompt string
        llm_instance: LLM instance to invoke (must have .invoke() method)
        max_retries: Maximum retry attempts on failure (default: 3)
        fallback_response: Response to return on complete failure (default: Danish error message)
        operation: Operation name for token tracking (e.g., "fact_answering", "validation")

    Returns:
        LLM response string, or fallback response on failure

    Raises:
        LLMError: Only if fallback_response is None and all retries fail

    Example:
        >>> from config.llm_config import get_llm
        >>> llm = get_llm()
        >>> response = safe_llm_invoke("Summarize patient history", llm, operation="summarization")
        >>> print(response)
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            logger.debug(
                f"LLM invocation attempt {attempt + 1}/{max_retries} for operation: {operation}"
            )

            response = llm_instance.invoke(
                prompt,
                tags=["safe_invoke", f"attempt_{attempt + 1}", operation]
            )

            # Track token usage if enabled
            if TOKEN_TRACKING_AVAILABLE:
                tracker = get_token_tracker()
                if (tracker.is_enabled() and
                    hasattr(llm_instance, 'last_usage') and
                    llm_instance.last_usage):

                    usage = llm_instance.last_usage
                    tracker.record(
                        prompt_tokens=usage.get('prompt_tokens', 0),
                        completion_tokens=usage.get('completion_tokens', 0),
                        total_tokens=usage.get('total_tokens', 0),
                        operation=operation,
                        model=getattr(llm_instance, 'model_name', 'unknown')
                    )
                    logger.debug(f"Token usage recorded: {usage.get('total_tokens', 0)} tokens")

            logger.debug(f"LLM invocation successful for operation: {operation}")
            return response

        except Exception as e:
            last_error = e
            logger.warning(
                f"LLM invocation failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )

            if attempt < max_retries - 1:
                backoff_delay = 2 ** attempt
                logger.info(f"Retrying in {backoff_delay} seconds...")
                time.sleep(backoff_delay)

    # All retries failed
    logger.error(
        f"LLM invocation failed after {max_retries} attempts for operation: {operation}. "
        f"Last error: {last_error}",
        exc_info=True
    )
    return fallback_response