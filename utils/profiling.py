"""
Performance monitoring utilities for medical document generation.

This module provides decorators and utilities for profiling function execution,
including timing measurements and memory profiling. Useful for identifying
performance bottlenecks and optimization opportunities.

Key Functions:
    - profile: Decorator for timing function execution
    - profile_with_memory: Decorator for timing and memory profiling

Dependencies:
    - utils.exceptions: Custom exception types

Example:
    >>> from utils.profiling import profile
    >>> @profile
    ... def expensive_function():
    ...     # Long-running operation
    ...     pass
    >>> expensive_function()
    # Logs: "expensive_function took 2.34s"
"""

import logging
import time
import tracemalloc
from functools import wraps
from typing import Callable, Any, TypeVar, cast

from .exceptions import ProfilingError

# Configure module logger
logger = logging.getLogger(__name__)

# Type variable for generic function typing
F = TypeVar('F', bound=Callable[..., Any])


def profile(func: F) -> F:
    """
    Decorator that measures and logs execution time of the wrapped function.

    Measures wall-clock time for function execution and logs results.
    Useful for performance monitoring and optimization.

    Args:
        func: The function to be profiled

    Returns:
        Wrapped function with profiling capabilities

    Example:
        >>> @profile
        ... def process_data(data):
        ...     # Processing logic
        ...     return result
        >>> result = process_data(my_data)
        # Logs: "process_data took 1.23s"
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug(f"Starting profiling for {func.__name__}")

        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time

            logger.info(
                f"PROFILE: {func.__name__} completed in {duration:.2f}s"
            )

            return result

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            logger.error(
                f"PROFILE: {func.__name__} failed after {duration:.2f}s: {e}"
            )
            raise

    return cast(F, wrapper)


def profile_with_memory(func: F) -> F:
    """
    Decorator that measures execution time and memory usage.

    Measures both wall-clock time and memory allocation for the wrapped
    function. Provides detailed performance metrics including peak memory usage.

    Args:
        func: The function to be profiled

    Returns:
        Wrapped function with memory profiling capabilities

    Raises:
        ProfilingError: If tracemalloc is not available or fails

    Example:
        >>> @profile_with_memory
        ... def load_large_dataset():
        ...     data = load_data()
        ...     return process(data)
        >>> result = load_large_dataset()
        # Logs: "load_large_dataset took 3.45s, peak memory: 256.7 MB"
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug(f"Starting memory profiling for {func.__name__}")

        try:
            # Start memory tracking
            tracemalloc.start()
            start_time = time.time()

            # Execute function
            result = func(*args, **kwargs)

            # Stop timing and get memory stats
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            duration = end_time - start_time
            peak_mb = peak / 1024 / 1024

            logger.info(
                f"PROFILE: {func.__name__} completed in {duration:.2f}s, "
                f"peak memory: {peak_mb:.2f} MB"
            )

            return result

        except Exception as e:
            # Clean up tracemalloc on error
            try:
                tracemalloc.stop()
            except:
                pass

            logger.error(
                f"PROFILE: {func.__name__} failed during memory profiling: {e}"
            )
            raise ProfilingError(
                f"Memory profiling failed for {func.__name__}",
                details=str(e)
            )

    return cast(F, wrapper)


def log_execution_time(operation_name: str) -> Callable[[F], F]:
    """
    Parameterized decorator for logging execution time with custom operation name.

    Useful when you want to provide a custom name for the operation being
    profiled, rather than using the function name.

    Args:
        operation_name: Custom name for the operation being profiled

    Returns:
        Decorator function

    Example:
        >>> @log_execution_time("Patient Data Processing")
        ... def process_patient_123():
        ...     # Processing logic
        ...     pass
        >>> process_patient_123()
        # Logs: "PROFILE: Patient Data Processing completed in 0.45s"
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                logger.info(
                    f"PROFILE: {operation_name} completed in {duration:.2f}s"
                )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"PROFILE: {operation_name} failed after {duration:.2f}s: {e}"
                )
                raise

        return cast(F, wrapper)
    return decorator